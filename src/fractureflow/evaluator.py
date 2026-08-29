# -*- coding: utf-8 -*-
"""锁定口径评测器。

铁律: 所有指标 = 隐伏点逐点 acos(|<pred,true>|) 度 (MAE/P50/P90),
**绝不切换口径**。每次评测都同时给出:
  - 模型本身 (raw)
  - 几何基线 top2_ens (同一 test 切分、同一 rng999 掩码)
=> 一眼看出模型 vs 强几何基线, 防止"换了东西假装达标"。
"""

import numpy as np
import torch

from .core import DEVICE
from .data import collate, prepare_net
from .inference import top2_ens_dirs


def _errs_to_stats(errs, srcs):
    errs = torch.cat(errs)
    by_site = {}
    for e, s in zip(errs.tolist(), srcs):
        by_site.setdefault(s, []).append(e)
    site = {k: round(float(np.mean(v)), 2) for k, v in sorted(by_site.items())}
    return {
        "mae": round(float(errs.mean()), 2),
        "p50": round(float(errs.median()), 2),
        "p90": round(float(torch.quantile(errs, 0.90)), 2),
        "n_hid": int(errs.numel()),
        "per_site": site,
    }


def evaluate_model(model, te, obs_frac=0.4, device=DEVICE):
    """模型在 test 切分上的逐点 acos 误差 (官方协议 rng=999)。"""
    model.eval()
    rng = np.random.default_rng(999)
    errs, srcs, anis = [], [], []
    with torch.no_grad():
        for b in collate(te, obs_frac, rng, device):
            out = model(b)
            pred = out["pred"]
            mask = b["mask"]
            cos = (pred * b["nrm_full"]).sum(-1).abs().clamp(-1, 1)
            e = torch.rad2deg(torch.acos(cos))               # [B,L]
            hid = (1.0 - mask) > 0.5
            for bi in range(b["pos"].shape[0]):
                h = hid[bi]
                errs.append(e[bi][h].cpu())
                srcs.extend([b["src"][bi]] * int(h.sum()))
                if "aniso" in out:
                    anis.append(out["aniso"][bi][h].cpu())
    s = _errs_to_stats(errs, srcs)
    if anis:
        s["mean_aniso_hidden"] = round(float(torch.cat(anis).mean()), 3)
    return s


def evaluate_geometric_baseline(te, obs_frac=0.4, device=DEVICE):
    """几何基线 top2_ens 在同一 test 切分 / 同一 rng999 掩码下的误差 (苹果对苹果)。"""
    rng = np.random.default_rng(999)
    errs, srcs = [], []
    for net in te:
        b = prepare_net(net, obs_frac, rng)
        pos = b["pos"].numpy()
        nrm = b["nrm"].numpy()
        mask = b["mask"].numpy().astype(bool)
        nrm_full = b["nrm_full"].numpy()
        pred, _ = top2_ens_dirs(pos, nrm, mask)
        cos = np.abs((pred * nrm_full).sum(-1))
        e = np.rad2deg(np.arccos(np.clip(cos, -1, 1)))
        hid = ~mask
        errs.append(torch.as_tensor(e[hid], dtype=torch.float32))
        srcs.extend([str(net.get("src", ""))] * int(hid.sum()))
    return _errs_to_stats(errs, srcs)


def compare_report(model, te, cfg):
    """统一报告: 模型 + 几何基线 + delta。"""
    obs_frac = cfg.get("obs_frac", 0.4)
    m = evaluate_model(model, te, obs_frac)
    g = evaluate_geometric_baseline(te, obs_frac)
    return {
        "model": m,
        "geometric_baseline_top2_ens": g,
        "delta_model_minus_baseline": round(m["mae"] - g["mae"], 2),
    }
