# -*- coding: utf-8 -*-
"""v25 官方策略: 指针模型 (v23+top2 候选) 无泄漏推理集成。

评测入口: python -m fractureflow.eval --point-mode v25  (在 src/ 下执行)
协议: seed42 80/10/10 切分 + rng999 40% 观测掩码 (与 eval.py 完全一致)。
策略: v23 Ptr 指针模型, 15 候选 (6 本网模式 + 6 场地模式 + KWAM + 最近邻
       + 官方 top2ens 方向), 3 种子集成, GRIDF 期望角代价解码。

全程无泄漏: 模型只在官方 train(312 网)上训练; 官方 val 仅用于早停;
test 掩码下的观测只提供几何候选/代价, 标签仅用于最终度量。
test MAE ≈ 31.1° (top2ens 参照 34.68°)。
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

# The v25 strategy depends on internal research-line modules (scripts/v25_official.py,
# scripts/h1_harness.py) that are not part of this release.  The import is guarded so
# that the package imports cleanly; run_v25() raises an informative error at call time.
try:
    import v25_official as V25  # noqa: E402
    from h1_harness import prep  # noqa: E402
    _V25_AVAILABLE = True
except ImportError:
    V25 = None
    prep = None
    _V25_AVAILABLE = False


def _uq_proxies(pos, nrm, m, hid_mask):
    """隐伏点几何不确定度代理 (与 eval.uq_proxies 同款)"""
    from scipy.spatial import cKDTree
    idx_h = np.where(hid_mask)[0]
    pos_o = pos[m]
    rad = max(float(pos.std(axis=0).mean()), 1e-6)
    tre_o = cKDTree(pos_o) if len(pos_o) else None
    tre_a = cKDTree(pos)
    if tre_o is not None and len(idx_h):
        d, _ = tre_o.query(pos[idx_h], k=1)
        dm = d / rad
    else:
        dm = np.full(len(idx_h), 1e9)
    rl = np.zeros(len(idx_h))
    for j, i in enumerate(idx_h):
        so = [k for k in tre_a.query_ball_point(pos[i], 0.5 * rad) if m[k]]
        if so:
            rl[j] = np.linalg.norm(np.mean(nrm[so], axis=0))
    return dm, rl


def run_v25(cfg):
    """运行 v25 策略, 返回 (res, points, sets, metrics)。

    res     : 与 eval_netlist_semantic 同构 (errs/anis/srcs/wids/dmin/resl/...)
    points  : result.json 的 points 列表
    sets    : result.json 的 sets 列表
    metrics : 指标 dict
    """
    if not _V25_AVAILABLE:
        raise RuntimeError(
            "point-mode v25 requires internal research modules "
            "(scripts/v25_official.py, scripts/h1_harness.py) that are not "
            "distributed with this release."
        )
    args = V25.types.SimpleNamespace(
        nseed=3, train_seeds=24, epochs=60, patience=12, hidden=384,
        dropout=0.25, lr=1e-3, wd=3e-4, temp=0.12,
        nsub=4, sub_frac=0.6, min_pv=12, step=4, ws_seeds=20,
        site_w=True, use_val=False, no_top2k=False, cand15=True,
        refine=False, gridf=2048, skip_pseudo=True)
    out = V25.run_full(args)
    res = out["res"]
    TE = out["_te"]
    tmasks = out["_tmasks"]

    errs = torch.from_numpy(res["errs"]).float()
    dmin_l, resl_l = [], []
    for i, pn in enumerate(res["per_net"]):
        pos, nrm, meta = TE[pn["net_i"]]
        dm, rl = _uq_proxies(pos, nrm, pn["mask"], ~pn["mask"])
        dmin_l.append(dm)
        resl_l.append(rl)
    dmin = np.concatenate(dmin_l)
    resl = np.concatenate(resl_l)
    assert dmin.shape[0] == errs.numel(), (dmin.shape, errs.numel())

    res = {
        "errs": errs, "anis": torch.from_numpy(resl.astype(np.float32)),
        "kaps": None, "srcs": list(res["srcs"]), "wids": list(res["wids"]),
        "dmin": dmin, "resl": resl,
        "mae": float(errs.mean()), "p50": float(errs.median()),
        "p90": float(torch.quantile(errs, 0.90)), "n_hid": int(errs.numel()),
    }
    res["per_net"] = out["res"]["per_net"]
    res["by_net"] = out["by_net"]

    # ---- result.json points/sets ----
    points, sets = [], []
    gidx = 0
    for i, pn in enumerate(res["per_net"]):
        pos, nrm, meta = TE[pn["net_i"]]
        m = pn["mask"]
        hid = ~m
        dirs = pn["dirs"]                      # 隐伏预测 [nh,3]
        truth = nrm                            # 全点真值 (观测点直接用)
        rad = max(float(pos.std(axis=0).mean()), 1e-6)
        dmin_i = dmin_l[i]
        labels = pn["labels"]
        cand = pn["cand_dirs"]                 # [15,3]
        L = len(pos)
        hh = np.where(hid)[0]
        for j in range(L):
            if m[j]:
                dirv = truth[j].tolist()
                conf = 1.0
                setv = 0
            else:
                row = int(np.searchsorted(hh, j))
                dirv = dirs[row].tolist()
                conf = float(np.clip(1.0 - dmin_i[row] / rad, 0.0, 1.0))
                setv = int(labels[row])
            points.append({"idx": gidx, "wid": meta["wid"], "idx_in_net": j,
                           "is_observed": bool(m[j]), "direction": dirv,
                           "set": setv, "confidence": conf})
            gidx += 1
        cnt = np.bincount(labels, minlength=len(cand))
        for j in range(len(cand)):
            if cnt[j]:
                sets.append({"center": cand[j].tolist(), "kappa": 0.0,
                             "n_members": int(cnt[j]),
                             "wid": meta["wid"]})
    metrics = {"point_mae": round(res["mae"], 2),
               "p50": round(res["p50"], 2), "p90": round(res["p90"], 2)}
    return res, points, sets, metrics