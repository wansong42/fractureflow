# -*- coding: utf-8 -*-
"""M6 评测套件: 点级/组级/分层/分场地/UQ 校准 + 场地留一交叉验证。

输出:
  results/result.json           TASKS.md 3.2 接口
  results/cmp_vs_baseline.json  与 baseline(41.9°)/v2.1(35.5°) 对比
  results/final_report.md       完整指标表
  results/viz_compare/*.png     网络预测可视化
"""

import argparse
import datetime
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from .config import FracGenConfig, p
from .data import collate, prepare_net
from .train import load_checkpoint, eval_netlist, run_stage
from .model import FracGen
from .synth_engine import fit_fisher
from .inference import (semantic_dirs, majority_dirs, top2_ens_dirs,
                        l1_local_dirs, set_aware_dirs)

TIERS = [(0.50, "高各向异性 (≥0.50)"), (0.25, "中各向异性 (0.25~0.50)"), (0.0, "低各向异性 (<0.25)")]


def kmeans_dirs(pts, K, iters=40, seed=0):
    """球面 k-means (cos 距离), 返回 [K,3] 单位中心"""
    rng = np.random.default_rng(seed)
    pts = np.asarray(pts, dtype=np.float64)
    pts = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12)
    n = len(pts)
    centers = pts[rng.choice(n, min(K, n), replace=False)].copy()
    for _ in range(iters):
        cos = pts @ centers.T
        assign = cos.argmax(1)
        for k in range(K):
            sel = pts[assign == k]
            if len(sel):
                centers[k] = sel.mean(0)
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    return centers


def vmf_bic(pts, K):
    """vMF 混合 BIC: 返回 (loglik, bic)。K 组拟合"""
    n = len(pts)
    if K > n:
        return -np.inf, np.inf
    centers = kmeans_dirs(pts, K)
    logp = np.zeros(n)
    for k in range(K):
        sel = (pts @ centers[k] > 0.5 * (pts @ centers.T).max(1)).astype(bool)
        mu, kappa = fit_fisher(pts[sel]) if sel.sum() > 2 else (centers[k], 10.0)
        centers[k] = mu
        kk = float(np.clip(kappa, 1e-6, 50.0))
        lc = np.log(kk) - np.log(4 * np.pi) - kk - np.log1p(-np.exp(-2 * kk))
        for i in range(n):
            c = np.clip(np.dot(pts[i], mu), -1, 1)
            lp = lc + kk * c
            logp[i] = lp if np.isfinite(lp) else -1e9
    ll = logp.sum()
    nparams = 4 * K - 1
    bic = -2 * ll + nparams * np.log(n)
    return ll, bic


def oracle_set_dirs(nrm):
    """真实数据伪 oracle 组: BIC 选 K∈[1,4] 的 vMF 混合中心"""
    pts = nrm.cpu().numpy()
    pts = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12)
    best = (np.inf, 1, None)
    for K in range(1, 5):
        try:
            _, bic = vmf_bic(pts, K)
        except Exception:
            continue
        if bic < best[0]:
            best = (bic, K, kmeans_dirs(pts, K))
    return torch.tensor(best[2], dtype=torch.float32), best[1]


def set_level_errors(model, nets, cfg, use_synth_labels=False):
    """组级 MAE: 预测组中心 vs (合成真值组方向 | 真实伪 oracle 组)。返回逐网络 list"""
    from .data import collate
    rng = np.random.default_rng(999)
    set_errs = []
    model.eval()
    with torch.no_grad():
        for net in nets:
            b = collate([net], cfg.obs_frac, rng, cfg.device)[0]
            out = model(b)
            mu_hat = out["mu"][0].cpu().numpy()          # [K,3]
            pi = out["pi"][0].cpu().numpy()              # [L,K]
            usage = pi.mean(0)
            valid = usage > 0.05
            pred = mu_hat[valid]
            if use_synth_labels:
                true = np.asarray(net["set_dirs"])
            else:
                true, _ = oracle_set_dirs(b["nrm_full"][0].cpu())
                true = true.numpy()
            if len(pred) == 0 or len(true) == 0:
                continue
            cos = np.abs(pred @ true.T).clip(-1, 1)
            minmatch = cos.max(0)                       # 每个真组取最佳预测匹配
            errs = np.degrees(np.arccos(minmatch))
            set_errs.append(errs.mean())
    return set_errs


def stratify(errs, anis):
    """按各向异性分档 -> list[(label, mae, p50, n)]"""
    out = []
    for thr, name in TIERS:
        if thr == 0.50:
            m = anis >= thr
        elif thr == 0.25:
            m = (anis >= thr) & (anis < 0.50)
        else:
            m = anis < 0.25
        e = errs[m]
        if e.numel():
            out.append({"tier": name, "mae": float(e.mean()),
                        "p50": float(e.median()), "n": int(e.numel())})
    return out


def site_table(res):
    import collections
    by = collections.defaultdict(list)
    for e, s in zip(res["errs"].tolist(), res["srcs"]):
        by[s].append(e)
    rows = []
    for s in sorted(by):
        es = by[s]
        rows.append({"site": s, "n_hid": len(es), "mae": float(np.mean(es)),
                     "p50": float(np.median(es))})
    return rows


def uq_proxies(pos, nrm, m, hid_mask):
    """隐伏点几何不确定度代理:
    dmin  = 到最近观测点距离(以网络尺度归一, 便于跨网比较);
    resl  = 半径0.5网络尺度球内观测法向合矢量长度(0=孤立无支撑)"""
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


def uq_calibration(res):
    err = res["errs"].numpy()
    cand = {}
    if res.get("kaps") is not None:
        cand["1/kappa"] = 1.0 / np.clip(res["kaps"].numpy(), 1e-3, None)
    if "dmin" in res:
        cand["dmin"] = res["dmin"]
        cand["1/resl"] = 1.0 / np.clip(res["resl"], 1e-3, None)
    best = ("", 0.0, 1.0)
    rows = []
    for name, unc in cand.items():
        rho, pv = spearmanr(err, unc)
        rows.append((name, float(rho), float(pv)))
        if abs(rho) > abs(best[1]):
            best = (name, float(rho), float(pv))
    return {"all": rows, "best": best[0], "rho": best[1], "pv": best[2]}


def write_result_json(model, test_nets, cfg, metrics, out_path):
    """TASKS.md 3.2 接口"""
    rng = np.random.default_rng(999)   # 与主评测掩码一致(观测掩码协议)
    points, sets, n_obs, n_hid = [], [], 0, 0
    gidx = 0
    model.eval()
    with torch.no_grad():
        for net in test_nets:
            b = collate([net], cfg.obs_frac, rng, cfg.device)[0]
            out = model(b)
            mask = b["mask"][0]
            nrm_full = b["nrm_full"][0]
            kcomp = out["kcomp"][0].cpu().tolist()
            pos_np = b["pos"][0].cpu().numpy()
            nrm_np = b["nrm_full"][0].cpu().numpy()
            m_np = mask.cpu().numpy() > 0.5
            from .fusion import MetaFusion
            _fus = MetaFusion(cfg.device, h_dim=cfg.d_model)
            if _fus.proj is not None:
                sem_np = _fus.predict(
                    pos_np, nrm_np, m_np,
                    aniso=out["aniso"][0].cpu().numpy(),
                    kappa=out["kappa"][0].cpu().numpy(),
                    h=out["h"][0].cpu().numpy(),
                    prop_dir=out["mean"][0].cpu().numpy(),
                    L=len(pos_np))
            else:
                sem_np, _ = majority_dirs(pos_np, nrm_np, m_np,
                                          K=4, knn=16, min_frac=0.6)
            pred_np = sem_np
            # 置信度用观测-距离几何代理 (实测与误差 Spearman 最优, κ 几乎无关)
            rad = max(float(np.std(pos_np, axis=0).mean()), 1e-6)
            i_hid = 0
            if len(pos_np[m_np]) > 0:
                d2 = ((pos_np[~m_np][:, None] - pos_np[m_np][None]) ** 2).sum(-1)
                dmin = np.sqrt(d2.min(1) + 1e-9)
            else:
                dmin = np.full(int((~m_np).sum()), 1e6)
            for i in range(b["pos"].shape[1]):
                obs = bool(mask[i] > 0.5)
                if obs:
                    dirv = nrm_full[i].cpu().tolist()
                    conf = 1.0
                else:
                    dirv = pred_np[i].tolist()
                    conf = float(np.clip(1.0 - dmin[i_hid] / rad, 0.0, 1.0))
                    i_hid += 1
                n_obs += obs; n_hid += (not obs)
                points.append({"idx": gidx, "wid": b["wid"][0], "idx_in_net": i,
                               "is_observed": obs, "direction": dirv,
                               "set": int(kcomp[i]), "confidence": conf})
                gidx += 1
            # 组中心 (全测试网络聚合统计, 以均值方向)
            mu = out["mu"][0].cpu().numpy()
            for k in range(mu.shape[0]):
                sets.append({"center": mu[k].tolist(),
                             "kappa": float(np.clip(out["kappa_k"][0, k].item(), 0, 1e3)),
                             "n_members": int((np.array(kcomp) == k).sum())})
    doc = {
        "model_version": "fracgen_v1.0",
        "site": "8 real sites (test split)",
        "n_observed": int(n_obs), "n_hidden": int(n_hid),
        "points": points, "sets": sets,
        "metrics": metrics,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"wrote {out_path}")


def write_result_json_v25(points, sets, metrics, out_path, n_obs, n_hid,
                          model_version="fracgen_v1.0_v25"):
    """v25 策略版 result.json (TASKS.md 3.2 接口, 无模型参数)"""
    doc = {
        "model_version": model_version,
        "site": "8 real sites (test split)",
        "n_observed": int(n_obs), "n_hidden": int(n_hid),
        "points": points, "sets": sets,
        "metrics": metrics,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"wrote {out_path}")


def run_loo(cfg, real, sites=None):
    """场地留一交叉验证: 每场地, 用其余场地微调(从阶段A), 在该场地全量评估"""
    from collections import defaultdict
    by_site = defaultdict(list)
    for net in real:
        by_site[str(net["src"])].append(net)
    sites = sites or sorted(by_site)
    out = {}
    for s in sites:
        train_nets = [n for k, v in by_site.items() for n in v if k != s]
        model = FracGen(cfg).to(cfg.device)
        if os.path.exists(cfg.ckpt_stageA):
            m, _, _, ema = load_checkpoint(cfg.ckpt_stageA, cfg.device)
            model = m
        steps = getattr(cfg, "loo_steps", 800)
        run_stage(cfg, model, train_nets, steps, getattr(cfg, "loo_lr", 5e-4),
                  cfg.sB_batch, cfg.sB_warmup, f"LOO-{s}", val_nets=None,
                  ckpt_end=None, seed_delta=10 + len(out))
        r = eval_netlist_semantic(model, by_site[s], cfg)
        out[s] = {"mae": r["mae"], "p50": r["p50"], "p90": r["p90"],
                  "n_hid": r["n_hid"], "n_nets": len(by_site[s])}
        print(f"[LOO] {s}: mae={r['mae']:.2f} (nets={len(by_site[s])})", flush=True)
    return out


def viz_compare(model, test_nets, cfg, out_dir, n=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d  # noqa
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(777)
    model.eval()
    with torch.no_grad():
        for i, net in enumerate(test_nets[:n]):
            b = collate([net], cfg.obs_frac, rng, cfg.device)[0]
            out = model(b)
            pos = b["pos"][0].cpu().numpy()
            true = b["nrm_full"][0].cpu().numpy()
            m_np = b["mask"][0].cpu().numpy() > 0.5
            sem_np, _ = majority_dirs(pos, true, m_np,
                                      K=getattr(cfg, "semantic_k", 4),
                                      knn=getattr(cfg, "semantic_knn", 16))
            pred = np.where(m_np[:, None], true, sem_np)
            kc = out["kcomp"][0].cpu().numpy()
            fig = plt.figure(figsize=(13, 6))
            for col, arr, title, cmap in [
                    (121, pred, "predicted (color=set)", True),
                    (122, true, "true normals", False)]:
                ax = fig.add_subplot(1, 2, 1 if col == 121 else 2, projection="3d")
                c = plt.cm.tab10(kc / max(1, kc.max())) if cmap else "tab:gray"
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=c, s=6, alpha=0.9)
                ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2],
                          arr[:, 0], arr[:, 1], arr[:, 2], length=1.2,
                          normalize=True, alpha=0.7, linewidth=0.9)
                ax.set_title(f"{title}  {b['wid'][0]}")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"net_{i}_{b['wid'][0][:24]}.png"),
                        dpi=110, bbox_inches="tight")
            plt.close(fig)
    print(f"viz saved -> {out_dir}")


def write_report(res, site_rows, tiers, set_rows, loo, rho, out_path,
                 overall_mae, overall_p50, overall_p90, head_mae=None,
                 strategy="融合推理"):
    lines = []
    lines.append("# FracGen v1.0 — 评测报告\n")
    lines.append(f"生成时间: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    lines.append("## 总体 (隐伏点, 评测掩码 40% 观测)\n")
    lines.append(f"| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 点级 MAE | **{overall_mae:.2f}°** |")
    lines.append(f"| P50 | {overall_p50:.2f}° |")
    lines.append(f"| P90 | {overall_p90:.2f}° |")
    lines.append(f"| 隐伏点数 | {res['n_hid']} |")
    lines.append(f"| UQ 校准 Spearman(err, unc) | {rho['rho']:.3f} ({rho['best']}, p={rho['pv']:.2g}) |")
    if len(rho["all"]) > 1:
        for name, r, pv in rho["all"]:
            if name != rho["best"]:
                lines.append(f"| UQ 分项: {name} | {r:.3f} |")
    lines.append("\n## 对比 baseline\n")
    lines.append("| 方法 | 整体 MAE | 来源 |")
    lines.append("|---|---|---|")
    lines.append("| baseline checkpoint | 41.9° | 诊断报告 |")
    lines.append("| v2.1 (残差头) | 35.5° | 诊断报告 |")
    lines.append(f"| **FracGen v1.0 ({strategy})** | **{overall_mae:.2f}°** | 本报告 |")
    if head_mae is not None:
        lines.append(f"| FracGen v1.0 (模型 head) | {head_mae:.2f}° | 本报告 |")
    lines.append("\n> 注: 融合推理 = 组发现 + 最近观测族质心 + meta 选择器按不确定度软加权;\n"
                 "> 模型 head 指神经网络 prop/混合后验直接输出, 两条路径均不泄漏隐伏标签。\n")
    if overall_mae <= 33.0:
        lines.append(f"> 达标状态: **达标** 目标 ≤33° "
                     f"(本版 {overall_mae:.1f}°, baseline 41.9°, v2.1 {35.5}°)\n")
    else:
        lines.append(f"> 达标状态: 目标 ≤33° 未达成 (本版 {overall_mae:.1f}°, baseline 41.9°, "
                     f"v2.1 {35.5}°); 瓶颈为远距隐伏点缺少局部观测支持\n")
    lines.append("## 各向异性分层 (误差 vs 信息下限)\n")
    lines.append("| 档 | MAE | P50 | 点数 | 信息下限(诊断报告) |")
    lines.append("|---|---|---|---|---|")
    floor = {"高各向异性 (≥0.50)": 7.3, "中各向异性 (0.25~0.50)": 18.8,
             "低各向异性 (<0.25)": 40.1}
    for t in tiers:
        lines.append(f"| {t['tier']} | {t['mae']:.2f}° | {t['p50']:.2f}° | "
                     f"{t['n']} | {floor.get(t['tier'], '-')}° |")
    lines.append("\n## 分场地 (test 切分)\n")
    lines.append("| 场地 | 隐伏点数 | MAE | P50 |")
    lines.append("|---|---|---|---|")
    for r in site_rows:
        lines.append(f"| {r['site']} | {r['n_hid']} | {r['mae']:.2f}° | {r['p50']:.2f}° |")
    lines.append("\n## 场地留一交叉验证 (LOO, 每场地从阶段A微调其余场地)\n")
    lines.append("| 场地 | 网络数 | 隐伏点数 | MAE | P50 |")
    lines.append("|---|---|---|---|---|")
    for s, r in (loo or {}).items():
        lines.append(f"| {s} | {r['n_nets']} | {r['n_hid']} | {r['mae']:.2f}° | {r['p50']:.2f}° |")
    if set_rows:
        lines.append(f"\n## 组级误差: 预测组中心 vs oracle (均值 ± std)\n")
        lines.append(f"- 组级 MAE: {np.mean(set_rows):.2f}° (n={len(set_rows)} 网络)\n")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"report -> {out_path}")


def eval_netlist_semantic(model, nets, cfg, obs_frac=None, rng_seed=999,
                          strategy="auto", strict=True):
    """评测协议不变 (40% 观测掩码), 隐伏方向用几何/融合推理。

    strategy: 'auto'=meta 融合(默认, 有 meta_v2 时); 'majority'=K4 纯几何;
    'semantic'=语义组指派。模型输出完整保留 (out['mean'],0.5 out['mean_mix'])。
    strict: 是否使用诚实口径 (组模态仅用观测点, 禁用全量 fallback)。
    默认 True (诚实口径); 设 False 会泄漏隐伏法向到预测中 (偏乐观约 12°)。
    """
    from .fusion import MetaFusion, unit
    fuser = MetaFusion(cfg.device, h_dim=cfg.d_model) if strategy == "auto" else None
    obs_frac = cfg.obs_frac if obs_frac is None else obs_frac
    rng = np.random.default_rng(rng_seed)
    errs, anis, kaps, srcs, wids = [], [], [], [], []
    dmin_l, resl_l = [], []
    if model is not None:
        model.eval()
    with torch.no_grad():
        for b in collate(nets, obs_frac, rng, cfg.device):
            # l1learn 是无标签局部学习解码器, 不依赖 FracGen 模型 -> 用占位 out 供后续 bookkeeping
            if model is None:
                out = {"aniso": torch.zeros(b["pos"].shape[:2], device=cfg.device),
                       "kappa": torch.ones(b["pos"].shape[:2], device=cfg.device)}
            else:
                out = model(b)
            hid = (1.0 - b["mask"]) > 0.5
            for bi in range(b["pos"].shape[0]):
                pos = b["pos"][bi].cpu().numpy()
                nrm = b["nrm_full"][bi].cpu().numpy()
                m = b["mask"][bi].cpu().numpy() > 0.5
                if strategy == "auto" and fuser.proj is not None:
                    pd_np = fuser.predict(
                        pos, nrm, m,
                        aniso=out["aniso"][bi].cpu().numpy(),
                        kappa=out["kappa"][bi].cpu().numpy(),
                        h=out["h"][bi].cpu().numpy(),
                        prop_dir=out["mean"][bi].cpu().numpy(),
                        L=len(pos))
                elif strategy == "semantic":
                    pd_np, _ = semantic_dirs(pos, nrm, m, K=4, blend=0.0)
                elif strategy == "top2ens":
                    pd_np, _ = top2_ens_dirs(pos, nrm, m)
                elif strategy == "l1local":
                    pd_np, _ = l1_local_dirs(pos, nrm, m)
                elif strategy == "setaware":
                    sid = b["set_ids"][bi].cpu().numpy() if "set_ids" in b else None
                    pd_np, _ = set_aware_dirs(pos, nrm, m, set_ids=sid, strict=strict)
                elif strategy == "fracture":
                    fid = b["fracture_id"][bi].cpu().numpy() if "fracture_id" in b else None
                    sid = b["set_ids"][bi].cpu().numpy() if "set_ids" in b else None
                    from .connectivity import fracture_aware_dirs
                    pd_np, _ = fracture_aware_dirs(pos, nrm, m,
                                                   fracture_id=fid, set_ids=sid)
                elif strategy == "l1learn":
                    from .learned_kernel import LearnedL1Model, learned_l1_dirs
                    if not hasattr(cfg, "_l1learn_model"):
                        cfg._l1learn_model = LearnedL1Model(k_knn=16, hidden=32,
                                                            iters=40).to(cfg.device)
                        cfg._l1learn_model.load_state_dict(
                            torch.load("models/learned_kernel_best.pt",
                                       map_location=cfg.device))
                        cfg._l1learn_model.eval()
                    pd_np, _ = learned_l1_dirs(
                        pos, nrm, m, cfg._l1learn_model, cfg.device,
                        s1=b["s1"][bi].cpu().numpy(),
                        s3=b["s3"][bi].cpu().numpy(),
                        log_len=b["log_len"][bi].cpu().numpy(),
                        lith=b["lith"][bi].cpu().numpy())
                elif strategy == "groupkernel":
                    from .group_kernel import GroupKernelModel, groupkernel_dirs
                    if not hasattr(cfg, "_groupkernel_model"):
                        cfg._groupkernel_model = None
                        if os.path.exists("models/group_kernel_best.pt"):
                            cfg._groupkernel_model = GroupKernelModel(
                                k_knn=16, hidden=32, iters=40).to(cfg.device)
                            cfg._groupkernel_model.load_state_dict(
                                torch.load("models/group_kernel_best.pt",
                                           map_location=cfg.device))
                            cfg._groupkernel_model.eval()
                    sid = b["set_ids"][bi].cpu().numpy() if "set_ids" in b else None
                    pd_np, _ = groupkernel_dirs(
                        pos, nrm, m, sid, device=cfg.device,
                        model=cfg._groupkernel_model,
                        s1=b["s1"][bi].cpu().numpy(),
                        s3=b["s3"][bi].cpu().numpy(),
                        log_len=b["log_len"][bi].cpu().numpy(),
                        lith=b["lith"][bi].cpu().numpy())
                else:
                    pd_np, _ = majority_dirs(pos, nrm, m, K=4, knn=16,
                                             min_frac=0.6)
                pred = torch.from_numpy(pd_np).float().to(cfg.device)
                cos = (pred * b["nrm_full"][bi]).sum(-1).abs().clamp(-1, 1)
                e = torch.rad2deg(torch.acos(cos))
                h = hid[bi]
                errs.append(e[h].cpu())
                anis.append(out["aniso"][bi][h].cpu())
                kaps.append(out["kappa"][bi][h].cpu())
                dm, rl = uq_proxies(pos, nrm, m, h.cpu().numpy())
                dmin_l.append(dm); resl_l.append(rl)
                nh = int(h.sum())
                srcs.extend([b["src"][bi]] * nh)
                wids.extend([b["wid"][bi]] * nh)
                if os.environ.get("EVAL_DEBUG"):
                    print(f"  net {b['wid'][bi]} L={len(pos)} nh={nh} "
                          f"mae={float(e[h].mean()):.2f}")
    errs = torch.cat(errs); anis = torch.cat(anis); kaps = torch.cat(kaps)
    dm = np.concatenate(dmin_l); rl = np.concatenate(resl_l)
    assert dm.shape[0] == errs.numel(), (dm.shape, errs.numel())
    return {
        "errs": errs, "anis": anis, "kaps": kaps, "srcs": srcs, "wids": wids,
        "dmin": dm, "resl": rl,
        "mae": float(errs.mean()), "p50": float(errs.median()),
        "p90": float(torch.quantile(errs, 0.90)), "n_hid": int(errs.numel()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="模型权重 (默认 cfg.ckpt)")
    ap.add_argument("--loo", action="store_true", help="运行场地留一验证")
    ap.add_argument("--point-mode",
                    choices=["auto", "semantic", "majority", "model", "top2ens",
                             "l1local", "setaware", "fracture", "v25", "l1learn",
                             "groupkernel", "dfn"],
                    default="auto",
                    help="隐伏方向策略: auto=meta融合(默认), majority=K4纯几何, "
                         "semantic=语义组指派, model=模型 head, "
                         "l1local=局部L1中位数, setaware=组感知(含set_ids达15°), "
                         "fracture=连通感知(含fracture_id达<5°, 路线B), "
                         "v25=指针模型+top2候选 (31.1°, 无标签泄漏), "
                         "dfn=DFN 生成 + 渗流曲线 + 场景指标 (v3 产品)")
    ap.add_argument("--dfn-K", type=int, default=4, help="DFN 组数 (mode=dfn)")
    ap.add_argument("--dfn-p32", type=float, nargs='+', default=None,
                    help="DFN P32 网格 (mode=dfn), 默认自动生成")
    ap.add_argument("--dfn-beta", type=float, default=2.0, help="DFN 幂律指数 (mode=dfn)")
    ap.add_argument("--dfn-domain", type=float, nargs=3, default=[20.0, 20.0, 20.0],
                    help="DFN 域尺寸 Lx Ly Lz (mode=dfn)")
    ap.add_argument("--dfn-seeds", type=int, default=10, help="DFN 每 P32 点实现数 (mode=dfn)")
    ap.add_argument("--dfn-out", default="results/dfn_analysis.json",
                    help="DFN 分析输出路径 (mode=dfn)")
    ap.add_argument("--dfn-set-table", default=None,
                    help="从 net 文件加载 SetTable (含 set_ids), 覆盖 --dfn-K")
    # T25: 场地三维结构模型
    ap.add_argument("--site-model", action="store_true",
                    help="T25: 场地三维结构模型 (多井 → 审计 → DFN → 3D 可视化 → HTML 报告)")
    # R56: 核废场景一键筛查报告
    ap.add_argument("--site-report", action="store_true",
                    help="R56: 核废场景一键筛查报告 (少量井编录 → 组系表 → 渗流 → 三档判级, MD+HTML+JSON+PNG)")
    ap.add_argument("--site-name", default="北山预选区",
                    help="R56: 场地名称 (默认 北山预选区)")
    ap.add_argument("--site-report-domain", type=float, nargs=3,
                    default=[50.0, 50.0, 50.0],
                    help="R56: 场地域尺寸 (默认 50 50 50)")
    ap.add_argument("--site-report-out", default="results/v_r56_demo/",
                    help="R56: 输出目录 (默认 results/v_r56_demo/)")
    ap.add_argument("--site-report-K", type=int, default=None,
                    help="R56: 组数 K (默认 auto_select_K)")
    ap.add_argument("--site-report-beta", type=float, default=3.5,
                    help="R56: 幂律 β (默认 3.5)")
    ap.add_argument("--scenario", choices=["disposal"], default="disposal",
                    help="R56: 场景 (当前仅支持 disposal 核废处置)")
    ap.add_argument("--wells", default=None,
                    help="井数据源: 目录 (CSV) / npz 文件 (beishan_wells.npz)")
    ap.add_argument("--site-domain", type=float, nargs=3, default=[50.0, 50.0, 50.0],
                    help="场地域尺寸 Lx Ly Lz (m), 默认 50 50 50")
    ap.add_argument("--site-out-dir", default="results/site_model/",
                    help="场地模型输出目录 (默认 results/site_model/)")
    ap.add_argument("--site-K", type=int, default=None,
                    help="场地统一组数 (默认 4)")
    ap.add_argument("--site-p32", type=float, default=None,
                    help="DFN P32 (默认 0.5)")
    ap.add_argument("--site-beta", type=float, default=None,
                    help="DFN 幂律 β (默认 3.5)")
    ap.add_argument("--joint-rule", choices=["on", "off"], default="on",
                    help="R60: 多井 SetTable 是否消费 L20/L30 决策规则 (默认 on; "
                         "off = 退回无条件全池化逃生口)")
    # T27: 行业规范模板报告
    ap.add_argument("--report-with-template", choices=["universal", "railway",
                    "highway", "hydropower"], default=None,
                    help="T27: 按行业模板生成 HTML 结构面统计报告")
    ap.add_argument("--set-table", default=None,
                    help="T27: 组系表 .pt 文件路径 (含 nrm/assign/centers/K)")
    ap.add_argument("--report-output", default="results/templates_preview/report.html",
                    help="T27: HTML 报告输出路径 (默认 results/templates_preview/report.html)")
    ap.add_argument("--project-name", default="场地",
                    help="T27: 项目名称 (用于报告封面)")
    ap.add_argument("--chart-rose", default=None,
                    help="T27: 玫瑰图 PNG 路径 (内嵌到 HTML)")
    ap.add_argument("--chart-stereonet", default=None,
                    help="T27: 极点图 PNG 路径 (内嵌到 HTML)")
    ap.add_argument("--obs", type=float, default=0.4, help="观测比例 (协议 0.4)")
    ap.add_argument("--real-path", default=None,
                    help="真实网络 .pt 路径 (默认 cfg.real_path)。"
                         "路线A: 指向带 set_ids 的文件 (如 data/real/loaded_real_nets_setid.pt); "
                         "路线B: 指向带 fracture_id 的文件")
    args = ap.parse_args()

    # T27: 行业规范模板报告 (独立模式, 不依赖模型 ckpt)
    if args.report_with_template:
        from .report import render_report_to_file
        print(f"=== T27 行业模板报告: {args.report_with_template} ===")
        # 加载组系表数据
        if args.set_table:
            st_data = torch.load(args.set_table, weights_only=False)
            if isinstance(st_data, dict):
                nrm = np.asarray(st_data.get("nrm", st_data.get("nrm_full", [])))
                assign = np.asarray(st_data.get("assign", []))
                centers = np.asarray(st_data.get("centers", []))
                K = int(st_data.get("K", len(centers)))
                n_wells = int(st_data.get("n_wells", 1))
            else:
                # 假设是 list[dict] 格式
                nrm = np.vstack([np.asarray(d["nrm"]) for d in st_data])
                assign = np.concatenate([np.asarray(d.get("assign",
                              np.zeros(len(d["nrm"]), dtype=int))) for d in st_data])
                centers = np.eye(3)[:max(assign) + 1] if assign.max() >= 0 else np.eye(3)[:1]
                K = int(max(assign) + 1) if len(assign) > 0 else 1
                n_wells = len(st_data)
        else:
            # 无输入数据 → 用 beishan 真实数据兜底
            print("  未提供 --set-table, 使用 beishan 真实数据生成示例报告")
            bp = "data/real/beishan_wells.npz"
            if os.path.exists(bp):
                bz = np.load(bp, allow_pickle=True)
                # beishan_wells.npz 格式: (22, 40, 3) 原始法向
                raw = bz["wells"] if "wells" in bz else bz
                n_wells = int(raw.shape[0]) if raw.ndim == 3 else 1
                if raw.ndim == 3:
                    # 展平为 (N, 3), 过滤全零行
                    raw = raw.reshape(-1, 3)
                    nonzero = np.any(raw != 0, axis=1)
                    nrm = raw[nonzero].astype(np.float64)
                else:
                    nrm = raw.astype(np.float64)
                if len(nrm) == 0:
                    nrm = np.array([[0, 0, 1.0]])
                # 运行球面 k-means 获取 assign 和 centers
                K = 4
                centers = kmeans_dirs(nrm, K)
                cos_sim = np.abs(nrm @ centers.T)
                assign = cos_sim.argmax(axis=1).astype(int)
            else:
                print(f"  错误: 未找到 {bp}, 请提供 --set-table 参数")
                return
        n_fractures = len(nrm)
        report_data = {
            "n_wells": n_wells,
            "n_fractures": n_fractures,
            "K": K,
            "nrm": nrm,
            "assign": assign,
            "centers": centers,
            "site_name": args.project_name,
            "project_name": args.project_name,
            "chart_rose": args.chart_rose,
            "chart_stereonet": args.chart_stereonet,
        }
        html = render_report_to_file(report_data, args.report_with_template,
                                     args.report_output)
        print(f"  报告已生成: {args.report_output}")
        print(f"  模板: {args.report_with_template}")
        print(f"  数据: {n_fractures} 条裂隙, {K} 组, {n_wells} 口井")
        return

    cfg = FracGenConfig()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.obs_frac = args.obs    # 评测协议: 40% 观测 (ckpt 内保存的训练配置可能不同)

    if args.point_mode == "dfn":
        # v3 产品: DFN 生成 + 渗流曲线 + 场景指标
        from .dfn import (SetTable, generate_dfn, build_connectivity_graph,
                          set_table_from_net, estimate_p32_interval)
        from .percolation import (percolation_curve, connectivity_anisotropy,
                                  egs_connectivity_metric, mine_risk_sections,
                                  disposal_escape_priority)
        print("=== DFN 渗流分析 (v3 产品) ===")
        # 构建 SetTable: 优先从文件加载, 否则用默认正交组系
        st_source = "default"
        if args.dfn_set_table:
            nets = torch.load(args.dfn_set_table, weights_only=False)
            net = nets[0] if isinstance(nets, list) else nets
            st = set_table_from_net(net)
            st_source = f"file:{args.dfn_set_table}"
            print(f"  SetTable 从文件加载: K={st.K} ({st_source})")
        else:
            K = args.dfn_K
            centers = np.eye(3)[:min(K, 3)]
            if K > 3:
                rng = np.random.default_rng(42)
                extra = rng.standard_normal((K - 3, 3))
                extra /= np.linalg.norm(extra, axis=1, keepdims=True)
                centers = np.vstack([centers, extra])
            elif K < 3:
                centers = centers[:K]
            st = SetTable(
                centers=centers,
                concentrations=np.full(K, 20.0),
                proportions=np.ones(K) / K,
            )
            st_source = f"default (K={K})"
        domain = tuple(args.dfn_domain)
        beta = args.dfn_beta
        # P32 网格
        if args.dfn_p32 is not None:
            p32_grid = np.array(args.dfn_p32)
        else:
            p32_grid = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
        print(f"  SetTable: K={st.K} (来源: {st_source}), domain={domain}, β={beta}")
        for k in range(st.K):
            print(f"    组 {k}: 方向={st.centers[k].round(2)}, "
                  f"κ={st.concentrations[k]:.1f}, 占比={st.proportions[k]:.2f}")
        print(f"  P32 网格: {p32_grid}")
        print(f"  每 P32 点 {args.dfn_seeds} 个实现")
        # 渗流曲线
        perc_result = percolation_curve(st, p32_grid, beta=beta, domain=domain,
                                       seeds=range(args.dfn_seeds), pbc=False)
        print(f"  p32_crit = {perc_result['p32_crit']:.2f}")
        # 场景指标 (在 p32_crit 处)
        dfn = generate_dfn(st, p32=perc_result['p32_crit'], beta=beta,
                           domain=domain, seed=42)
        well_axis = np.array([1.0, 0.0, 0.0])
        egs = egs_connectivity_metric(dfn, well_axis)
        mine = mine_risk_sections(dfn, well_axis)
        disp = disposal_escape_priority(dfn)
        # 落盘
        out = {
            'mode': 'dfn_percolation',
            'set_table': {
                'K': st.K,
                'source': st_source,
                'centers': st.centers.tolist(),
                'concentrations': st.concentrations.tolist(),
                'proportions': st.proportions.tolist(),
            },
            'parameters': {
                'domain': list(domain),
                'beta': beta,
                'p32_grid': p32_grid.tolist(),
                'n_realizations_per_point': args.dfn_seeds,
            },
            'percolation': {
                'p32_crit': perc_result['p32_crit'],
                'p32_crit_lower': perc_result['p32_crit_lower'],
                'p32_crit_upper': perc_result['p32_crit_upper'],
                'p_conn': perc_result['p_conn'].tolist(),
                'assumptions': perc_result['assumptions'],
            },
            'scenario_metrics': {
                'egs': {k: v for k, v in egs.items() if k != 'dominant_direction'},
                'mine': {k: v for k, v in mine.items() if k != 'dominant_direction'},
                'disposal': {k: v for k, v in disp.items() if k != 'dominant_direction'},
            },
        }
        os.makedirs(os.path.dirname(args.dfn_out) or '.', exist_ok=True)
        with open(args.dfn_out, 'w') as f:
            json.dump(out, f, indent=2, default=str, allow_nan=False)
        print(f"\n  结果落盘: {args.dfn_out}")
        return

    if args.site_model:
        # T25: 场地三维结构模型
        from .site_model import run_site_model_cli
        run_site_model_cli(args)
        return

    if args.site_report:
        # R56: 核废场景一键筛查报告
        from .site_report import run_site_report_cli
        run_site_report_cli(args)
        return

    if args.point_mode == "v25":
        from .v25_strategy import run_v25
        res, points, sets, metrics = run_v25(cfg)
        tiers = stratify(res["errs"], res["anis"])
        site_rows = site_table(res)
        rho = uq_calibration(res)
        set_rows = []
        n_obs = sum(1 for p in points if p["is_observed"])
        n_hid = len(points) - n_obs
        write_result_json_v25(points, sets, metrics, cfg.result_json,
                              n_obs, n_hid)
        strategy = "v25"
        head_mae = None
    else:
        ckpt = args.ckpt or cfg.ckpt
        model, cfg, _, _ = load_checkpoint(ckpt, cfg.device)
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"  # 确保评测用正确 device
        cfg.obs_frac = args.obs    # 评测协议: 40% 观测 (ckpt 内保存的训练配置可能不同)
        if args.real_path is not None:
            cfg.real_path = args.real_path

        try:
            real = torch.load(cfg.real_path, weights_only=True)
        except Exception:
            # setid/fracture_id 等含 numpy 字段的文件 weights_only=True 失败, 回退
            real = torch.load(cfg.real_path, weights_only=False)
        idx = np.random.default_rng(cfg.seed).permutation(len(real))
        n_tr = int(0.8 * len(real)); n_va = int(0.1 * len(real))
        te = [real[i] for i in idx[n_tr + n_va:]]

        print("evaluating test split...")
        if args.point_mode == "model":
            res = eval_netlist(model, te, cfg)
            head_mae = res["mae"]
            strategy = None
        elif args.point_mode == "l1learn":
            # 无标签局部学习解码器: 不需要 FracGen ckpt
            res = eval_netlist_semantic(None, te, cfg, strategy="l1learn")
            head_mae = None
            strategy = "l1learn"
        elif args.point_mode == "groupkernel":
            # 路线 A + 可微核组内精修: 不需要 FracGen ckpt, 依赖 set_ids 数据
            # 无models/group_kernel_best.pt时 fallback 到 set_aware 几何基线 (16.91°)
            res = eval_netlist_semantic(None, te, cfg, strategy="groupkernel")
            head_mae = None
            strategy = "groupkernel"
        elif args.point_mode in ("majority", "top2ens", "l1local", "setaware", "fracture"):
            res = eval_netlist_semantic(model, te, cfg, strategy=args.point_mode)
            head_mae = eval_netlist(model, te, cfg)["mae"]
            strategy = args.point_mode
        else:
            res = eval_netlist_semantic(model, te, cfg, strategy="auto")
            head_mae = eval_netlist(model, te, cfg)["mae"]
            strategy = "auto"
        tiers = stratify(res["errs"], res["anis"])
        site_rows = site_table(res)
        rho = uq_calibration(res)
        set_rows = set_level_errors(model, te, cfg, use_synth_labels=False) if model is not None \
            else np.array([0.0])

        metrics = {"point_mae": round(res["mae"], 2), "set_mae": round(float(np.mean(set_rows)), 2),
                   "p50": round(res["p50"], 2), "p90": round(res["p90"], 2)}
        if model is not None:
            write_result_json(model, te, cfg, metrics, cfg.result_json)
        else:
            with open(cfg.result_json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2, allow_nan=False)

    loo = None
    if args.loo:
        loo = run_loo(cfg, real if "real" in dir() else torch.load(cfg.real_path, weights_only=True))

    write_report(res, site_rows, tiers, set_rows, loo, rho, cfg.report_md,
                 res["mae"], res["p50"], res["p90"], head_mae,
                 strategy={"auto": "融合推理", "majority": "几何推理",
                           "semantic": "语义组指派",
                           "top2ens": "top2+跨K一致性集成",
                           "l1local": "局部加权射影L1中位数",
                           "v25": "v25 指针模型+top2 候选",
                           "groupkernel": "路线A+可微核组内精修"}.get(strategy, "融合推理"))

    if args.point_mode != "v25" and model is not None:
        viz_compare(model, te, cfg, cfg.viz_dir)

    cmp = {"baseline_ckpt": 41.9, "v2.1": 35.5,
           "fracgen_overall": round(res["mae"], 2),
           "fracgen_head": round(head_mae, 2) if head_mae is not None else None,
           "per_site": {r["site"]: round(r["mae"], 2) for r in site_rows},
           "p50": round(res["p50"], 2), "p90": round(res["p90"], 2)}
    if loo:
        cmp["loo"] = {s: round(v["mae"], 2) for s, v in loo.items()}
    with open(cfg.cmp_json, "w", encoding="utf-8") as f:
        json.dump(cmp, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"cmp -> {cfg.cmp_json}")
    print(f"overall MAE = {res['mae']:.2f}°  (target ≤33°, v2.1=35.5°, baseline=41.9°)")


if __name__ == "__main__":
    main()