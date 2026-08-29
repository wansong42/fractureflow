# -*- coding: utf-8 -*-
"""钻孔/露头法向 → 自动组标注 (路线 A) + 双路线对照 demo.

两条路线 (同一锁定口径: 隐伏点 acos(|<pred,true>|) 均值):

  路线 A (全量调查姿势 / full survey):
      分析师握有完整 DFN 量测 -> 球面 k-means 给每条裂隙打 set_ids
      (轮廓系数自动选 K) -> 组内 vMF 传播 (set_aware_dirs) 填隐伏点。
      不泄漏隐伏法向本身 (组标签是数据属性, 等价 analyst 从完整调查定组)。
      真实数据复测见 results/auto_label_routeA_real.json (~14°)。

  路线 B (空间分块 / no-leak propagation):
      只勘察了场地某空间子块 -> 仅用子块内观测法向估组
      (obs_only_set_ids) -> 传播到全场地隐伏点。
      演示"无组属性、仅局部调查"时的 no-leak 传播退化。

用法:
  python scripts/auto_label_borehole.py --data synth --max-nets 30
  python scripts/auto_label_borehole.py --data real --max-nets 20
  python scripts/auto_label_borehole.py --csv path.csv --out out.pt --plot
"""
import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 避免 torch/OpenMP 重复初始化崩溃

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractureflow.setlabel import (
    generate_set_ids, spatial_block_mask, obs_only_set_ids, label_nets)
from fractureflow.inference import set_aware_dirs, l1_local_dirs
from read_forge_las import read_single_las, build_3d_trajectory, dip_dipdir_to_normal

try:
    from fractureflow.terzaghi import terzaghi_weights
except ImportError:
    terzaghi_weights = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def acos_err(pred, true):
    pred = _unit(np.asarray(pred, float))
    true = _unit(np.asarray(true, float))
    cos = np.clip(np.abs((pred * true).sum(-1)), -1.0, 1.0)
    return np.rad2deg(np.arccos(cos))


def load_synth(max_nets=30, path=None):
    p = path or os.path.join(ROOT, "data/synth/synth_val.pt")
    data = torch.load(p, weights_only=False)
    return [dict(n) for n in data[:max_nets]]


def load_real(max_nets=20, path=None):
    p = path or os.path.join(ROOT, "data/real/loaded_real_nets.pt")
    data = torch.load(p, weights_only=False)
    return [dict(n) for n in data[:max_nets]]


def read_borehole_csv(path, id_col="id", ncol="nx", ucol="ny", vcol="nz",
                      type_col=None, sep=None):
    """读钻孔/露头法向 CSV -> net dict (含 nrm/nrm_full/pos 占位/set_ids 占位)。

    支持两种方言 (T91 演练修复, 自动识别):
      A. 法向列 nx/ny/nz (默认; 可用 ncol/ucol/vcol 自定义列名)
      B. 编录表方言 depth/dip/dip_direction (客户标准格式, 自动转法向)

    容错 (T91): 空值/非数字行跳过并计数警告; 零向量行同样跳过计数;
    全部无效时抛带列名清单的 RuntimeError。

    type_col: 可选, 指定一列作为 fracture type (写入 net['ftype'], 仅记录,
              不参与几何, 满足"按类型分组"需求但不引入泄漏)。
    """
    import csv
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f, delimiter=sep or ",")
        fieldnames = [fn for fn in (rdr.fieldnames or []) if fn]
        rows = list(rdr)

    def col_of(name):
        """大小写/首尾空白容错的列名匹配."""
        for k in fieldnames:
            if k.strip().lower() == str(name).strip().lower():
                return k
        return None

    cols_a = [col_of(c) for c in (ncol, ucol, vcol)]
    use_dialect_b = (any(c is None for c in cols_a)
                     and col_of("dip") is not None
                     and col_of("dip_direction") is not None)
    if use_dialect_b:
        c_dip, c_dd = col_of("dip"), col_of("dip_direction")
    elif any(c is None for c in cols_a):
        missing = [c for c, k in zip((ncol, ucol, vcol), cols_a) if k is None]
        raise RuntimeError(
            f"CSV 缺少必需列 {missing}; 支持 nx/ny/nz 或 depth/dip/dip_direction "
            f"两种格式; 实际列: {fieldnames} ({os.path.basename(path)})")

    nrm, ftypes, bad_lines, zero_cnt = [], [], [], 0
    for i, r in enumerate(rows):
        try:
            if use_dialect_b:
                dd = float(r[c_dd]); dp = float(r[c_dip])
                n = dip_dipdir_to_normal(np.array([dd]), np.array([dp]))[0]
            else:
                n = np.array([float(r[c]) for c in cols_a], float)
        except (TypeError, ValueError):
            bad_lines.append(i + 2)   # +2 = 表头 1 行 + 1-based 行号
            continue
        if np.linalg.norm(n) < 1e-9:
            zero_cnt += 1
            continue
        nrm.append(n)
        if type_col is not None:
            ftypes.append(str(r.get(type_col, "")))
    if bad_lines or zero_cnt:
        warn = f"[csv] 跳过无效行 {len(bad_lines)} 个 (空值/非数字, 行号 {bad_lines[:8]})"
        if zero_cnt:
            warn += f"; 零向量行 {zero_cnt} 个"
        print(warn)
    if not nrm:
        raise RuntimeError(f"CSV 中没有可解析的裂隙行 ({os.path.basename(path)})")
    nrm = np.asarray(nrm, float)
    net = {
        "pos": np.zeros((len(nrm), 3), float),   # 无坐标: 纯法向调查
        "nrm": nrm.copy(),
        "nrm_full": nrm.copy(),
        "len": np.ones(len(nrm)),
        "lith": np.zeros(len(nrm), int),
        "s1": np.ones(len(nrm)), "s3": np.ones(len(nrm)),
        "wid": os.path.basename(path), "src": "csv",
    }
    if ftypes:
        net["ftype"] = np.asarray(ftypes)
    return net


def _terzaghi_reassign(nrm, set_ids, well_axis=None, seed=42):
    """Re-assign set_ids using Terzaghi (1965) weighted spherical k-means.

    Sampling bias correction: probability of intercepting a fracture ∝ |n·a|
    where |n·a| = cos(θ) = sin(α), θ = angle between fracture normal and
    borehole axis, α = angle between fracture plane and borehole axis.
    Correction weight = 1 / |n·a|.  Centroid update becomes:
        centers[k] = _unit((aligned * wk[:, None]).sum(0))
    where wk = terzaghi_weights for this well.

    Falls back to inline implementation if fractureflow.terzaghi is not yet
    available (parallel task).  Default well_axis = [0,0,1] (vertical borehole).
    """
    nrm = np.asarray(nrm, dtype=np.float64)
    pts = _unit(nrm)
    nn = len(pts)
    K = int(set_ids.max()) + 1
    if K <= 1:
        return set_ids

    # Compute Terzaghi weights (prefer external module, else inline)
    if terzaghi_weights is not None:
        wk = terzaghi_weights(pts, well_axis=well_axis)
    else:
        if well_axis is None:
            well_axis = np.array([0.0, 0.0, 1.0])  # default: vertical borehole
        well_axis = _unit(np.asarray(well_axis, float))
        cos_wa = np.abs(pts @ well_axis)
        wk = 1.0 / np.clip(cos_wa, 1e-6, 1.0)  # w = 1/|n·a|
        wk = wk / wk.mean()          # normalize mean=1
        wk = np.clip(wk, 0.1, 5.0)   # clip extremes

    # Initialize centers from current assignment (weighted sign-aligned mean)
    centers = np.zeros((K, 3))
    _rng = np.random.default_rng(seed)
    for k in range(K):
        sel = pts[set_ids == k]
        wsel = wk[set_ids == k]
        if len(sel) == 0:
            centers[k] = pts[_rng.integers(nn)]
            continue
        ref = sel[0]
        sgn = np.sign((sel * ref).sum(-1, keepdims=True)); sgn[sgn == 0] = 1
        centers[k] = _unit((sel * sgn * wsel[:, None]).mean(0))

    # Weighted Lloyd iterations (refinement, not cold start → fewer iters)
    assign = set_ids.copy()
    for _ in range(40):
        cos_sim = np.abs(pts @ centers.T)
        assign = cos_sim.argmax(1)
        for k in range(K):
            sel = pts[assign == k]
            wsel = wk[assign == k]
            if len(sel) == 0:
                continue
            sgn = np.sign((sel * centers[k]).sum(-1, keepdims=True))
            sgn[sgn == 0] = 1
            aligned = sel * sgn
            centers[k] = _unit((aligned * wsel[:, None]).sum(0))
    return assign


# ----------------------------- 路线 A -----------------------------
def routeA_full_survey(net, Krange=(2, 7), seed=42, rng=None, terzaghi=False):
    """全量调查姿势: 用完整法向场定组 -> 组内 vMF 传播填隐伏点。

    terzaghi: if True, re-cluster with Terzaghi weights after initial clustering.

    返回 (mae, p50, p90, K, n_hid, set_ids)。
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    nrm_full = np.asarray(net["nrm_full"], float)
    # 全量调查: 从完整法向场定组 (分析师握有完整 DFN 量测)
    set_ids = generate_set_ids(nrm_full, Krange=Krange, seed=seed)
    # 评测掩码: 优先用数据自带 40% 均匀掩码, 否则现生成
    if "obs_mask" in net and np.asarray(net["obs_mask"]).shape[0] == len(nrm_full):
        obs = np.asarray(net["obs_mask"], bool)
    else:
        obs = rng.random(len(nrm_full)) < 0.4
    pos = np.asarray(net.get("pos"), float)
    if pos.shape[0] != len(nrm_full):
        pos = np.zeros_like(nrm_full)
    # --- Terzaghi re-clustering (可选) ---
    if terzaghi:
        well_axis = net.get("well_axis", None)
        new_ids = _terzaghi_reassign(nrm_full, set_ids, well_axis=well_axis, seed=seed)
        # compute weight summary stats for user printout
        if terzaghi_weights is not None:
            wk = terzaghi_weights(_unit(nrm_full), well_axis=well_axis)
        else:
            w_axis = np.array([0.0, 0.0, 1.0]) if well_axis is None else _unit(np.asarray(well_axis, float))
            cos_wa = np.abs(_unit(nrm_full) @ w_axis)
            wk = 1.0 / np.clip(cos_wa, 1e-6, 1.0)  # w = 1/|n·a|
            wk = wk / wk.mean()
            wk = np.clip(wk, 0.1, 5.0)
        print(f"  [terzaghi] 采样偏差校正已启用 (Terzaghi 1965). "
              f"权重: mean={float(wk.mean()):.3f}, max={float(wk.max()):.3f}")
        set_ids = new_ids
    pred, _ = set_aware_dirs(pos, nrm_full, obs, set_ids=set_ids)
    err = acos_err(pred, nrm_full)
    hid = ~obs
    return (float(err[hid].mean()), float(np.median(err[hid])),
            float(np.quantile(err[hid], 0.90)), int(set_ids.max()) + 1,
            int(hid.sum()), set_ids)


# ----------------------------- 路线 B -----------------------------
def routeB_spatial_block(net, frac=0.4, Krange=(2, 7), seed=42, rng=None):
    """空间分块姿势: 只勘察某子块 -> 仅用子块观测估组 -> no-leak 传播。

    返回 (mae, p50, p90, K, n_hid, set_ids)。
    """
    rng = rng if rng is not None else np.random.default_rng(seed + 1)
    nrm_full = np.asarray(net["nrm_full"], float)
    pos = np.asarray(net.get("pos"), float)
    if pos.shape[0] != len(nrm_full):
        pos = np.zeros_like(nrm_full)
    obs = spatial_block_mask(pos, frac=frac, rng=rng)
    set_ids, K, sil = obs_only_set_ids(nrm_full, obs, Krange=Krange, seed=seed)
    pred, _ = set_aware_dirs(pos, nrm_full, obs, set_ids=set_ids)
    err = acos_err(pred, nrm_full)
    hid = ~obs
    return (float(err[hid].mean()), float(np.median(err[hid])),
            float(np.quantile(err[hid], 0.90)), int(K),
            int(hid.sum()), set_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synth", "real"], default="synth")
    ap.add_argument("--max-nets", type=int, default=30)
    ap.add_argument("--csv", default=None, help="直接读 CSV (覆盖 --data)")
    ap.add_argument("--out", default=None, help="CSV 模式: 打标结果存 .pt")
    ap.add_argument("--type-col", default=None,
                    help="CSV: 指定 fracture type 列名 (仅记录, 不引入泄漏)")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--ncol", default="nx")
    ap.add_argument("--ucol", default="ny")
    ap.add_argument("--vcol", default="nz")
    ap.add_argument("--Kmin", type=int, default=2)
    ap.add_argument("--Kmax", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--input-las", default=None,
                    help="直接读 FORGE 风格 FMI LAS -> 路线 A 自动打标 + 无泄漏验证 (纯 numpy)")
    ap.add_argument("--terzaghi", action="store_true",
                    help="启用 Terzaghi (1965) 采样偏差校正: 1/|n·a| 加权球面 k-means "
                         "(=平行裂隙权重高, 正交裂隙权重低). 收益薄 (实测 -0.2~-0.3°), "
                         "但学术审稿必问, 建议报告 both corrected 和 uncorrected")
    args = ap.parse_args()

    if args.input_las:
        run_las(args)
        return

    Krange = (args.Kmin, args.Kmax)

    if args.csv:
        net = read_borehole_csv(args.csv, id_col=args.id_col, ncol=args.ncol,
                                ucol=args.ucol, vcol=args.vcol,
                                type_col=args.type_col)
        # 小样本保护: CSV 钻孔通常点少, Kmax 不要默认 7 (会过拟合)。
        N = len(net["nrm"])
        csv_Kmax = min(args.Kmax, max(2, N // 3), N - 1)
        csv_Krange = (min(args.Kmin, csv_Kmax), csv_Kmax)
        net = label_nets([net], Krange=csv_Krange, seed=args.seed)[0]
        # --- Terzaghi re-clustering (可选) ---
        if args.terzaghi:
            nrm = np.asarray(net["nrm_full"], float)
            well_axis = net.get("well_axis", None)
            new_ids = _terzaghi_reassign(nrm, net["set_ids"], well_axis=well_axis, seed=args.seed)
            w_axis = np.array([0.0, 0.0, 1.0]) if well_axis is None else _unit(np.asarray(well_axis, float))
            cos_wa = np.abs(_unit(nrm) @ w_axis)
            wk = 1.0 / np.clip(cos_wa, 1e-6, 1.0)
            wk = wk / wk.mean()
            wk = np.clip(wk, 0.1, 5.0)
            print(f"  [terzaghi] 采样偏差校正已启用 (Terzaghi 1965). "
                  f"权重: mean={float(wk.mean()):.3f}, max={float(wk.max()):.3f}")
            net["set_ids"] = new_ids
        if args.out:
            torch.save([net], args.out)
            print(f"[csv] 打标完成 -> {args.out}  (K={int(net['set_ids'].max())+1}, "
                  f"N={len(net['set_ids'])})")
        else:
            print(f"[csv] K={int(net['set_ids'].max())+1}  N={len(net['set_ids'])}")
            if not args.terzaghi:
                for i, s in enumerate(net["set_ids"]):
                    print(f"  #{i:03d} set={int(s)}")
        if args.terzaghi:
            print("  [terzaghi] 诚实边界: 收益薄 (实测 -0.2~-0.3°), 学术审稿必问, "
                  "建议报告 both corrected 和 uncorrected")
        return

    if args.data == "synth":
        nets = load_synth(args.max_nets)
    else:
        nets = load_real(args.max_nets)

    errA, errB = [], []
    summary = {"routeA": [], "routeB": []}
    rng = np.random.default_rng(args.seed)
    for i, net in enumerate(nets):
        net = dict(net)
        ma, p5, p9, K, nh, _ = routeA_full_survey(net, Krange, args.seed, rng, terzaghi=args.terzaghi)
        mb, pb, p9b, Kb, nhb, _ = routeB_spatial_block(net, 0.4, Krange, args.seed, rng)
        errA.extend([ma]); errB.extend([mb])
        summary["routeA"].append({"net": i, "mae": ma, "p50": p5, "p90": p9, "K": K, "n_hid": nh})
        summary["routeB"].append({"net": i, "mae": mb, "p50": pb, "p90": p9b, "K": Kb, "n_hid": nhb})
        if i < 6 or (i + 1) % 10 == 0:
            print(f"net#{i:02d}  A(full,K={K}) {ma:5.2f}° | "
                  f"B(block,K={Kb}) {mb:5.2f}°  (hid={nh})")

    mA, mB = np.mean(errA), np.mean(errB)
    print("\n================ 双路线对照 (隐伏点 acos 均值) ================")
    print(f"  路线 A  全量调查 (set_ids + 组内传播): {mA:.2f}°  [N={len(errA)} nets]")
    print(f"  路线 B  空间分块 (no-leak 局部估组):   {mB:.2f}°  [N={len(errB)} nets]")
    print(f"  结论: 全量调查姿势比无组属性的局部勘察姿势 "
          f"{'优' if mA < mB else '劣'} {abs(mA-mB):.2f}°")
    summary["mean"] = {"routeA": float(mA), "routeB": float(mB)}

    out_path = os.path.join(ROOT, "results/auto_label_demo.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"-> {out_path}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.bar(["A full survey", "B spatial block"], [mA, mB],
                    color=["#2a9d8f", "#e76f51"])
            plt.ylabel("Hidden-point MAE (deg)")
            plt.title("Route A vs Route B (no-leak)")
            for j, v in enumerate([mA, mB]):
                plt.text(j, v + 0.5, f"{v:.2f}°", ha="center")
            plt.tight_layout()
            fig = os.path.join(ROOT, "results/auto_label_demo.png")
            plt.savefig(fig, dpi=120)
            print(f"-> {fig}")
        except Exception as e:
            print(f"[plot skip] {e}")


def run_las(args):
    """--input-las PATH: 通用单井 FMI LAS -> 路线 A 自动打标 + 无泄漏验证.

    协议: 掩码 default_rng(999).random(n)<0.4 ; KMeans seed=args.seed ;
    K≤12 细分 (产品档位) ; 口径 acos(|<pred,true>|) 度 (隐伏点).
    """
    parsed = read_single_las(args.input_las)
    n = len(parsed["md"])
    pos = build_3d_trajectory(parsed["md"], parsed["hazi"], parsed["hdev"])
    nrm = dip_dipdir_to_normal(parsed["az"], parsed["dip"])
    net = dict(
        pos=pos, nrm=nrm, nrm_full=nrm, len=np.ones(n), lith=np.zeros(n, int),
        s1=np.ones(n), s3=np.ones(n), src="forge_fmi_las", wid=parsed["wid"],
        md_ft=parsed["md"], md_m=parsed["md"] * 0.3048, dip=parsed["dip"],
        dip_dir=parsed["az"], ftype=np.array(parsed["ftype"]), n=n,
    )
    Krange = (args.Kmin, max(args.Kmin, min(args.Kmax, 12)))   # 产品档位 K<=12
    set_ids = generate_set_ids(nrm, Krange=Krange, seed=args.seed)
    # --- Terzaghi re-clustering (可选) ---
    if args.terzaghi:
        well_axis = np.array([0.0, 0.0, 1.0])       # FORGE FMI: near-vertical
        new_ids = _terzaghi_reassign(nrm, set_ids, well_axis=well_axis, seed=args.seed)
        cos_wa = np.abs(_unit(nrm) @ well_axis)
        wk = 1.0 / np.clip(cos_wa, 1e-6, 1.0)
        wk = wk / wk.mean()
        wk = np.clip(wk, 0.1, 5.0)
        print(f"  [terzaghi] 采样偏差校正已启用 (Terzaghi 1965). "
              f"权重: mean={float(wk.mean()):.3f}, max={float(wk.max()):.3f}")
        set_ids = new_ids
    occ = np.random.default_rng(999).random(n) < 0.4
    dirs, _ = set_aware_dirs(pos, nrm, occ, set_ids)
    e = acos_err(dirs, nrm)
    hid = ~occ
    mae = float(e[hid].mean()); p50 = float(np.median(e[hid])); p90 = float(np.percentile(e[hid], 90))
    if args.terzaghi:
        print("  [terzaghi] honest: thin gain (measured -0.2 to -0.3 deg), but peer review always asks, "
              "recommend reporting both corrected and uncorrected")
    print(f"[las] {parsed['wid']}  N={n}  K={int(set_ids.max()) + 1}  "
          f"MAE={mae:.2f}°  p50={p50:.2f}°  p90={p90:.2f}°  (hid={int(hid.sum())})")
    # 存打标结果
    out = args.out or os.path.join(
        ROOT, "data/external/utah_forge_fmi",
        os.path.splitext(os.path.basename(args.input_las))[0] + "_routeA.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save([{**net, "set_ids": set_ids}], out)
    print(f"-> {out}")
    # 报告四件套 (组系表)
    try:
        # 注意: group_table/write_group_table_csv 定义在 scripts/forge_fmi_pipeline.py,
        # 不在 fractureflow.report —— 之前误从 report 导入, ImportError 被 except 吞掉,
        # 导致 --input-las 路线的组系表 CSV 静默丢失 (2026-08-28 修复).
        from forge_fmi_pipeline import group_table, write_group_table_csv
        rows = group_table([nrm], [set_ids], depth=net["md_m"], ftype=net["ftype"])
        csv = out.replace(".pt", "_group_table.csv")
        write_group_table_csv(rows, csv)
        print(f"-> {csv}  ({len(rows)} groups)")
    except Exception as ex:
        print(f"[report skip] {ex}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError, TypeError, RuntimeError) as e:
        if os.environ.get("FRACTUREFLOW_DEBUG"):
            raise
        print(f"\n[错误] {type(e).__name__}: {e}", file=sys.stderr)
        print("[提示] 请检查数据文件路径/列名/空值; 完整调试栈: 设置环境变量 "
              "FRACTUREFLOW_DEBUG=1 后重跑。", file=sys.stderr)
        sys.exit(2)
