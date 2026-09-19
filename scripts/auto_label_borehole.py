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

# 井斜三列同义词 (R213-T2 / R211-R1): 测深 + 井斜角 (偏离铅垂) + 井斜方位 (自北顺时针)。
# 注意: 裂隙的 dip_direction (倾向) **不是**井斜方位, 两组键刻意不重叠。
_SURVEY_SYNONYMS = (
    ("md", ("md", "measured_depth", "测深", "孔深", "井深", "depth_md", "depth")),
    ("dev", ("dev", "incl", "inclination", "well_dev", "井斜", "井斜角", "天顶角")),
    ("azi", ("azi", "azimuth", "well_azi", "dev_azi", "井方位", "井斜方位", "方位角")),
)


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


# 路线 A/B 的必需键: `routeA_full_survey` / `routeB_spatial_block` 一律取 net["nrm_full"],
# 只给 legacy 的 nrm **不够** —— 在这里放宽成 "nrm 也算" 会让校验通过、下一行照样 KeyError,
# 等于把缺陷往后挪 (R268 T1 首跑实测踩过)。给 nrm 兜底 = 默认路径行为改动, 属另单。
NET_REQUIRED_KEYS = ("nrm_full",)
# 逐裂隙对齐键: 行数必须与法向条数一致 (不齐时 routeA 会**静默**把 pos 换成零阵, 路线 B
# 的空间分块随即塌成空观测集 ⇒ 深处 IndexError, 与真缺陷位置无关)
NET_ROW_ALIGNED_KEYS = ("pos", "nrm", "lith", "obs_mask", "set_ids",
                        "md_m", "depth", "ftype", "len")
MIN_NET_ROWS = 1


def check_net_keysets(nets, source):
    """基准档 (--data real/synth) 入口键集/形态校验: 不齐 ⇒ 单行中文提示, 禁裸 KeyError.

    为什么必须有 (R261-N1 / R268 T1): 客户递来的 .pt/npz 键集不齐是常态, 而生产默认文件
    `data/real/loaded_real_nets.pt` 本身 391/391 网就没有 `nrm_full` —— 改前在
    `routeA_full_survey` 里抛 `KeyError: 'nrm_full'`, 消息既不说这个网络是第几个、
    也不说它实际有哪些键, 客户无法自查 (R268 实测: 同一批缺陷输入下 routeB 抛的是
    `IndexError: index -1 is out of bounds...`, 与缺陷毫无关系)。
    更糟的一类是**静默产出垃圾数字**: 全零法向与含 NaN 法向改前不报错, 直接给出
    MAE=90.0 / MAE=nan (R268 t1_probe_pre.json) ⇒ 比崩更难发现。

    本函数只做校验与呈现, **不做任何兜底替代** (缺键时改用别的键 / 补算 nrm_full =
    默认路径行为改动, 须另单并带逐位锚)。
    """
    for i, net in enumerate(nets):
        keys = sorted(net.keys())
        wid = net.get("wid", f"#{i}")
        who = f"第 {i + 1}/{len(nets)} 个网络 (wid={wid}, 来源 {source})"
        missing = [k for k in NET_REQUIRED_KEYS if k not in net]
        if missing:
            raise ValueError(
                f"[数据键缺失] {who} 缺必需键 {missing}; 该网络实际键 = {keys}; "
                f"基准档要求每个网带全部裂隙单位法向 (N,3) 于 nrm_full "
                f"(注意: 只有 legacy 的 nrm 不够, 本档不代为换算)。")
        nrm = np.asarray(net["nrm_full"], float)
        if nrm.ndim != 2 or nrm.shape[1] != 3:
            raise ValueError(
                f"[数据形状不齐] {who} 的 nrm_full 形状 = {nrm.shape}, 需要 (N,3) "
                f"= 每行一个单位法向 (nx, ny, nz); 实际键 = {keys}")
        if nrm.shape[0] < MIN_NET_ROWS:
            raise ValueError(
                f"[数据为空] {who} 的 nrm_full 形状 = {nrm.shape}, 0 条裂隙无法出组系表; "
                f"实际键 = {keys}")
        bad_len = [(k, len(np.asarray(net[k]))) for k in NET_ROW_ALIGNED_KEYS
                   if k in net and len(np.asarray(net[k])) != nrm.shape[0]]
        if bad_len:
            raise ValueError(
                f"[数据行不齐] {who}: nrm_full 有 {nrm.shape[0]} 行, 但这些逐裂隙键行数不同 "
                f"{bad_len}; 改前该情况会静默丢掉空间坐标并按零坐标继续算 "
                f"(路线 B 随后在深处抛 IndexError), 已拒绝。")
        if not np.isfinite(nrm).all():
            bad = int((~np.isfinite(nrm).all(1)).sum())
            raise ValueError(
                f"[数据含非有限值] {who} 的 nrm_full 有 {bad} 行含 NaN/Inf (首行索引 "
                f"{int(np.argmax(~np.isfinite(nrm).all(1)))}); 这类输入改前会静默算出 "
                f"MAE=nan 并写进交付报告 —— 已拒绝, 请先清洗该列")
        norms = np.linalg.norm(nrm, axis=1)
        if (norms < 1e-9).any():
            bad = int((norms < 1e-9).sum())
            raise ValueError(
                f"[数据含零向量] {who} 的 nrm_full 有 {bad} 行法向模长为 0 (首行索引 "
                f"{int(np.argmax(norms < 1e-9))}); 零向量无法定向, 改前会静默算出 "
                f"MAE=90.0 的最差值并当成结果输出 —— 已拒绝, 请补测该行产状")
    return nets



def _resolve_survey(col_of, synonyms):
    """井斜列同义词 → 实际列名 (大小写/空白容错); 无匹配 → None。"""
    for s in synonyms:
        c = col_of(s)
        if c is not None:
            return c
    return None


def read_borehole_csv(path, id_col="id", ncol="nx", ucol="ny", vcol="nz",
                      type_col=None, sep=None):
    """读钻孔/露头法向 CSV -> net dict (含 nrm/nrm_full/pos 占位/set_ids 占位)。

    支持两种方言 (T91 演练修复, 自动识别):
      A. 法向列 nx/ny/nz (默认; 可用 ncol/ucol/vcol 自定义列名)
      B. 编录表方言 depth/dip/dip_direction (客户标准格式, 自动转法向)

    容错 (T91): 空值/非数字行跳过并计数警告; 零向量行同样跳过计数;
    全部无效时抛带列名清单的 RuntimeError。

    井斜三列 (R211-R1 / R213-T2, 可选): 测深 + 井斜角(偏离铅垂) + 井斜方位
      (中英同义词见 _SURVEY_SYNONYMS)。三列齐备且测深严格递增时, 用与 LAS 路线
      同一个 `build_3d_trajectory` 积分出井迹, 存进**新键** `net["traj_pos"]`
      (由它反解井轴做 Terzaghi 去偏)。`net["pos"]` 保持历史行为 (零占位) ——
      把井迹升格为 pos 会改到空间传播口径的产物, 须另立基准单, 见 R213 裁决 R-2。

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
    survey_cols = [(key, _resolve_survey(col_of, syns)) for key, syns in _SURVEY_SYNONYMS]
    have_survey = all(c is not None for _, c in survey_cols)
    survey_rows, survey_bad = [], 0
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
        if have_survey:
            try:
                survey_rows.append([float(r[c]) for _, c in survey_cols])
            except (TypeError, ValueError):
                survey_bad += 1
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
    # --- 井迹积分 (R213-T2 / R211-R1): 只喂井轴反解, 不改 net["pos"] ---
    if not have_survey:
        got = [c for _, c in survey_cols if c]
        if got:
            print(f"[csv] 井斜列不齐 (只找到 {got}), 需 测深+井斜角+井斜方位 三列齐备 "
                  f"-> 不做井迹反解")
    elif survey_bad:
        print(f"[csv] 井斜三列齐备但 {survey_bad} 行含非数值 -> 整段井迹反解跳过 "
              f"(禁半途积分出错轴做去偏)")
    elif len(survey_rows) < 2:
        print("[csv] 有效站数 <2, 无法积分井迹")
    else:
        arr = np.asarray(survey_rows, float)
        md_v, dev_v, azi_v = arr[:, 0], arr[:, 1], arr[:, 2]
        if not np.all(np.diff(md_v) > 0):
            print("[csv] 测深列非严格递增 -> 井迹反解跳过 (请按测深排序后重跑)")
        else:
            # build_3d_trajectory(md, hazi, hdev): 井斜自铅垂、方位自北顺时针;
            # 单位只影响位置尺度不影响方向, 而此处只消费方向。
            net["traj_pos"] = build_3d_trajectory(md_v, azi_v, dev_v)
            print(f"[csv] 井迹已积分 ({len(md_v)} 站, 测深 {md_v[0]:.1f}→{md_v[-1]:.1f}, "
                  f"井斜 {dev_v.min():.1f}–{dev_v.max():.1f}°) -> 井轴将自井迹反解")
    return net


def _terzaghi_weights_for(nrm, well_axis):
    """Terzaghi 权重的**唯一**实现出口 (R261 F1: 原先四份手写副本合一)。

    井轴一维 (常向量) 时逐字调用冻结库 `fractureflow.terzaghi.terzaghi_weights`,
    以保持历史交付数字的逐位连续性 (R261 P0 实测两库一维权重非逐位相同 ⇒ 按预注册
    例外条款保留冻结库路径);
    井轴逐点 (N,3) 时走可微层的 legacy 分支 (弯曲井段的物理正确形态, 该分支在本次
    修复前不可达 ⇒ 零漂移风险)。**井轴缺失即硬错误** —— 静默落竖直是 R211-D1 的根形。
    """
    if well_axis is None:
        raise ValueError(
            "Terzaghi 权重需要显式井轴, 本次未给 —— 静默按竖直井 (0,0,1) 计算会让"
            "「已做采样偏差校正」这句话在斜井上不成立 (R211-D1)。"
            "三选一: ① LAS 提供 HAzi/HDev 或 CSV 提供「测深+井斜角+井斜方位」三列 "
            "② --well-axis \"x,y,z\" ③ --terzaghi-assume-vertical 明示接受竖直近似")
    a = np.asarray(well_axis, dtype=np.float64)
    if a.ndim == 2:
        from fractureflow.orthostats.terzaghi_layer import terzaghi_weights_t
        return np.asarray(terzaghi_weights_t(np.asarray(nrm, float), axis=a,
                                             mode="legacy"), dtype=np.float64)
    if a.shape != (3,):
        raise ValueError("--well-axis 需 (3,) 常向量或 (N,3) 逐点轴, got %r" % (a.shape,))
    if terzaghi_weights is None:
        raise ImportError(
            "fractureflow.terzaghi 不可用 —— 拒绝在 CLI 内重抄一份权重公式 "
            "(八期 1/sin 事故的成因就是多份手写副本, R261 F1)")
    return terzaghi_weights(_unit(np.asarray(nrm, float)), well_axis=a)


def _terzaghi_reassign(nrm, set_ids, well_axis=None, seed=42, weights=None):
    """Re-assign set_ids using Terzaghi (1965) weighted spherical k-means.

    Sampling bias correction: probability of intercepting a fracture ∝ |n·a|
    where |n·a| = cos(θ) = sin(α), θ = angle between fracture normal and
    borehole axis, α = angle between fracture plane and borehole axis.
    Correction weight = 1 / |n·a|.  Centroid update becomes:
        centers[k] = _unit((aligned * wk[:, None]).sum(0))
    where wk = terzaghi_weights for this well.

    `well_axis` **必须显式给出** (R261 修复 R211-D1: 旧默认 None 会静默落竖直);
    `weights` 可传入调用侧已算好的权重, 使指派与披露共用同一份数字 (禁重复计算)。
    """
    nrm = np.asarray(nrm, dtype=np.float64)
    pts = _unit(nrm)
    nn = len(pts)
    K = int(set_ids.max()) + 1
    if K <= 1:
        return set_ids
    wk = _terzaghi_weights_for(nrm, well_axis) if weights is None \
        else np.asarray(weights, dtype=np.float64)
    if wk.shape != (nn,):
        raise ValueError("Terzaghi 权重形状 %r 与法向数 %d 不符" % (wk.shape, nn))

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


# ------------------------- Terzaghi 井轴解析与披露 (R261) -------------------------
# 一条铁律: 校正可以按竖直近似, 但**必须自曝**——终端一行 + 交付件一份 terzaghi_meta。
# 档位来源: 预注册映射 R-B 命中,
# 但客户可达形态 72.73% 拿不到井轴 ⇒ 按 availability_tiebreak 降为 R-C 自动降级 + 明示)。
AXIS_RESOLVED = "resolved"
AXIS_OVERRIDE = "assumed_vertical_user_override"     # 有轴可用, 用户仍要求竖直
AXIS_DECLARED = "assumed_vertical_user_declared"    # 无轴且用户明示接受竖直近似
AXIS_NO_DATA = "assumed_vertical_no_survey_data"    # 无轴且输入里根本没有井斜数据


def _resolve_terzaghi_axis(args, traj=None, tag=""):
    """井轴解析: --well-axis > 井迹反解 > 竖直近似 (近似一律标来源, 禁静默)。

    与 `--decoder em` 共用 `em_decoder.resolve_axis` (同一来源优先级、同一词汇
    axis_source / dev_from_z_deg), 两个解码器不再各说各话。
    """
    from fractureflow.em_decoder import resolve_axis      # 延迟导入: 默认路径零依赖
    explicit = _parse_axis_arg(args.well_axis) if args.well_axis else None
    r = resolve_axis(axis=explicit, pos=traj, traj_mode=args.traj_mode)
    found = r["axis"] is not None
    if found and not args.terzaghi_assume_vertical:
        return {"axis": r["axis"], "axis_source": r["source"], "degradation": AXIS_RESOLVED,
                "axis_source_cn": ("显式 --well-axis" if r["source"] == "explicit"
                                   else "井迹反解 (%s)" % r["source"]),
                "dev_from_z_deg": round(float(r["dev_from_z_deg"]), 4),
                "assumed_vertical": False, "explicit_axis_arg": explicit is not None,
                "discarded_axis_dev_deg": None}
    if not found:
        src = AXIS_DECLARED if args.terzaghi_assume_vertical else AXIS_NO_DATA
        why = ("用户明示接受竖直近似 (--terzaghi-assume-vertical)" if args.terzaghi_assume_vertical
               else "输入里没有可用井轴 (未给 --well-axis, 且井迹不可反解: LAS 无 HAzi/HDev / "
                    "CSV 缺「测深+井斜角+井斜方位」三列 / 测深非严格递增 / 有效站数 <2)")
        return {"axis": np.array([0.0, 0.0, 1.0]), "axis_source": src, "degradation": src,
                "axis_source_cn": "竖直近似 —— " + why,
                "dev_from_z_deg": None, "assumed_vertical": True,
                "explicit_axis_arg": explicit is not None, "discarded_axis_dev_deg": None}
    # 井轴本可解析, 但用户用 --terzaghi-assume-vertical 主动放弃 ⇒ 把放弃掉的偏离角说清楚
    return {"axis": np.array([0.0, 0.0, 1.0]), "axis_source": AXIS_OVERRIDE,
            "degradation": AXIS_OVERRIDE,
            "axis_source_cn": ("竖直近似 —— 本可自 %s 反解井轴 (实测相对竖直最大偏离 %.2f°), "
                               "但 --terzaghi-assume-vertical 显式要求按竖直"
                               % (r["source"], float(r["dev_from_z_deg"]))),
            "dev_from_z_deg": None, "assumed_vertical": True,
            "explicit_axis_arg": explicit is not None,
            "discarded_axis_dev_deg": round(float(r["dev_from_z_deg"]), 4)}


def _terzaghi_notice(meta, wk):
    """一行说明「本次校正吃的是哪个轴、值不值得信」—— 随交付件落盘, 不只打在终端。"""
    from fractureflow.orthostats.terzaghi_layer import terzaghi_ess
    ess_frac = float(terzaghi_ess(wk)) / len(wk)
    meta["ess_frac"] = round(ess_frac, 6)
    meta["n_clipped"] = int(((wk <= 0.1 + 1e-12) | (wk >= 5.0 - 1e-12)).sum())
    meta["w_max"] = round(float(wk.max()), 4)
    tail = "ESS/N = %.3f, 权重顶格 %d 条" % (ess_frac, meta["n_clipped"])
    if meta["degradation"] == AXIS_RESOLVED:
        return ("Terzaghi 采样偏差校正已启用: 井轴来源 = %s, 相对竖直最大偏离 %.2f°, %s"
                % (meta["axis_source_cn"], meta["dev_from_z_deg"], tail))
    extra = ""
    if meta["degradation"] == AXIS_OVERRIDE:
        extra = ("  [注意] 已放弃可反解的实测井轴 (偏离 %.2f°) —— 这是用户显式选择"
                 % meta["discarded_axis_dev_deg"])
    elif meta["degradation"] == AXIS_NO_DATA:
        extra = ("  [警告] 本输入无井斜数据, 竖直假设**只在直井上成立**; 若本井实为斜井, "
                 "该口径不成立 (R211-D1), 补井斜三列或 --well-axis 即可改用实测轴")
    else:
        extra = "  [注意] 竖直近似由用户明示接受 (--terzaghi-assume-vertical)"
    return ("明示降级: 未使用实测井轴, 按**竖直井近似**算 w=1/|n·a|; %s; %s" % (tail, extra))


def _apply_terzaghi(nrm, set_ids, args, traj=None, tag=""):
    """该分支的唯一入口: 解析井轴 → 一次算出权重 → 重指派 → 生成披露记录。"""
    meta = _resolve_terzaghi_axis(args, traj=traj, tag=tag)
    wk = _terzaghi_weights_for(nrm, meta["axis"])
    new_ids = _terzaghi_reassign(nrm, set_ids, well_axis=meta["axis"],
                                 seed=args.seed, weights=wk)
    meta["notice"] = _terzaghi_notice(meta, wk)
    meta["K_before"] = int(np.asarray(set_ids).max()) + 1
    meta["K_after"] = int(new_ids.max()) + 1
    meta["rows_reassigned"] = int((np.asarray(new_ids) != np.asarray(set_ids)).sum())
    meta["n_rows"] = int(len(nrm))
    meta["axis_vector"] = ([round(float(x), 9) for x in np.asarray(meta["axis"]).ravel()]
                           if np.asarray(meta["axis"]).ndim == 1 else
                           "per_point_tangent (%d rows)" % int(np.asarray(meta["axis"]).shape[0]))
    print("  [terzaghi] " + meta["notice"])
    return new_ids, meta


def _write_terzaghi_meta(meta, out_pt, wid):
    """落 terzaghi 元数据 (与 _em_meta 同型): 轴来源/偏离角/ESS/顶格数/是否明示假定。

    这是**产品交付件**的诚实性记录: 客户与审计只看 CSV+meta 也能判定本次到底
    用没用上本井的真实井轴 (R261 F4)。
    """
    rec = {"schema": "r261-terzaghi-meta-v1", "task": "R261-R211-D1-fix",
           "defect_closed": "R211-D1 (竖直井轴硬编码 / CSV 死键 / --well-axis 对该分支无效)",
           "well": wid, "decoder": "kmeans_terzaghi_reassign",
           "provenance": ("井轴逐字走 fractureflow.em_decoder.resolve_axis "
                          "(--well-axis > 井迹反解 > 明示假定竖直); 权重一维轴逐字走 "
                          "fractureflow.terzaghi.terzaghi_weights, 逐点轴走 "
                          "orthostats.terzaghi_layer(legacy)"),
           **{k: v for k, v in meta.items() if k != "axis"}}
    p = out_pt.replace(".pt", "_terzaghi_meta.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2, allow_nan=False)
    print("  [terzaghi] meta -> %s" % p)


# ------------------------- EM 解码器 (R211 接入) -------------------------
# 资产来源: R208 T0-P3 离线天花板诊断 (`scripts/r208_diag_em.py`, 已冻结)。
# 一个 ~100 行的经典估计器 (kmeans 初始化 + 责任度 EM + Terzaghi 加权) 在 4/6 个
# r207 TRUTH 靶站把组系表模态误差比同 K 同协议 kmeans_obs 改善 1.29–2.28°。
# 产线移植件: `src/fractureflow/em_decoder.py` (数学 verbatim + κ 防溢出 + 似然单调自检)。
# 适用域: em_terzaghi_K 需要井轴 a; 井轴缺失 = 硬错误 (R213-T2), 要 em_plain 须显式
# --em-allow-plain / --em-weights none; ESS/N 不过闸仍降级 em_plain 但事实进交付件。
def _parse_axis_arg(s):
    """--well-axis "x,y,z" -> np.array([x,y,z])。"""
    try:
        v = np.array([float(x) for x in str(s).replace(";", ",").split(",")], float)
    except ValueError:
        raise ValueError(f"--well-axis 需 3 个逗号分隔数字, got {s!r}")
    if v.shape != (3,):
        raise ValueError(f"--well-axis 需 3 个分量, got {v.shape}")
    return v


def _em_notice(dec):
    """一行说明「本次组心到底做没做采样偏差校正」—— 随交付件落盘, 不只打在终端。"""
    reason = dec.get("downgrade_reason")
    if reason is None:
        return ("Terzaghi 采样偏差校正已生效 (w=1/|n·a|, 自归一化 clip[0.1,5], "
                "ESS/N=%.3f)" % float(dec["ess_frac"]))
    if reason == "no_well_axis":
        return ("无井轴 -> 已降级 em_plain_K: 未做 Terzaghi 采样偏差校正, 总体组心估计"
                "方差更大 (本档的显式入口 = --em-allow-plain 或 --em-weights none)")
    return ("Terzaghi 权重有效样本量不过闸 (ESS/N=%.3f < %.2f) -> 已降级 em_plain_K: "
            "权重长尾会让方差放大吃掉去偏, 总体组心估计方差更大"
            % (float(dec["terzaghi_ess_frac_raw"]),
               float(dec["ess_gate"]["min_ess_frac"])))


def _em_decode(nrm, K, args, pos=None, tag=""):
    """调 `fractureflow.em_decoder` 解组系表三件套 (组心 / κ̂ / n)。

    井轴优先级: --well-axis > 井迹反解 (--traj-mode) > 无。
    R213-T2 (并入 R211-R1): **无井迹 = 硬错误** —— 想要不加权的 em_plain 必须显式
    `--em-allow-plain`; 旧行为是打一行提示后换档, 客户拿到的其实是另一套口径。
    ESS/N 不过闸仍按 R211 裁定降级, 但降级事实进交付件 (`dec["notice"]`)。
    失败即响亮拒绝 (ValueError / EMDecoderError -> 顶层 exit 2), 禁静默回退到 kmeans。
    """
    from fractureflow import em_decoder as _em      # 延迟导入: 默认 kmeans 路径零依赖
    axis = _parse_axis_arg(args.well_axis) if args.well_axis else None
    if args.em_weights == "terzaghi" and not args.em_allow_plain:
        if _em.resolve_axis(axis=axis, pos=pos, traj_mode=args.traj_mode)["axis"] is None:
            raise ValueError(
                "--decoder em 需要井轴, 本次两种来源都拿不到: 未给 --well-axis, 且井迹"
                "不可反解 (CSV 缺「测深+井斜角+井斜方位」三列 / LAS 无 HAzi,HDev / "
                "测深非严格递增 / 有效站数 <2)。没有井轴就没有 Terzaghi 去偏, 不再静默"
                "换档。三选一: ① 补齐井斜三列 ② --well-axis \"x,y,z\" "
                "③ 明示接受不加权档 --em-allow-plain (出 em_plain_K, 组心方差更大)")
    dec = _em.decode_group_table(nrm, K, axis=axis, pos=pos,
                                 weights=args.em_weights, traj_mode=args.traj_mode,
                                 seed=(args.em_seed if args.em_seed is not None
                                       else args.seed),
                                 ess_min=args.em_ess_min)
    dec["notice"] = _em_notice(dec)
    dev = ("n/a" if dec["axis_dev_from_z_deg"] is None
           else "%.1f°" % dec["axis_dev_from_z_deg"])
    print(f"  [em] 解码器 = {dec['decoder']}  K = {dec['K_fit']}  N = {dec['n_obs']}  "
          f"井轴来源 = {dec['axis_source']}  (相对竖直最大偏离 {dev})")
    if dec["ess_gate"] is not None:
        print(f"  [em] Terzaghi 权重 1/|n·a| 自归一化 clip[0.1,5] (mode={dec['w_mode']}): "
              f"ESS/N = {dec['terzaghi_ess_frac_raw']:.3f}  "
              f"闸门 ≥{dec['ess_gate']['min_ess_frac']:.2f} = "
              f"{'PASS' if dec['ess_gate']['pass'] else 'FAIL'}")
    print(f"  [em] " + ("降级明示: 请求 weights=%s -> " % dec["weights_requested"]
                        if dec["downgrade_reason"] else "") + dec["notice"])
    print(f"  [em] 似然单调 = {dec['ll_monotone']} (轮数 {dec['n_iter']}, "
          f"空分量 {dec['n_empty_components']}); κ̂ = "
          + ", ".join("%.1f" % k for k in dec["kappa"]))
    if tag:
        print(f"  [em] {tag}")
    return dec


def _em_enrich_rows(rows, dec):
    """组系表补三件套列 (原地补列, 既有列一字不动): κ̂ / n_eff / Terzaghi 修正占比。

    「占比」在现役表里是**观测计数占比**, 被 |n·a| 偏置 (近垂直面被系统性低估)。
    Terzaghi 修正占比 = Σw_k / Σw 是无偏估计, 与原始占比并列给出 —— 不覆盖原列。
    """
    w = dec.get("weights")
    tot = float(np.sum(w)) if w is not None else None
    wsum = (np.array([float(np.sum(w[dec["assign"] == k]))
                      for k in range(dec["K_fit"])]) if w is not None else None)
    for r in rows:
        # 行不变量先落 (含降级事实) —— 客户只打开 CSV 也必须看见本次到底去没去偏
        r["decoder"] = dec["decoder"]
        r["axis_source"] = dec["axis_source"]
        r["em_weights_used"] = dec["weights_used"]
        r["terzaghi_corrected"] = ("是" if dec["downgrade_reason"] is None else "否")
        r["em_ess_frac"] = round(float(dec["ess_frac"]), 3)
        r["em_downgrade_reason"] = dec["downgrade_reason"] or ""
        r["em_notice"] = dec.get("notice", "")
        k = int(r["group_id"]) - 1
        if k >= dec["K_fit"]:
            continue
        r["kappa_hat"] = round(float(dec["kappa"][k]), 2)
        r["n_eff_terzaghi"] = (None if dec["n_eff"] is None
                               else round(float(dec["n_eff"][k]), 1))
        if wsum is not None and tot:
            pct = float(wsum[k]) / tot * 100.0
            r["proportion_weighted"] = round(pct, 1)
            r["development_grade_weighted"] = (
                "很发育" if pct >= 20.0 else "发育" if pct >= 10.0
                else "较发育" if pct >= 5.0 else "不发育")
    return rows


def _write_em_meta(dec, out_pt, wid):
    """落 EM 元数据 (自检证据 + 可复算参数), 便于审计与回归。"""
    meta = {
        "task": "R211-em-decoder", "watermark": "研究线数字, 禁入对外",
        "well": wid, "decoder": dec["decoder"],
        "K_fit": dec["K_fit"], "n_obs": dec["n_obs"],
        "n_points": [int(x) for x in dec["n_points"]],
        "kappa_hat": [round(float(x), 4) for x in dec["kappa"]],
        "pi": [round(float(x), 6) for x in dec["pi"]],
        "weights_requested": dec["weights_requested"],
        "weights_used": dec["weights_used"], "w_mode": dec["w_mode"],
        "downgrade_reason": dec["downgrade_reason"],
        "notice": dec.get("notice", ""),
        "axis_source": dec["axis_source"],
        "axis_dev_from_z_deg": (None if dec["axis_dev_from_z_deg"] is None
                                else round(float(dec["axis_dev_from_z_deg"]), 4)),
        "terzaghi_ess": round(float(dec["ess"]), 3),
        "terzaghi_ess_frac": round(float(dec["ess_frac"]), 6),
        "terzaghi_ess_frac_raw": (None if dec.get("terzaghi_ess_frac_raw") is None
                                  else round(float(dec["terzaghi_ess_frac_raw"]), 6)),
        "ess_gate": dec.get("ess_gate"),
        "ll_monotone": bool(dec["ll_monotone"]),
        "ll_trace_tail": [round(float(x), 8) for x in dec["ll_trace"][-3:]],
        "n_iter": dec["n_iter"], "n_empty_components": dec["n_empty_components"],
        "provenance": "src/fractureflow/em_decoder.py (R208 T0-P3 修正版产线移植); "
                      "证据: 内部基线诊断件 (未随本发布分发)",
    }
    p = out_pt.replace(".pt", "_em_meta.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"  [em] meta -> {p}")


def _em_group_table_csv(nrm, set_ids, dec, out_pt, depth=None, ftype=None):
    """EM 路径组系表落盘 (复用 forge_fmi_pipeline.group_table + 三件套补列)。"""
    from forge_fmi_pipeline import group_table, write_group_table_csv
    rows = group_table([np.asarray(nrm, float)], [np.asarray(set_ids, int)],
                       depth=depth, ftype=ftype)
    _em_enrich_rows(rows, dec)
    csv = out_pt.replace(".pt", "_group_table.csv")
    write_group_table_csv(rows, csv)
    print(f"  [em] 组系表 (含 κ̂ / n_eff / 修正占比) -> {csv}  ({len(rows)} groups)")
    return rows


# ----------------------------- 路线 A -----------------------------
def routeA_full_survey(net, Krange=(2, 7), seed=42, rng=None, terzaghi=False,
                       axis_args=None):
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
    # --- Terzaghi re-clustering (可选; 井轴必须解析, R261 修复 R211-D1) ---
    if terzaghi:
        if axis_args is None:
            raise ValueError(
                "routeA_full_survey(terzaghi=True) 需要 axis_args (井轴来源参数) —— "
                "命令行请用 --terzaghi 配 --well-axis 或 --terzaghi-assume-vertical; "
                "研究脚本内传 argparse.Namespace(well_axis=..., traj_mode=..., "
                "terzaghi_assume_vertical=..., seed=...)")
        set_ids, meta = _apply_terzaghi(nrm_full, set_ids, axis_args,
                                        traj=net.get("traj_pos"), tag="routeA")
        net["terzaghi_meta"] = meta
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


def _console_safe_streams():
    """客户机器上的默认控制台码 (中文 Windows cmd = cp936/GBK) 编不出组合字符
    (κ̂ 里的 U+0302 就是实例) —— 一个 print 失败会让整条链在交付件落盘之前崩掉。

    只放宽**错误处理** (编码本身不动): UTF-8 环境下没有字符会触发替换 ⇒ 对既有输出逐位
    无影响; GBK 环境下不可编码字符退化为 '?' 而不是整单失败 (R261-A-3 / 闸门 G7)。
    """
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass            # 被测试框架/管道包装替换掉的流: 无 reconfigure 可用


def main():
    _console_safe_streams()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synth", "real"], default="synth")
    ap.add_argument("--real-path", default=None,
                    help="--data real 的 .pt 路径 (默认 data/real/loaded_real_nets.pt; "
                         "给自产/客户文件时用于就地核对键集)")
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
                         "(=与井轴平行的裂隙权重高, 近正交的权重低)。井轴自动取用: "
                         "--well-axis 优先, 否则由井迹反解 (LAS 的 HAzi/HDev, 或 CSV 的"
                         "「测深+井斜角+井斜方位」三列); 反解不出时按竖直近似并在交付件里"
                         "明示 (R211-D1 修复, R261)。用的哪个轴、偏离竖直多少度、ESS 多少 "
                         "同时打在终端并写进 <out>_terzaghi_meta.json。")
    ap.add_argument("--terzaghi-assume-vertical", action="store_true",
                    help="--terzaghi 专用: 强制按竖直井 (0,0,1) 近似, 即使本可反解出实测井轴 "
                         "(用于复现历史竖直假设口径)。斜井上该口径不成立, 故被放弃的实测偏离角"
                         "会一并写进交付件")
    ap.add_argument("--decoder", choices=["kmeans", "em"], default="kmeans",
                    help="组系表解码器: kmeans = 现役行为 (默认, 零改动) | "
                         "em = polar-Bingham 混合 EM (+可选 Terzaghi 去偏)。"
                         "R208 T0-P3 证据: em_terzaghi_K 在 4/6 个 r207 TRUTH 靶站优于同 K "
                         "kmeans_obs 1.29–2.28°; 真实站点只报一致性。")
    ap.add_argument("--em-weights", choices=["terzaghi", "none"], default="terzaghi",
                    help="--decoder em 专用: terzaghi = 1/|n·a| 采样偏差校正 (需井轴) | "
                         "none = em_plain (不加权)")
    ap.add_argument("--em-K", type=int, default=0,
                    help="--decoder em 专用: 组数; 0 = 沿用 kmeans 路径选定的 K (同 K 才可比)")
    ap.add_argument("--em-seed", type=int, default=None,
                    help="--decoder em 专用: kmeans 初始化种子 (默认 = --seed)")
    ap.add_argument("--em-ess-min", type=float, default=0.5,
                    help="--decoder em 专用: Terzaghi 权重有效样本量闸门 ESS/N ≥ 阈值, "
                         "不过闸则显式降级 em_plain (0 = 关闭闸门)")
    ap.add_argument("--em-allow-plain", action="store_true",
                    help="--decoder em 专用: 明示接受无井轴时的不加权档 em_plain_K。"
                         "不加此旗且井轴两种来源都缺 => 硬错误 (R213-T2, 不再静默换档)")
    ap.add_argument("--well-axis", default=None,
                    help='井轴 "x,y,z" (East,North,Up); 缺省自井迹反解 '
                         "(LAS: HAzi/HDev; CSV: 测深+井斜角+井斜方位三列)")
    ap.add_argument("--traj-mode", choices=["tangent", "chord"], default="tangent",
                    help="自井迹反解井轴: tangent = 逐点局部切向 (默认, 弯曲井物理正确) | "
                         "chord = 整段弦方向")
    args = ap.parse_args()

    if args.decoder == "em" and not (args.input_las or args.csv):
        print("[em] 注意: --decoder em 作用于**客户钻孔路径** (--csv / --input-las); "
              "synth/real 基准档 (路线 A/B) 未接入, 本次仍按现役口径跑。")

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
        # --- Terzaghi re-clustering (可选; kmeans 路径专用, 与 EM 解码器互斥) ---
        tz_meta = None
        if args.terzaghi and args.decoder != "em":
            nrm = np.asarray(net["nrm_full"], float)
            net["set_ids"], tz_meta = _apply_terzaghi(
                nrm, net["set_ids"], args, traj=net.get("traj_pos"),
                tag=os.path.basename(str(args.csv)))
        # --- EM 解码器 (R211; 组系表三件套) ---
        if args.decoder == "em":
            if args.terzaghi:
                print("  [em] 注意: --terzaghi (kmeans 路径的重聚类) 在 --decoder em 下不适用, "
                      "已忽略; EM 侧的采样偏差校正由 --em-weights 控制 "
                      f"(当前 = {args.em_weights})")
            nrm_full = np.asarray(net["nrm_full"], float)
            K_em = int(args.em_K) or (int(net["set_ids"].max()) + 1)
            dec = _em_decode(nrm_full, K_em, args, pos=net.get("traj_pos"),
                             tag="拟合输入 = 表内全量法向 (与现役 kmeans 同输入 ⇒ 同 K 可比)")
            net["set_ids"] = dec["assign"]
            if args.out:
                _em_group_table_csv(nrm_full, net["set_ids"], dec, args.out)
                _write_em_meta(dec, args.out, os.path.basename(str(args.csv)))
        if args.out:
            torch.save([net], args.out)
            print(f"[csv] 打标完成 -> {args.out}  (K={int(net['set_ids'].max())+1}, "
                  f"N={len(net['set_ids'])})")
            if tz_meta is not None:
                _write_terzaghi_meta(tz_meta, args.out, os.path.basename(str(args.csv)))
        else:
            print(f"[csv] K={int(net['set_ids'].max())+1}  N={len(net['set_ids'])}")
            if not args.terzaghi or args.decoder == "em":
                for i, s in enumerate(net["set_ids"]):
                    print(f"  #{i:03d} set={int(s)}")
        if args.terzaghi and args.decoder != "em":
            print("  [terzaghi] 诚实边界: 校正收益取决于井轴是否为**本井实测轴**与 ESS/N 高低; "
                  "建议同时报告 corrected 与 uncorrected 两档 —— 本次用的轴、偏离竖直角度、"
                  "ESS 已写入 <out>_terzaghi_meta.json, 不看终端也能核对")
        return

    if args.data == "synth":
        source = "data/synth/synth_val.pt"
        nets = load_synth(args.max_nets)
    else:
        source = args.real_path or "data/real/loaded_real_nets.pt"
        nets = load_real(args.max_nets, path=args.real_path)
    check_net_keysets(nets, source)

    errA, errB = [], []
    summary = {"routeA": [], "routeB": []}
    rng = np.random.default_rng(args.seed)
    for i, net in enumerate(nets):
        net = dict(net)
        ma, p5, p9, K, nh, _ = routeA_full_survey(net, Krange, args.seed, rng,
                                                  terzaghi=args.terzaghi, axis_args=args)
        mb, pb, p9b, Kb, nhb, _ = routeB_spatial_block(net, 0.4, Krange, args.seed, rng)
        errA.extend([ma]); errB.extend([mb])
        summary["routeA"].append({"net": i, "mae": ma, "p50": p5, "p90": p9, "K": K, "n_hid": nh})
        if args.terzaghi and "terzaghi_meta" in net:
            summary["routeA"][-1]["terzaghi"] = {
                k: v for k, v in net["terzaghi_meta"].items()
                if k in ("axis_source", "dev_from_z_deg", "assumed_vertical", "ess_frac",
                         "n_clipped", "rows_reassigned", "notice")}
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
    dec = None
    # --- EM 解码器 (R211; 组系表三件套 / 与现役 kmeans 同 K 可比) ---
    if args.decoder == "em":
        if args.terzaghi:
            print("  [em] 注意: --terzaghi (kmeans 路径的加权重聚类) 与 --decoder em 互斥, "
                  "本次已忽略; EM 侧的采样偏差校正由 --em-weights 控制 "
                  f"(当前 = {args.em_weights}), 井轴自井迹反解 (--traj-mode {args.traj_mode})")
        K_em = int(args.em_K) or (int(set_ids.max()) + 1)
        dec = _em_decode(nrm, K_em, args, pos=pos,
                         tag="拟合输入 = 全量调查法向 (与现役 kmeans 同输入 ⇒ 同 K 可比); "
                             "井轴自 HAzi/HDev 井迹反解")
        set_ids = dec["assign"]
    # --- Terzaghi re-clustering (可选; kmeans 路径专用, 与 EM 解码器互斥) ---
    # R261 修复 R211-D1: 本块曾把井轴写成字面量 (0,0,1) 并注称"本数据近竖直",
    # 而 16B 实测井斜中位 64.58° ⇒ 打印"校正已启用"时用户无从知道吃的是假轴。
    # 现在轴一律自本井井迹反解 (弯井默认逐点切向) 或 --well-axis 显式给;
    # 两种来源都拿不到时按竖直近似, 但降级事实写进 <out>_terzaghi_meta.json (禁静默)。
    tz_meta = None
    if args.terzaghi and args.decoder != "em":
        set_ids, tz_meta = _apply_terzaghi(nrm, set_ids, args, traj=pos,
                                           tag=parsed["wid"])
    occ = np.random.default_rng(999).random(n) < 0.4
    dirs, _ = set_aware_dirs(pos, nrm, occ, set_ids)
    e = acos_err(dirs, nrm)
    hid = ~occ
    mae = float(e[hid].mean()); p50 = float(np.median(e[hid])); p90 = float(np.percentile(e[hid], 90))
    if tz_meta is not None:
        print("  [terzaghi] 诚实边界: 校正收益取决于井轴是否为**本井实测轴**与 ESS/N 高低; "
              "建议同时报告 corrected 与 uncorrected 两档 —— 本次用的轴、偏离竖直角度、ESS "
              "已写入 <out>_terzaghi_meta.json, 不看终端也能核对")
    print(f"[las] {parsed['wid']}  N={n}  K={int(set_ids.max()) + 1}  "
          f"MAE={mae:.2f}°  p50={p50:.2f}°  p90={p90:.2f}°  (hid={int(hid.sum())})")
    # 存打标结果
    out = args.out or os.path.join(
        ROOT, "data/external/utah_forge_fmi",
        os.path.splitext(os.path.basename(args.input_las))[0] + "_routeA.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save([{**net, "set_ids": set_ids}], out)
    print(f"-> {out}")
    if tz_meta is not None:
        _write_terzaghi_meta(tz_meta, out, parsed["wid"])
    # 报告四件套 (组系表)
    try:
        # 注意: group_table/write_group_table_csv 定义在 scripts/forge_fmi_pipeline.py,
        # 不在 fractureflow.report —— 之前误从 report 导入, ImportError 被 except 吞掉,
        # 导致 --input-las 路线的组系表 CSV 静默丢失 (2026-08-28 修复).
        from forge_fmi_pipeline import group_table, write_group_table_csv
        rows = group_table([nrm], [set_ids], depth=net["md_m"], ftype=net["ftype"])
        if dec is not None:
            _em_enrich_rows(rows, dec)
            _write_em_meta(dec, out, parsed["wid"])
        csv = out.replace(".pt", "_group_table.csv")
        write_group_table_csv(rows, csv)
        print(f"-> {csv}  ({len(rows)} groups)")
    except Exception as ex:
        if dec is not None:
            raise       # EM 路径: 报告件失败必须响亮 (红线: 失败模式显式输出, 禁静默)
        print(f"[report skip] {ex}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError, TypeError, RuntimeError,
            IndexError) as e:
        # IndexError 于 R268 T1 加入: 客户数据的行列不齐在深处抛它 (R261-N1 实测), 与
        # KeyError 同族同类 ⇒ 同一呈现纪律; 真·代码 bug 仍可用 FRACTUREFLOW_DEBUG=1
        # 拿回完整栈 (该开关就是为此存在), 故这不是把错误藏起来。
        if os.environ.get("FRACTUREFLOW_DEBUG"):
            raise
        print(f"\n[错误] {type(e).__name__}: {e}", file=sys.stderr)
        print("[提示] 请检查数据文件路径/列名/空值; 完整调试栈: 设置环境变量 "
              "FRACTUREFLOW_DEBUG=1 后重跑。", file=sys.stderr)
        sys.exit(2)
