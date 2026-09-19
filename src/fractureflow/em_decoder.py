# -*- coding: utf-8 -*-
"""组系表解码器 (经典估计版) —— polar Bingham 混合 EM + Terzaghi 采样偏差校正。

【本模块的来历 (R211 T2)】
R208 T0-P3 诊断证明: 一个 ~100 行的经典离线估计器 (kmeans 初始化 + 责任度 EM +
Terzaghi 加权) 在 4/6 个 r207 TRUTH 靶站上把「组系表模态误差」比同 K 同协议
`kmeans_obs` 改善 1.29–2.28°。本模块是那份诊断实现的**产线化移植** (verbatim 数学),
并把三件 R208 修正带过来:
  1. κ-MLE 的正确二分方向 (E_κ[x1²] = d1, E_κ[x1²] 对 κ 单调递增);
  2. 数值防护: e^{κ(t²−1)} (≤1) 取代 e^{κt²}, κ≈2000 不上溢;
  3. EM 加权对数似然逐轮单调自检 (EM 有效性硬证据)。
原实现留在 `scripts/r208_diag_em.py` (R208 冻结件, 一字未动); 本模块与它的逐位
一致性由 `scripts/r211_regression_anchor.py` 断言。

【为什么是"组心"而不是"逐点"】
产品交付物是**组系表**: 每井 K 组 × (组心方向 / κ̂ / 点数)。EM 的估计目标是
**总体组心** (population set center), 而观测目录 (catalog) 被 |n·a| 乘性偏置过:
钻孔与裂隙面相交概率 ∝ |n·a| ⇒ 近井轴平行的面 (n ⊥ a) 被系统性高估、近垂直面被低估。
Terzaghi (1965) 权重 w_i = 1/|n·a_i| (自归一化, 缺失质量未知 ⇒ self-normalized IS)
正是这一步去偏。**没有井轴 a 就没有 Terzaghi** ⇒ 无井轴时显式降级 `em_plain`
(不是静默回退, 见 `decode_group_table` 的 `downgrade_reason`)。

【与现役 kmeans 的关系】
`kmeans_obs` = 球面 k-means + 符号对齐均值, 是 EM 的**初始化**与基线;
EM 在其上做 MLE 精化 (责任度软指派 + 加权散布主特征向量 + κ̂ 闭式 MLE)。
同 K 同权重下 EM 的似然 ≥ kmeans 的似然 (EM 单调性), 但"似然更优"不保证
"组系表更准" —— 部署上与基线并列给数, 不代基线做主。

【确定性与失败显式】
纯 numpy float64, CPU 单线程可复现; 似然非单调 / 非有限 / 分量全空 ⇒ 抛
`EMDecoderError` (响亮拒绝, 禁静默回退)。空分量 (责任度和≈0) 记账并警告。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .setlabel import spherical_kmeans, _sign_align
from .orthostats import log_bingham_norm_np
from .orthostats.terzaghi_layer import (
    terzaghi_weights_t, terzaghi_ess, W_LO_DEFAULT, W_HI_DEFAULT)

__all__ = [
    "EMDecoderError", "ex1", "solve_kappa", "em_mixture",
    "terzaghi_weights", "axis_from_trajectory", "axis_deviation_deg",
    "resolve_axis", "decode_group_table",
    "KAPPA_LO", "KAPPA_HI", "W_LO", "W_HI",
]

KAPPA_LO, KAPPA_HI = 0.5, 2000.0     # κ̂ MLE 求值区间 (R208 T0-P3 verbatim)
W_LO, W_HI = W_LO_DEFAULT, W_HI_DEFAULT   # 0.1 / 5.0 (与现役 terzaghi 同口径)

# 400 点 Gauss-Legendre 求积 (确定性; leggauss 无随机源)
_GL_T, _GL_W = np.polynomial.legendre.leggauss(400)


class EMDecoderError(RuntimeError):
    """EM 解码失败 (响亮拒绝: 不静默回退到 kmeans)。"""


def _unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


# ------------------------------------------------------------------ κ̂ MLE --
def ex1(kappa) -> np.ndarray:
    """E_κ[x1²] = ∫t²e^{κt²}dt / ∫e^{κt²}dt = ∂log c / ∂κ 的极性 Bingham 形式。

    对 κ **单调递增** (κ→0 ⇒ 1/3, κ→∞ ⇒ 1)。
    数值: 写成 e^{κ(t²−1)} (≤1) 防 κ≈2000 上溢 —— 比值里 e^{−κ} 因子相消, 数学等价。
    """
    k = np.atleast_1d(np.asarray(kappa, dtype=np.float64))
    p = np.exp(k[:, None] * (_GL_T ** 2 - 1.0)) * _GL_W
    return (p * (_GL_T ** 2)).sum(1) / p.sum(1)


def solve_kappa(d1) -> np.ndarray:
    """解 E_κ[x1²] = d1 (单调递增 ⇒ 二分: E<d1 抬 lo)。向量化, 60 步。

    历史缺陷 (R208 T0-P3, 已修): 旧实现二分方向写反且短路分支同向, 使 κ 恒取下界
    0.5 ⇒ 混合退化为近均匀分量 ⇒ 责任度趋同 ⇒ 组心坍缩到全局主轴 (modal_err 假高
    到 90°−α)。修前该脚本从未运行过, 零历史结论受影响。
    """
    d = np.atleast_1d(np.asarray(d1, dtype=np.float64))
    lo = np.full_like(d, KAPPA_LO)
    hi = np.full_like(d, KAPPA_HI)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        up = ex1(mid) < d                     # 需要更大的 κ
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return np.clip(0.5 * (lo + hi), KAPPA_LO, KAPPA_HI)


# -------------------------------------------------------------------- EM --
def em_mixture(nrm, K, w_pts=None, iters: int = 60, seed: int = 42,
               tol: float = 1e-11, init_centers=None) -> dict:
    """polar Bingham 混合 EM (kmeans 初始化 + 责任度 EM + 精确 M-step)。

    参数
    ----
    nrm   : (N,3) 无向单位法向 (±n 同面, 内部按 |n·c| 指派)
    K     : 组数 (K > N 时截断到 N)
    w_pts : (N,) 逐点权重, 默认 1。Terzaghi 走 `terzaghi_weights`。
    seed  : kmeans 初始化种子 (见 T1 init-seed 稳健性闸)
    init_centers : 可选 (K,3) 外部初始化组心 (R216 修订 1, 最小附加式注入口;
                   None 时走现役 kmeans(seed) 路径, **默认路径逐位不变**)。
                   非定向输入按无向语义使用 (EM 内部 |n·c| 指派 + M-step 符号
                   对齐), 不新增任何数学。

    返回 dict
    --------
    centers / pi / kappa : (K,3) 组心, (K,) 混合权重 (= 加权责任度份额), (K,) κ̂ MLE
    resp_sum             : 末轮**未加权**责任度和 Σ_i r_ik (选择规则口径)
    ll_trace / n_iter / mono_ok : 加权对数似然逐轮轨迹 / 轮数 / 单调自检
    d1                   : 各组加权散布最大特征值 (κ̂ 的充分统计量)
    n_empty              : 责任度和≈0 的分量个数 (已按 init 保留并记账)
    ess                  : 权重的有效样本量 (Σw)²/Σw²
    """
    nrm = _unit(np.asarray(nrm, dtype=np.float64))
    n = nrm.shape[0]
    if n == 0:
        raise EMDecoderError("em_mixture: 空法向输入 (禁静默产出 NaN 组心)")
    if not np.all(np.isfinite(nrm)):
        raise EMDecoderError("em_mixture: 法向含 NaN/Inf")
    K = int(min(int(K), n))
    if K < 1:
        raise EMDecoderError("em_mixture: K 必须 ≥1, got %r" % K)
    w_pts = np.ones(n) if w_pts is None else np.asarray(w_pts, dtype=np.float64)
    if w_pts.shape[0] != n or not np.all(np.isfinite(w_pts)) or np.any(w_pts <= 0):
        raise EMDecoderError("em_mixture: 权重必须 (N,) 有限正数")
    sw_tot = float(w_pts.sum())

    if init_centers is None:
        cents, _ = spherical_kmeans(nrm, K, seed=seed)   # 现役默认路径 (逐位不变)
    else:
        cents = np.asarray(init_centers, dtype=np.float64)
        if cents.ndim != 2 or cents.shape != (K, 3):
            raise EMDecoderError(
                "em_mixture: init_centers 需 (K,3)=(%d,3), got %r" % (K, (cents.shape,)))
        if not np.all(np.isfinite(cents)):
            raise EMDecoderError("em_mixture: init_centers 含 NaN/Inf")
        cents = cents.copy()
    cents = _unit(cents)
    pi = np.full(K, 1.0 / K)
    kappa = np.full(K, 20.0)
    ll_trace, resp_sum, mono_ok, n_empty = [], None, True, 0
    d1 = np.full(K, np.nan)

    for _ in range(int(iters)):
        cos2 = (nrm @ cents.T) ** 2                                   # (n,K)
        logc = log_bingham_norm_np(np.stack([np.zeros(K), kappa, kappa], axis=1))
        logf = kappa[None, :] * cos2 - kappa[None, :] - logc[None, :]
        logr = np.log(pi)[None, :] + logf
        mx = logr.max(axis=1, keepdims=True)
        r = np.exp(logr - mx)
        rsum = r.sum(axis=1, keepdims=True)
        r /= rsum
        ll = float((w_pts * (mx[:, 0] + np.log(rsum[:, 0]))).sum() / sw_tot)
        ll_trace.append(ll)
        if len(ll_trace) >= 2 and ll_trace[-1] < ll_trace[-2] - 1e-9:
            mono_ok = False
        resp_sum = r.sum(axis=0)
        # ---- M-step ----
        n_empty = 0
        for k in range(K):
            wk = r[:, k] * w_pts
            sw = float(wk.sum())
            if sw < 1e-12:
                d1[k] = ex1(kappa[k])
                n_empty += 1
                continue
            S = (wk[:, None, None] * nrm[:, :, None] * nrm[:, None, :]).sum(0) / sw
            ev, evec = np.linalg.eigh(S)
            m = evec[:, -1]
            if m @ cents[k] < 0:
                m = -m
            cents[k] = _unit(m)
            d1[k] = float(ev[-1])
        kappa = solve_kappa(d1)
        pi = (w_pts[:, None] * r).sum(0) / sw_tot
        pi = np.clip(pi, 1e-9, None)
        pi = pi / pi.sum()
        if len(ll_trace) >= 2 and abs(ll_trace[-1] - ll_trace[-2]) < tol:
            break

    if not (np.all(np.isfinite(cents)) and np.all(np.isfinite(kappa))
            and np.all(np.isfinite(pi))):
        raise EMDecoderError("em_mixture: 输出含 NaN/Inf (估算发散)")
    if n_empty == K:
        raise EMDecoderError("em_mixture: 全部分量责任度和≈0 (数据与 K 不匹配)")
    if not mono_ok:
        raise EMDecoderError(
            "em_mixture: EM 加权对数似然非单调 (第 %d 轮) —— 估计不可信, 拒绝出表"
            % next(i for i in range(1, len(ll_trace)) if ll_trace[i] < ll_trace[i - 1] - 1e-9))
    return {"centers": cents, "pi": pi, "kappa": kappa, "resp_sum": resp_sum,
            "ll_trace": ll_trace, "n_iter": len(ll_trace), "mono_ok": mono_ok,
            "d1": d1, "n_empty": int(n_empty),
            "ess": float(terzaghi_ess(w_pts)) if w_pts is not None else float(n)}


# --------------------------------------------------------------- Terzaghi --
def terzaghi_weights(nrm, axis=None, mode: str = "legacy") -> np.ndarray:
    """Terzaghi (1965) 采样偏差权重 w = 1/|n·a| (自归一化 mean=1, 裁剪 [0.1,5])。

    **逐字调用 `orthostats.terzaghi_weights_t`** (不重实现)。`mode="legacy"` =
    硬裁剪, 逐位复现现役 numpy `terzaghi.terzaghi_weights` 语义 (= R208 T0-P3 用的
    口径, 该口径下的增益才有实测证据); `mode="soft"` 为可微版 (差异只在裁剪邻域)。
    """
    if axis is None:
        raise EMDecoderError("terzaghi_weights: 缺少井轴 a —— 无井轴不做 Terzaghi 校正")
    a = np.asarray(axis, dtype=np.float64)
    if a.ndim not in (1, 2):
        raise EMDecoderError("terzaghi_weights: axis 需 (3,) 或 (N,3), got %r" % (a.shape,))
    return np.asarray(terzaghi_weights_t(np.asarray(nrm, dtype=np.float64), axis=a,
                                         mode=mode), dtype=np.float64)


# ------------------------------------------------------------------ 井轴 --
def axis_from_trajectory(pos, mode: str = "tangent", min_chord: float = 1e-3):
    """从井迹 (裂隙交点位置, 按 MD 升序) 反解井轴。

    mode="tangent": 逐点局部单位切向 (中心差分) —— 物理上正确 (某深度的裂隙是被
                    **该处**的井轴钻到的); 直井退化为常向量。
    mode="chord"  : 首尾弦方向 (整段平均轴), 与 R208 协议的"固定轴"同构。

    返回 (axis (N,3) 或 (3,), source_str); 井迹退化 (跨度 < min_chord) 时返回 (None, None)。
    """
    if pos is None:
        return None, None
    p = np.asarray(pos, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2:
        return None, None
    span = float(np.linalg.norm(p.max(0) - p.min(0)))
    if span < min_chord:
        return None, None
    if mode == "chord":
        v = p[-1] - p[0]
        nv = float(np.linalg.norm(v))
        if nv < min_chord:
            return None, None
        return _unit(v), "trajectory_chord"
    if mode != "tangent":
        raise EMDecoderError("axis_from_trajectory: mode 必须 tangent/chord")
    # 先判井迹是否本质一维 (直井): 此时逐点切向无意义, 且行序未必是 MD 序 (梯度会乱)
    # ⇒ 直接取主方向 (无向, 只用于 |n·a|)。弯曲井迹 (FORGE 造斜段) 才走逐点切向。
    q = p - p.mean(0)
    _, sv, vt = np.linalg.svd(q, full_matrices=False)
    if sv[0] <= min_chord or (sv[1] / max(sv[0], 1e-30)) < 1e-3:
        return _unit(vt[0]), "trajectory_straight"
    d = np.gradient(p, axis=0)          # 中心差分 (端点单边)
    nv = np.linalg.norm(d, axis=1)
    if np.any(nv < min_chord):
        good = nv >= min_chord
        if not good.any():
            return None, None
        d[~good] = d[good][0]           # 退化步长处用邻点方向补 (禁 NaN 传播)
        nv = np.linalg.norm(d, axis=1)
    return d / nv[:, None], "trajectory_tangent"


def axis_deviation_deg(axis, ref=(0.0, 0.0, 1.0)) -> float:
    """井轴相对参考方向 (默认竖直) 的最大偏离角 (度)。无向 (±a 同轴)。"""
    a = np.asarray(axis, dtype=np.float64)
    if a.ndim == 1:
        a = a[None]
    a = _unit(a)
    r = _unit(np.asarray(ref, dtype=np.float64))
    return float(np.rad2deg(np.arccos(np.clip(np.abs(a @ r), -1.0, 1.0))).max())


def resolve_axis(axis=None, pos=None, traj_mode: str = "tangent") -> dict:
    """井轴解析 (优先级: 显式向量 → 井迹 → 无)。

    返回 dict(axis, source, dev_from_z_deg, n_axis) —— axis=None 表示**无井轴**
    (调用侧必须显式降级 em_plain 并把 source 写进产物, 禁沉默)。
    """
    if axis is not None:
        a = np.asarray(axis, dtype=np.float64)
        if a.ndim == 2 and a.shape[1] == 3:
            a = a.copy()
        elif a.ndim == 1 and a.size == 3:
            a = a.copy()                     # 保持一维: 走模块的常向量路径 (逐位)
        else:
            raise EMDecoderError("resolve_axis: 显式井轴需 (3,) 或 (N,3), got %r"
                                 % (a.shape,))
        a = _unit(a)
        if not np.all(np.isfinite(a)):
            raise EMDecoderError("resolve_axis: 显式井轴含 NaN/Inf")
        return {"axis": a, "source": "explicit",
                "dev_from_z_deg": axis_deviation_deg(a),
                "n_axis": int(np.asarray(a).reshape(-1, 3).shape[0])}
    a, src = axis_from_trajectory(pos, mode=traj_mode)
    if a is None:
        return {"axis": None, "source": "unavailable",
                "dev_from_z_deg": None, "n_axis": 0}
    return {"axis": a, "source": src,
            "dev_from_z_deg": axis_deviation_deg(a),
            "n_axis": int(np.asarray(a).shape[0])}


# ------------------------------------------------------------- 解码入口 --
def decode_group_table(nrm, K, axis=None, pos=None, weights: str = "terzaghi",
                       traj_mode: str = "tangent", iters: int = 60,
                       seed: int = 42, w_mode: str = "legacy",
                       ess_min: float = 0.5, init_centers=None) -> dict:
    """产线解码入口: 观测法向 → 组系表三件套 (组心 / κ̂ / n)。

    参数
    ----
    nrm     : (N,3) 用于估组的法向 (产线 = 全量调查; 评测 = 观测子集)
    K       : 组数 (产品档位 / silhouette 选定的 K)
    axis / pos : 井轴的两种给法 (显式向量 / 井迹)。二者都无 ⇒ 降级 em_plain。
    weights : "terzaghi" (默认, 有井轴时) / "none" (em_plain)
    ess_min : Terzaghi 权重的有效样本量闸门 (默认 0.5, 0 = 关闭)。
              权重长尾 ⇒ 方差放大吃掉去偏, `terzaghi_layer` 自陈「ESS/N < 0.5 基本
              不值得信」。不过闸即**显式降级 em_plain** 并把实测 ESS 留在产物里
              (`terzaghi_ess_frac_raw` + `ess_gate`), 不是静默忽略。
    init_centers : 可选 (K,3) 外部初始化 (R216 修订 1, 纯透传 `em_mixture`;
                   None = 现役 kmeans 默认路径, 逐位不变)。

    返回 dict: centers/pi/kappa/n_points/assign/weights/ess/ess_gate/axis_source/
    weights_requested/weights_used/downgrade_reason/ll_monotone/n_iter/n_empty/decoder
    """
    if weights not in ("terzaghi", "none"):
        raise EMDecoderError("decode_group_table: weights 必须 terzaghi/none")
    nrm = np.asarray(nrm, dtype=np.float64)
    ax = resolve_axis(axis=axis, pos=pos, traj_mode=traj_mode)
    requested = weights
    downgrade_reason = None
    use_w, ess_raw, gate = None, None, None
    if weights == "terzaghi":
        if ax["axis"] is None:
            # 适用域 = 有井轴钻孔 (DECOVALEX 等无井轴输入): 自动降级 + 明示
            weights = "none"
            downgrade_reason = "no_well_axis"
        else:
            w_raw = terzaghi_weights(nrm, ax["axis"], mode=w_mode)
            ess_raw = float(terzaghi_ess(w_raw)) / len(nrm)
            gate = {"min_ess_frac": float(ess_min), "terzaghi_ess_frac": ess_raw,
                    "pass": bool(ess_min <= 0 or ess_raw >= ess_min)}
            if gate["pass"]:
                use_w = w_raw
            else:
                weights = "none"
                downgrade_reason = ("terzaghi_ess_below_gate (ESS/N=%.3f < %.2f)"
                                    % (ess_raw, ess_min))
    decoder = "em_terzaghi_K" if weights == "terzaghi" else "em_plain_K"
    res = em_mixture(nrm, K, w_pts=use_w, iters=iters, seed=seed,
                     init_centers=init_centers)
    centers = res["centers"]
    pts = _unit(np.asarray(nrm, dtype=np.float64))
    assign_all = np.abs(pts @ centers.T).argmax(1)
    n_points = np.bincount(assign_all, minlength=len(centers))
    # Terzaghi 再加权: 该组有效样本量 (把采样偏差再贴回组规模的可信度)
    n_eff = None
    if use_w is not None:
        n_eff = [float((use_w[assign_all == k].sum() ** 2)
                       / max(float((use_w[assign_all == k] ** 2).sum()), 1e-300))
                 for k in range(len(centers))]
    return {
        "decoder": decoder,
        "centers": centers,
        "kappa": res["kappa"],
        "pi": res["pi"],
        "n_points": n_points,
        "n_eff": n_eff,
        "assign": assign_all,
        "weights": use_w,
        "weights_requested": requested,
        "weights_used": weights,
        # w_mode 反映"被评估过的 Terzaghi 权重口径" (被闸门拒掉时也要留证)
        "w_mode": w_mode if (use_w is not None or ess_raw is not None) else None,
        "downgrade_reason": downgrade_reason,
        "terzaghi_ess_frac_raw": ess_raw,
        "ess_gate": gate,
        "axis": ax["axis"],
        "axis_source": ax["source"],
        "axis_dev_from_z_deg": ax["dev_from_z_deg"],
        "n_axis_rows": ax["n_axis"],
        "ess": res["ess"] if use_w is not None else float(len(pts)),
        "ess_frac": (res["ess"] / len(pts)) if use_w is not None else 1.0,
        "ll_trace": res["ll_trace"],
        "n_iter": res["n_iter"],
        "ll_monotone": res["mono_ok"],
        "n_empty_components": res["n_empty"],
        "K_fit": int(len(centers)),
        "n_obs": int(len(pts)),
    }


def group_centers_from_labels(nrm, assign, axis=None, mode: str = "legacy") -> dict:
    """给定已有组标签 → 组心 + κ̂ (加权散布 MLE) + 三件套 (对照用, 不跑 EM)。

    用途: 把产线现役 kmeans 的组标签放进**同一张组系表**做逐组对照
    (同 K 同数据, 只差解码器)。返回结构与 `decode_group_table` 的 centers/kappa 对齐。
    """
    pts = _unit(np.asarray(nrm, dtype=np.float64))
    assign = np.asarray(assign, dtype=int)
    K = int(assign.max()) + 1 if len(assign) else 0
    use_w = terzaghi_weights(pts, axis, mode=mode) if axis is not None else None
    centers = np.zeros((K, 3))
    kappa = np.full(K, np.nan)
    n_eff = []
    for k in range(K):
        sel = pts[assign == k]
        if len(sel) == 0:
            continue
        centers[k] = _unit(_sign_align(sel).mean(0))
        wk = np.ones(len(sel)) if use_w is None else use_w[assign == k]
        S = (wk[:, None, None] * sel[:, :, None] * sel[:, None, :]).sum(0) / wk.sum()
        ev = np.linalg.eigvalsh(0.5 * (S + S.T))
        kappa[k] = float(solve_kappa(np.array([float(ev[-1])]))[0])
        n_eff.append(float((wk.sum() ** 2) / max(float((wk ** 2).sum()), 1e-300)))
    return {"centers": centers, "kappa": kappa, "n_points": np.bincount(assign, minlength=K),
            "n_eff": n_eff, "ess": None if use_w is None else float(terzaghi_ess(use_w))}
