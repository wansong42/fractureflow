# -*- coding: utf-8 -*-
"""组系表质量指标 —— 产品交付物的评测模块。

产品的实际交付物是**组系表**: 每井 K 组, 每组 (模态方向, 点数, 可选置信)。
真值也有组系表。本模块定义组系表之间的对比指标。

参照: docs/技术路线_诚实评测与组系表产品化_v1.md §B1。

用法:
    from fractureflow.set_table_eval import set_table_score, match_tables, evaluate_set_table
    score = evaluate_set_table(pred_centers, truth_centers, pred_assign, truth_assign)
    # score = {modal_err_deg, K_diff, coverage, ari, ...}
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 1. 核心指标
# ---------------------------------------------------------------------------

def match_tables(pred_centers: np.ndarray,
                 truth_centers: np.ndarray) -> list:
    """匈牙利匹配, 代价 = acos|<c_i, c_j>|。

    返回 [(pred_i, truth_j), ...] 匹配对列表。
    """
    Kp = pred_centers.shape[0]
    Kt = truth_centers.shape[0]
    if Kp == 0 or Kt == 0:
        return []
    cost = np.zeros((Kp, Kt))
    for i in range(Kp):
        for j in range(Kt):
            cos = np.clip(np.abs(pred_centers[i] @ truth_centers[j]), 0, 1)
            cost[i, j] = np.degrees(np.arccos(cos))
    row_ind, col_ind = linear_sum_assignment(cost)
    return list(zip(row_ind.tolist(), col_ind.tolist()))


def set_table_score(
    pred_centers: np.ndarray,
    truth_centers: np.ndarray,
    pred_assign: Optional[np.ndarray] = None,
    truth_assign: Optional[np.ndarray] = None,
    match_threshold: float = 30.0,
    pred_nrm: np.ndarray = None,
    truth_nrm: np.ndarray = None,
) -> dict:
    """组系表评分 (主指标)。

    参数:
        pred_centers: (K_pred, 3) 预测组模态方向 (单位向量)
        truth_centers: (K_truth, 3) 真值组模态方向 (单位向量)
        pred_assign: (L,) 预测逐点组指派 (可选, 用于算 ARI)
        truth_assign: (L,) 真值逐点组指派 (可选, 用于算 ARI)
        match_threshold: 匹配成功阈值 (度), 默认 30°
        pred_nrm: (L, 3) 预测侧点云法向 (可选, 用于算组内离散/最大偏差)
        truth_nrm: (L, 3) 真值侧点云法向 (可选)

    返回:
        modal_err_deg: 匹配上的组, 模态方向误差的均值 (主指标)
        K_pred, K_truth: 组数
        K_diff: |K_pred - K_truth|
        coverage: 被成功匹配 (|误差|<threshold) 的 truth 组占比
        ari: 点级 adjusted rand index (需要 pred_assign/truth_assign)
        matched_pairs: [(pred_i, truth_j, err_deg), ...]
        dispersion_pred/dispersion_truth: 各预测/真值组系表组内离散与最大偏差
            (新增, 纯加法; 仅当提供对应 nrm+assign 时计算, 否则为 None)
    """
    pred_centers = np.asarray(pred_centers, dtype=np.float64)
    truth_centers = np.asarray(truth_centers, dtype=np.float64)

    # 单位化
    pred_centers = pred_centers / (np.linalg.norm(pred_centers, axis=1, keepdims=True) + 1e-12)
    truth_centers = truth_centers / (np.linalg.norm(truth_centers, axis=1, keepdims=True) + 1e-12)

    K_pred = pred_centers.shape[0]
    K_truth = truth_centers.shape[0]

    if K_pred == 0 or K_truth == 0:
        return {
            "modal_err_deg": 90.0,
            "K_pred": K_pred,
            "K_truth": K_truth,
            "K_diff": abs(K_pred - K_truth),
            "coverage": 0.0,
            "n_matched": 0,
            "matched_pairs": [],
            "ari": None,
        }

    matches = match_tables(pred_centers, truth_centers)
    errs = []
    matched_pairs = []
    matched_truth = set()
    for pi, ti in matches:
        cos = np.clip(np.abs(pred_centers[pi] @ truth_centers[ti]), 0, 1)
        e = float(np.degrees(np.arccos(cos)))
        errs.append(e)
        matched_pairs.append((pi, ti, round(e, 3)))
        matched_truth.add(ti)

    modal_err = float(np.mean(errs)) if errs else 90.0
    n_success = sum(1 for e in errs if e < match_threshold)
    coverage = n_success / max(K_truth, 1)

    result = {
        "modal_err_deg": round(modal_err, 3),
        "K_pred": K_pred,
        "K_truth": K_truth,
        "K_diff": abs(K_pred - K_truth),
        "coverage": round(coverage, 3),
        "n_matched": len(matches),
        "n_success": n_success,
        "matched_pairs": matched_pairs,
        "ari": None,
    }

    # ARI (如果有点级指派)
    if pred_assign is not None and truth_assign is not None:
        result["ari"] = round(_adjusted_rand_index(pred_assign, truth_assign), 4)

    # 组内离散 / 最大偏差 (纯加法, 无点云时为 None, 旧调用零漂移)
    # 口径: dispersion = arccos(mean|cos|) (球面, 全项目统一); max_dev = 到组心最大角距
    # 最小样本规则: N<3 标 statistical=False (不可统计, 详见 docs/评测标准与口径锁定.md §7)
    result["dispersion_pred"] = _group_dispersion(pred_centers, pred_assign, pred_nrm)
    result["dispersion_truth"] = _group_dispersion(truth_centers, truth_assign, truth_nrm)

    return result


def _group_dispersion(centers: np.ndarray, assign: np.ndarray, nrm: np.ndarray) -> list:
    """计算各组组内离散 (arccos(mean|cos|)) 与 max_dev (到组心最大角距).

    返回 list[dict], 每个元素: {k, n, dispersion, max_dev, statistical}
      - dispersion / max_dev 单位: 度
      - statistical: n >= 3 才是可统计的组内离散 (N<3 标 None + False)
    无点云 (nrm/assign 缺失) 时返回 None (调用侧须自行处理).
    """
    if nrm is None or assign is None:
        return None
    centers = np.asarray(centers, dtype=np.float64)
    nrm = np.asarray(nrm, dtype=np.float64)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    out = []
    for k in range(len(centers)):
        sel = assign == k
        n_k = int(sel.sum())
        if n_k == 0:
            out.append({"k": k, "n": 0, "dispersion": None, "max_dev": None, "statistical": False})
            continue
        cos = np.clip(np.abs(nrm[sel] @ centers[k]), 0, 1)
        ang = np.degrees(np.arccos(cos))
        out.append({
            "k": k,
            "n": n_k,
            "dispersion": round(float(ang.mean()), 3),
            "max_dev": round(float(ang.max()), 3),
            "statistical": n_k >= 3,
        })
    return out


# ---------------------------------------------------------------------------
# 2. 从点云估计组系表
# ---------------------------------------------------------------------------

def estimate_set_table(pos: np.ndarray, nrm: np.ndarray, occ: np.ndarray,
                       K: int, seed: int = 42, strict: bool = True) -> dict:
    """从点云估计组系表 (观测-only, 无泄漏)。

    参数:
        pos: (L, 3) 位置
        nrm: (L, 3) 法向 (隐伏点可任意, 不会被使用)
        occ: (L,) bool 观测掩码
        K: 组数
        seed: k-means 种子
        strict: True = 只用观测点 (诚实); False = 用全量 (泄漏)

    返回:
        centers: (K, 3) 组模态方向
        assign: (L,) 组指派 (-1 = 未指派)
        n_points: (K,) 每组点数
    """
    from .setlabel import spherical_kmeans, _sign_align

    pos = np.asarray(pos, dtype=np.float64)
    nrm = np.asarray(nrm, dtype=np.float64)
    occ = np.asarray(occ, dtype=bool)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)

    if strict:
        # 诚实口径: 只用观测点
        nrm_use = nrm[occ]
    else:
        # 泄漏口径: 用全量
        nrm_use = nrm

    K_eff = min(K, len(nrm_use))
    if K_eff < 2:
        # 退化: 单组 (小井边界; 语义反转 bug 已修 2026-08-29)
        # occ 为 True 表示"已观测"; 此刻全部已观测点退回唯一组。
        # assign 语义与常规分支统一: >=0 入组 (此处全为 0), -1 表示未观测。
        center = _sign_align(nrm_use).mean(0) if len(nrm_use) > 0 else np.zeros(3)
        center = center / (np.linalg.norm(center) + 1e-12)
        assign = np.full(len(nrm), -1, dtype=int)
        assign[occ] = 0                       # 修正: 已观测点入组 (原 ~occ 反向, 休眠 bug)
        # 修正: 报已观测点数 (bincount 仅统计已入组点), 不再错报 K 值
        n_points = np.array([int(np.sum(occ))])
        return {"centers": center[None], "assign": assign, "n_points": n_points}

    cents, assign_obs = spherical_kmeans(nrm_use, K_eff, seed=seed)

    # 组模态: 符号对齐均值
    modes = np.zeros((K_eff, 3))
    for k in range(K_eff):
        sel = assign_obs == k
        if sel.sum():
            pts = nrm_use[sel]
            pts_s = _sign_align(pts)
            modes[k] = pts_s.mean(0)
            modes[k] /= np.linalg.norm(modes[k]) + 1e-12

    # 全量指派 (含隐伏点)
    assign_all = np.abs(nrm @ modes.T).argmax(1)

    # 每组点数
    n_points = np.bincount(assign_all, minlength=K_eff)

    return {"centers": modes, "assign": assign_all, "n_points": n_points}


# ---------------------------------------------------------------------------
# 3. 完整评测管道
# ---------------------------------------------------------------------------

def evaluate_set_table(pred_centers: np.ndarray, truth_centers: np.ndarray,
                       pred_assign: Optional[np.ndarray] = None,
                       truth_assign: Optional[np.ndarray] = None,
                       match_threshold: float = 30.0) -> dict:
    """完整评测管道 (别名, 与 set_table_score 同)。"""
    return set_table_score(pred_centers, truth_centers, pred_assign, truth_assign,
                           match_threshold)


# ---------------------------------------------------------------------------
# 4. 辅助函数
# ---------------------------------------------------------------------------

def _adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand Index (点级划分相似度)。"""
    n = len(labels_a)
    if n < 2:
        return 0.0

    # 列联表
    classes_a = np.unique(labels_a)
    classes_b = np.unique(labels_b)
    contingency = np.zeros((len(classes_a), len(classes_b)), dtype=int)
    for i, ca in enumerate(classes_a):
        for j, cb in enumerate(classes_b):
            contingency[i, j] = int(np.sum((labels_a == ca) & (labels_b == cb)))

    # ARI 计算
    sum_comb_c = sum(_n_choose_2(int(c)) for c in contingency.ravel())
    sum_comb_rows = sum(_n_choose_2(int(r)) for r in contingency.sum(axis=1))
    sum_comb_cols = sum(_n_choose_2(int(c)) for c in contingency.sum(axis=0))
    sum_comb_n = _n_choose_2(n)

    if sum_comb_n == 0:
        return 0.0
    expected = sum_comb_rows * sum_comb_cols / sum_comb_n
    numerator = sum_comb_c - expected
    denominator = 0.5 * (sum_comb_rows + sum_comb_cols) - expected
    if abs(denominator) < 1e-10:
        return 1.0 if abs(numerator) < 1e-10 else 0.0
    return float(numerator / denominator)


def _n_choose_2(n: int) -> int:
    return n * (n - 1) // 2


# ---------------------------------------------------------------------------
# 5. Bootstrap 置信区间 (B4)
# ---------------------------------------------------------------------------

def bootstrap_modal_ci(nrm_group: np.ndarray, n_boot: int = 200,
                       confidence: float = 0.95, seed: int = 0) -> dict:
    """对单组模态做 bootstrap, 返回 95% CI。

    参数:
        nrm_group: (M, 3) 该组观测法向
        n_boot: bootstrap 重采样次数
        confidence: 置信水平
        seed: 随机种子

    返回:
        center: 原始模态方向
        ci_low, ci_high: 置信区间 (度, 与 center 的角距)
        ci_half_width: CI 半宽 (度)
    """
    from .setlabel import _sign_align

    nrm_group = np.asarray(nrm_group, dtype=np.float64)
    nrm_group = nrm_group / (np.linalg.norm(nrm_group, axis=1, keepdims=True) + 1e-12)
    M = len(nrm_group)
    if M < 2:
        center = nrm_group.mean(0) if M > 0 else np.zeros(3)
        center = center / (np.linalg.norm(center) + 1e-12)
        return {"center": center, "ci_low": 0.0, "ci_high": 0.0,
                "ci_half_width": 0.0, "n_points": M}

    # 原始模态
    center = _sign_align(nrm_group).mean(0)
    center = center / (np.linalg.norm(center) + 1e-12)

    rng = np.random.default_rng(seed)
    boot_angles = []
    for _ in range(n_boot):
        idx = rng.integers(0, M, size=M)
        boot_sample = nrm_group[idx]
        boot_center = _sign_align(boot_sample).mean(0)
        boot_center = boot_center / (np.linalg.norm(boot_center) + 1e-12)
        # 角距
        cos = np.clip(np.abs(center @ boot_center), 0, 1)
        angle = float(np.degrees(np.arccos(cos)))
        boot_angles.append(angle)

    boot_angles = np.array(boot_angles)
    alpha = 1 - confidence
    ci_low = float(np.percentile(boot_angles, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_angles, 100 * (1 - alpha / 2)))
    ci_half_width = ci_high

    return {"center": center, "ci_low": ci_low, "ci_high": ci_high,
            "ci_half_width": ci_half_width, "n_points": M}
