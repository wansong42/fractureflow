# -*- coding: utf-8 -*-
"""连通性分析: 渗流曲线 + 场景化指标。

输入: SetTable + P32 网格 + DFN 参数。
输出: p_conn(P32) 渗流曲线 + 拐点 p32_crit + 连通各向异性方向。

物理基础:
  - 排除体积理论 (Balberg 1984): 随机盘状网络 n_c·⟨V_ex⟩ ≈ 2.7
  - ⟨V_ex⟩ 为尺寸分布的二阶矩函数
  - 连通阈值随平均尺寸立方移动 → 尺寸未知时只能区间筛查

场景化指标 (B2):
  - 地热 EGS: 连通体积分数 + 优势连通面 vs 井对连线夹角
  - 矿山突水: 组系 × 巷道轴向 → 高连通概率方位段标记
  - 核废料处置: 垂直渗透逃逸方向优先级

⚠️ 法向/方向约定 (十四期 P0 修复): connectivity_anisotropy 的 dominant_direction
是最大连通分量的**优势面法向**, 不是"通道延伸方向"。连通通道在裂隙面内延伸,
故所有场景判定必须用**面-轴夹角** = 90° − 法向-轴夹角。历史上三个场景函数
直接拿法向算夹角并套阈值, 三处判级全部反向 (水平组系+竖直井对误判"对齐良好"
等), 已于 2026-08 修复并以已知几何单测锁定 (tests/test_percolation_scenarios.py)。

防坑:
  - H1: 把"筛查"说成"判定" → 统一输出 assumptions 字段
  - H5: 理论锚点对不上就调参 → 先查 bug, 不许调参凑曲线
  - H6: 渗流曲线用 3 个实现就出结论 → 每 P32 点 >= 20 实现

用法:
    from fractureflow.percolation import percolation_curve, connectivity_anisotropy
    result = percolation_curve(set_table, p32_grid, beta=3.5, domain=(100,100,100))
    # result = {p32_grid, p_conn, p32_crit, p32_crit_lower, p32_crit_upper, ...}
"""

import numpy as np
from typing import Tuple, Optional
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .dfn import (SetTable, DFNRealization, generate_dfn,
                  build_connectivity_graph, compute_p32)


# ---------------------------------------------------------------------------
# 渗流判定
# ---------------------------------------------------------------------------

def _find_spanning_cluster(adj: sparse.csr_matrix, centers: np.ndarray,
                           domain: Tuple[float, float, float],
                           axis: int = 0) -> Tuple[bool, np.ndarray]:
    """判断是否存在横跨域的连通分量 (spanning cluster)。

    判定: 在指定轴方向上, 分量同时包含接近 domain 两壁 (x < -L/2+10%L 与
    x > +L/2-10%L) 的点。

    注意: 必须用工件真实 domain 边界判定 (八期 B7 教训: 数据极值 ≠ domain,
    稀疏 DFN 上数据极值 < domain 会误判贯穿)。generate_dfn 中心 ∈ [-L/2, L/2]。

    返回: (has_spanning, component_labels)
    """
    M = adj.shape[0]
    if M == 0:
        return False, np.empty(0, dtype=int)

    n_comp, labels = connected_components(adj, directed=False)

    L = float(np.asarray(domain)[axis])
    if L < 1e-9:
        return False, labels
    # 真实 domain 边界 (非数据极值)
    lo_thr = -L / 2.0 + 0.1 * L
    hi_thr = L / 2.0 - 0.1 * L

    for c in range(n_comp):
        mask = labels == c
        if mask.sum() < 2:
            continue
        c_coords = centers[mask][:, axis]
        if (c_coords.min() <= lo_thr) and (c_coords.max() >= hi_thr):
            return True, labels

    return False, labels


def _find_largest_cluster_anisotropy(dfn: DFNRealization, labels: np.ndarray) -> np.ndarray:
    """最大连通分量内裂隙法向的聚合方向 (连通通道优势面方向)。"""
    unique_labels = np.unique(labels)
    if len(unique_labels) == 0:
        return np.zeros(3)

    # 找最大分量
    sizes = np.array([(labels == l).sum() for l in unique_labels])
    largest = unique_labels[np.argmax(sizes)]

    mask = labels == largest
    if mask.sum() == 0:
        return np.zeros(3)

    nrms = dfn.normals[mask]
    nrms = nrms / (np.linalg.norm(nrms, axis=1, keepdims=True) + 1e-12)

    # 符号对齐均值 (无向法向)
    ref = nrms[0]
    sgn = np.sign((nrms * ref).sum(-1, keepdims=True))
    sgn[sgn == 0] = 1
    nrms = nrms * sgn

    mean_dir = nrms.mean(0)
    norm = np.linalg.norm(mean_dir)
    if norm < 1e-12:
        return np.zeros(3)
    return mean_dir / norm


# ---------------------------------------------------------------------------
# 渗流曲线
# ---------------------------------------------------------------------------

def percolation_curve(set_table: SetTable, p32_grid: np.ndarray, beta: float,
                      domain: Tuple[float, float, float],
                      seeds: range = range(20),
                      pbc: bool = False,
                      axis: int = 0) -> dict:
    """渗流曲线: 对每个 P32 生成 N 个实现, 统计横跨比例 p_conn(P32)。

    参数:
        set_table: 组系表
        p32_grid: P32 扫描网格 (m²/m³)
        beta: 幂律指数
        domain: 域尺寸 (Lx, Ly, Lz)
        seeds: 随机种子序列 (每 P32 点生成 len(seeds) 个实现)
        pbc: 周期性边界
        axis: 横跨判定轴 (0=x, 1=y, 2=z)

    返回:
        {
            'p32_grid': array,
            'p_conn': array (与 p32_grid 等长),
            'p32_crit': float (p_conn=0.5 对应的 P32),
            'p32_crit_lower': float (p_conn=0.1 对应的 P32),
            'p32_crit_upper': float (p_conn=0.9 对应的 P32),
            'n_realizations': int,
            'assumptions': str,
        }
    """
    p32_grid = np.asarray(p32_grid)
    p_conn = np.zeros_like(p32_grid, dtype=float)
    n_seeds = len(seeds)

    for i, p32 in enumerate(p32_grid):
        n_spanning = 0
        for s in seeds:
            dfn = generate_dfn(set_table, p32, beta, domain, seed=s)
            if dfn.n_fractures == 0:
                continue
            G = build_connectivity_graph(dfn, pbc=pbc, domain=domain)
            has_span, _ = _find_spanning_cluster(G, dfn.centers, domain, axis=axis)
            if has_span:
                n_spanning += 1
        p_conn[i] = n_spanning / max(n_seeds, 1)

    # 拐点估计 (p_conn = 0.5)
    p32_crit = _interp_threshold(p32_grid, p_conn, 0.5)
    p32_crit_lower = _interp_threshold(p32_grid, p_conn, 0.1)
    p32_crit_upper = _interp_threshold(p32_grid, p_conn, 0.9)

    return {
        'p32_grid': p32_grid,
        'p_conn': p_conn,
        'p32_crit': p32_crit,
        'p32_crit_lower': p32_crit_lower,
        'p32_crit_upper': p32_crit_upper,
        'n_realizations': n_seeds,
        'assumptions': (
            f"Baecher 盘模型, 幂律 β={beta}, 域 {domain}m, "
            f"pbc={pbc}, 横跨轴={axis}, 每 P32 点 {n_seeds} 实现. "
            f"筛查级结论: 基于组系几何 + 假设尺寸分布, 非真实连通性判定."
        ),
    }


def _interp_threshold(x: np.ndarray, y: np.ndarray, y_target: float) -> float:
    """在 (x, y) 曲线上插值求 y=y_target 对应的 x。"""
    # 单调递增
    if y.max() < y_target:
        return float(x[-1])  # 未达到
    if y.min() > y_target:
        return float(x[0])   # 已超过
    # 找交叉点
    for i in range(len(y) - 1):
        if y[i] <= y_target <= y[i + 1]:
            # 线性插值
            t = (y_target - y[i]) / (y[i + 1] - y[i] + 1e-12)
            return float(x[i] + t * (x[i + 1] - x[i]))
    return float(x[len(y) // 2])


# ---------------------------------------------------------------------------
# 连通各向异性
# ---------------------------------------------------------------------------

def connectivity_anisotropy(dfn: DFNRealization, pbc: bool = False) -> dict:
    """连通各向异性分析。

    返回最大连通分量的优势方向 + 连通分量统计。

    返回:
        {
            'dominant_direction': (3,) 优势面法向,
            'largest_fraction': float (最大分量占全部分数),
            'n_components': int,
            'component_sizes': list[int],
        }
    """
    G = build_connectivity_graph(dfn, pbc=pbc, domain=dfn.domain)
    n_comp, labels = connected_components(G, directed=False)

    if dfn.n_fractures == 0:
        return {
            'dominant_direction': np.zeros(3),
            'largest_fraction': 0.0,
            'n_components': 0,
            'component_sizes': [],
        }

    unique, counts = np.unique(labels, return_counts=True)
    largest_idx = np.argmax(counts)
    largest_size = counts[largest_idx]

    dominant_dir = _find_largest_cluster_anisotropy(dfn, labels)

    # 最大分量内的法向玫瑰
    return {
        'dominant_direction': dominant_dir,
        'largest_fraction': float(largest_size / dfn.n_fractures),
        'n_components': int(n_comp),
        'component_sizes': counts.tolist(),
    }


# ---------------------------------------------------------------------------
# 理论锚点: 排除体积
# ---------------------------------------------------------------------------

def exclusion_volume_threshold(dfn: DFNRealization, domain: Tuple[float, float, float]) -> dict:
    """计算 n·⟨V_ex⟩ 与理论值 2.7 比较 (自检)。

    对于球体近似: V_ex(i,j) = (4/3)π(r_i + r_j)³
    ⟨V_ex⟩ = mean over all pairs

    ⚠️ 理论口径注记 (2026-08-28 诚实化): Balberg n_c·⟨V_ex⟩≈2.7 的 ⟨V_ex⟩ 是
    **随机取向薄圆盘**的排除体积 (∝π²⟨r⟩³), 与本处球体近似不可比 ——
    本函数输出仅作趋势级自检 (单调性/量级), 'deviation' 数值**不得**当作
    "物理基础已验证到 2.7" 的证据引用。

    返回:
        {
            'n_ve': float (实测 n·⟨V_ex⟩, 球体近似口径),
            'n_density': float (数密度 1/m³),
            'mean_ve': float (平均排除体积, 球体近似),
            'theoretical': 2.7,
            'deviation': float (与 2.7 的差; 见上方口径注记),
        }
    """
    V = domain[0] * domain[1] * domain[2]
    M = dfn.n_fractures
    if M < 2 or V <= 0:
        return {'n_ve': 0.0, 'n_density': 0.0, 'mean_ve': 0.0,
                'theoretical': 2.7, 'deviation': -2.7}

    n_density = M / V

    # 采样估计 ⟨V_ex⟩ (pairwise, 用随机子集加速)
    n_sample = min(M, 500)
    rng = np.random.default_rng(0)
    idx = rng.choice(M, n_sample, replace=False)
    radii_sample = dfn.radii[idx]

    # 配对排除体积
    ve_list = []
    for i in range(n_sample):
        for j in range(i + 1, n_sample):
            ve_list.append((4.0 / 3.0) * np.pi * (radii_sample[i] + radii_sample[j]) ** 3)

    mean_ve = float(np.mean(ve_list)) if ve_list else 0.0
    n_ve = n_density * mean_ve

    return {
        'n_ve': n_ve,
        'n_density': n_density,
        'mean_ve': mean_ve,
        'theoretical': 2.7,
        'deviation': n_ve - 2.7,
    }


# ---------------------------------------------------------------------------
# 场景化指标 (B2)
#
# 统一几何约定: dominant_direction 是优势面**法向**; 判定用面-轴夹角
# plane_axis_deg = 90° − 法向-轴夹角 (轴越"躺"在连通面内, 夹角越小)。
# ---------------------------------------------------------------------------

def _plane_axis_angle_deg(dominant_normal: np.ndarray, axis: np.ndarray) -> float:
    """优势面(法向)与轴的面-轴夹角 (度): 90° − |n·a| 对应的法向夹角。

    返回 0° 表示轴完全躺在面内 (通道沿轴贯通最有利), 90° 表示轴垂直于面。
    """
    c = float(np.clip(abs(np.dot(dominant_normal, axis)), 0.0, 1.0))
    return 90.0 - float(np.degrees(np.arccos(c)))


def _anisotropic(dominant_normal: np.ndarray) -> bool:
    """dominant_direction 近零向量 => 最大分量内无优势方向 (各向同性)."""
    return float(np.linalg.norm(dominant_normal)) > 1e-6


def egs_connectivity_metric(dfn: DFNRealization, well_pair_axis: np.ndarray,
                            pbc: bool = False) -> dict:
    """地热 EGS 连通指标。

    well_pair_axis: (3,) 井对连线方向 (单位向量)。
    判定语义: 井对轴线是否落在最大连通分量的优势裂隙面内 —— 面内对齐
    (面-轴夹角小) 时压裂通道才能串起两井。

    返回: 连通体积分数 + 优势面 vs 井对连线夹角
    ('angle_to_well_pair_deg' = 面-轴夹角; 'normal_axis_angle_deg' = 法向-轴夹角)。
    """
    aniso = connectivity_anisotropy(dfn, pbc=pbc)
    dominant = aniso['dominant_direction']
    well_pair_axis = np.asarray(well_pair_axis, dtype=float)
    well_pair_axis = well_pair_axis / (np.linalg.norm(well_pair_axis) + 1e-12)

    angle_deg = _plane_axis_angle_deg(dominant, well_pair_axis)
    normal_angle_deg = 90.0 - angle_deg

    isotropic = not _anisotropic(dominant)

    return {
        'metric_name': 'EGS_connectivity',
        'connectivity_fraction': aniso['largest_fraction'],
        'dominant_direction': dominant,
        'well_pair_axis': well_pair_axis,
        'angle_to_well_pair_deg': None if isotropic else angle_deg,
        'normal_axis_angle_deg': None if isotropic else normal_angle_deg,
        'assessment': (
            '各向同性 (无优势连通方向)' if isotropic else
            '对齐良好' if angle_deg < 30 else
            '中等对齐' if angle_deg < 60 else
            '对齐较差 (建议调整压裂方位)'
        ),
        'uncertainty_band': '基于假设尺寸分布, 需现场试验标定',
        'assumptions': (
            f"Baecher 盘, PBC={pbc}. "
            f"夹角=优势连通面-井对轴夹角 (由法向换算 90°−法向角); "
            f"<30° 良好 / <60° 中等 / ≥60° 较差."),
    }


def mine_risk_sections(dfn: DFNRealization, tunnel_axis: np.ndarray,
                       tunnel_dip: float = 0.0, pbc: bool = False) -> dict:
    """矿山/隧洞 突水通道 风险段标记。

    tunnel_axis: (3,) 巷道轴向 (单位向量)。
    tunnel_dip: 巷道倾角 (度)。
    判定语义: 巷道轴向落在最大连通分量的优势裂隙面内 (面-轴夹角小)
    → 连通通道沿巷道贯通 → 高突水风险; 轴垂直于优势面 → 低。

    返回: 组系 × 巷道轴向 → 高连通概率方位段。
    """
    aniso = connectivity_anisotropy(dfn, pbc=pbc)
    tunnel_axis = np.asarray(tunnel_axis, dtype=float)
    tunnel_axis = tunnel_axis / (np.linalg.norm(tunnel_axis) + 1e-12)

    dominant = aniso['dominant_direction']
    angle_deg = _plane_axis_angle_deg(dominant, tunnel_axis)
    normal_angle_deg = 90.0 - angle_deg
    isotropic = not _anisotropic(dominant)

    # 风险判定: 巷道轴向躺在连通面内 → 高风险 (沿巷道贯通)
    if isotropic:
        risk = '不适用 (各向同性)'
    elif angle_deg < 30:
        risk = '高'
    elif angle_deg < 60:
        risk = '中'
    else:
        risk = '低'

    return {
        'metric_name': 'mine_water_inrush_risk',
        'connectivity_fraction': aniso['largest_fraction'],
        'dominant_direction': dominant,
        'tunnel_axis': tunnel_axis,
        'angle_to_tunnel_deg': None if isotropic else angle_deg,
        'normal_axis_angle_deg': None if isotropic else normal_angle_deg,
        'risk_level': risk,
        'uncertainty_band': '筛查级, 非稳定性结论',
        'assumptions': (
            f"Baecher 盘, PBC={pbc}, 巷道倾角 {tunnel_dip}°. "
            f"夹角=优势连通面-巷道轴夹角; <30° 高(沿巷道贯通) / "
            f"<60° 中 / ≥60° 低."),
    }


def disposal_escape_priority(dfn: DFNRealization,
                             vertical: np.ndarray = np.array([0.0, 0.0, 1.0]),
                             pbc: bool = False) -> dict:
    """核废料处置 垂直渗透逃逸方向优先级。

    垂直方向 (默认 z): 核素重力迁移主方向。
    判定语义: 垂直方向落在最大连通分量的优势裂隙面内 (面-轴夹角小,
    即陡立连通面) → 提供近垂直逃逸通道 → 高优先级; 水平连通面 → 低。

    返回: 连通方向 × 垂直方向 → 逃逸优先级。
    """
    aniso = connectivity_anisotropy(dfn, pbc=pbc)
    vertical = np.asarray(vertical, dtype=float)
    vertical = vertical / (np.linalg.norm(vertical) + 1e-12)

    dominant = aniso['dominant_direction']
    angle_deg = _plane_axis_angle_deg(dominant, vertical)
    normal_angle_deg = 90.0 - angle_deg
    isotropic = not _anisotropic(dominant)

    # 垂直方向躺在陡立连通面内 → 高逃逸优先级
    if isotropic:
        priority = '不适用 (各向同性)'
    elif angle_deg < 30:
        priority = '高'
    elif angle_deg < 60:
        priority = '中'
    else:
        priority = '低'

    return {
        'metric_name': 'disposal_escape_priority',
        'connectivity_fraction': aniso['largest_fraction'],
        'dominant_direction': dominant,
        'vertical_direction': vertical,
        'angle_to_vertical_deg': None if isotropic else angle_deg,
        'normal_axis_angle_deg': None if isotropic else normal_angle_deg,
        'escape_priority': priority,
        'uncertainty_band': '筛查级, 需耦合运移模拟确认',
        'assumptions': (
            f"Baecher 盘, PBC={pbc}. "
            f"夹角=优势连通面-垂直方向夹角 (陡立连通面→高逃逸); "
            f"<30° 高 / <60° 中 / ≥60° 低."),
    }
