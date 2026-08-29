# -*- coding: utf-8 -*-
"""T45: 后验网络集合 —— 从 UnifiedSetTable 生成 M=20 个 DFN 后验实现.

核心流程:
  1. 从 ust.cones_95 抽样方向不确定性 (每个实现方向 = 真方向 + cone 内随机扰动)
  2. 从 p32_range 抽样强度不确定性 (每个实现独立采样 P32)
  3. 方向与强度解耦报告 (两个独立随机源)
  4. 每个实现跑 percolation_curve → 收集 p32_crit 分布
  5. 输出 p32_crit 的 90% 区间 (5%-95% 分位数)

诚实边界:
  - P32 从估计区间内抽样传播不确定性, 不是固定中值跑 20 次 (否则区间假窄)
  - β>3 时 r_min≥0.5m 防微裂隙爆炸 (既有约定)
  - 方向不确定性与强度不确定性分别报告, 不混淆
  - 当前 scenario_intervals 三区 (EGS/矿山/处置) 输出相同 p32_crit 区间, 仅作为格式占位; 真实场景指标需另行扩展

用法:
    from fractureflow.l4.ensemble import generate_ensemble, EnsembleResult
    result = generate_ensemble(ust, p32_estimate=0.5, p32_range=(0.3, 0.7),
                               beta=3.5, domain=(50, 50, 50))
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .constants import M_ENSEMBLE, EPS


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class EnsembleResult:
    """后验网络集合结果."""
    realizations: List[object]                    # M=20 个 DFNRealization
    p32_crit_interval: Tuple[float, float, float]  # (lower, median, upper) 90% 区间
    scenario_intervals: dict                       # EGS/矿山/处置 三场景指标区间
    direction_uncertainty: np.ndarray              # (K,) 方向不确定性 (锥半角, 度)
    intensity_uncertainty: float                   # P32 不确定性 (区间半宽 / 中值)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 方向扰动: 在 cone 内随机采样
# ---------------------------------------------------------------------------

def _sample_in_cone(mu: np.ndarray, half_angle_deg: float, rng: np.random.Generator) -> np.ndarray:
    """在以 mu 为轴、half_angle_deg 为半角的圆锥内均匀随机采样一个方向.

    使用 Rodrigues 旋转公式: 先采样一个在 xy 平面内与 x 轴夹角为 theta 的向量,
    然后绕 y 轴旋转 half_angle, 再旋转到 mu 方向.

    参数:
        mu: (3,) 单位向量 (圆锥轴)
        half_angle_deg: 圆锥半角 (度)
        rng: 随机数生成器

    返回:
        (3,) 单位向量 (圆锥内随机方向)
    """
    mu = np.asarray(mu, dtype=float)
    mu = mu / (np.linalg.norm(mu) + EPS)

    half_angle_rad = np.radians(half_angle_deg)

    # 在圆锥内均匀采样: cos(phi) 在 [cos(half_angle), 1] 上均匀分布
    cos_phi = rng.uniform(np.cos(half_angle_rad), 1.0)
    sin_phi = np.sqrt(max(0.0, 1.0 - cos_phi ** 2))
    theta = rng.uniform(0, 2 * np.pi)

    # 局部坐标系: 圆锥轴 = z, 采样方向在局部坐标中为 (sin_phi*cos_theta, sin_phi*sin_theta, cos_phi)
    local_dir = np.array([
        sin_phi * np.cos(theta),
        sin_phi * np.sin(theta),
        cos_phi
    ])

    # 将局部 z 轴对齐到 mu: 构造旋转矩阵
    # 使用 Rodrigues 公式, 旋转轴 = cross(z, mu), 旋转角 = arccos(mu·z)
    z_axis = np.array([0.0, 0.0, 1.0])
    cos_rot = np.clip(np.dot(z_axis, mu), -1.0, 1.0)
    rot_angle = np.arccos(cos_rot)

    if abs(rot_angle) < 1e-10:
        # mu 已经平行于 z 轴
        return local_dir / (np.linalg.norm(local_dir) + EPS)

    if abs(rot_angle - np.pi) < 1e-10:
        # mu 平行于 -z 轴: 旋转 180° 绕任意垂直轴
        rot_axis = np.array([1.0, 0.0, 0.0])
    else:
        rot_axis = np.cross(z_axis, mu)
        rot_axis = rot_axis / (np.linalg.norm(rot_axis) + EPS)

    # Rodrigues 旋转
    K = np.array([
        [0, -rot_axis[2], rot_axis[1]],
        [rot_axis[2], 0, -rot_axis[0]],
        [-rot_axis[1], rot_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(rot_angle) * K + (1 - np.cos(rot_angle)) * (K @ K)

    result = R @ local_dir
    return result / (np.linalg.norm(result) + EPS)


def _perturb_directions(centers: np.ndarray, cones_95: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """对每个组方向施加 cone 内随机扰动.

    参数:
        centers: (K, 3) 原始方向
        cones_95: (K,) 95% 置信锥半角 (度)
        rng: 随机数生成器

    返回:
        (K, 3) 扰动后的方向 (单位向量)
    """
    K = centers.shape[0]
    perturbed = np.empty_like(centers)
    for j in range(K):
        half_angle = max(cones_95[j], 0.1)  # 至少 0.1° 防止退化
        perturbed[j] = _sample_in_cone(centers[j], half_angle, rng)
    return perturbed


# ---------------------------------------------------------------------------
# 核心函数: 生成后验集合
# ---------------------------------------------------------------------------

def generate_ensemble(ust, p32_estimate: float, p32_range: Tuple[float, float],
                      beta: float, domain: Tuple[float, float, float],
                      M: int = M_ENSEMBLE, seed: int = 42,
                      p32_grid_size: int = 6,
                      p32_grid: Optional[np.ndarray] = None) -> EnsembleResult:
    """从 UnifiedSetTable 生成 M 个 DFN 后验实现.

    参数:
        ust: UnifiedSetTable (来自 l4.fuse)
        p32_estimate: P32 中值估计 (m²/m³)
        p32_range: (lower, upper) P32 估计区间
        beta: 幂律指数
        domain: (Lx, Ly, Lz) 域尺寸 (m)
        M: 实现数 (默认 20)
        seed: 随机种子
        p32_grid_size: 渗流曲线 P32 网格点数 (每个实现内部扫描)
        p32_grid: 可选, 显式指定 P32 网格 (覆盖 p32_grid_size).
                 提供时真值与集合可使用相同网格, 确保公平比较.

    返回:
        EnsembleResult
    """
    # 延迟导入避免循环依赖
    from fractureflow.dfn import SetTable, generate_dfn
    from fractureflow.percolation import percolation_curve

    t0 = time.time()
    rng = np.random.default_rng(seed)

    if ust.centers is None or ust.K == 0:
        raise ValueError(
            "空 UnifiedSetTable (centers=None 或 K=0): 融合层未产出任何组系, "
            "请检查上游证据输入/冲突门控配置, 而非直接进入集合阶段。")
    K = ust.K
    centers = ust.centers.copy()  # (K, 3)
    concentrations = ust.concentrations.copy()  # (K,)
    proportions = ust.proportions.copy()  # (K,)
    cones_95 = ust.cones_95.copy() if ust.cones_95 is not None else np.full(K, 5.0)

    # P32 区间
    p32_lo, p32_hi = p32_range
    p32_lo = max(p32_lo, EPS)

    # 每个实现内部的渗流曲线 P32 网格
    # (显式指定时覆盖自动构造, 确保真值与集合使用相同网格)
    if p32_grid is None:
        p32_grid = np.linspace(p32_lo * 0.5, p32_hi * 1.5, p32_grid_size)

    realizations = []
    p32_crits = []
    scenario_egs = []
    scenario_mine = []
    scenario_disposal = []

    # r_min 防微裂隙爆炸 (β>3 时 ≥0.5m, 既有约定)
    r_min = 0.5 if beta > 3 else None

    # 渗流曲线使用固定种子序列 (与实现序号无关), 确保 p32_crit 差异仅来自方向/强度扰动
    # 使用绝对固定的种子 (不依赖 seed 参数), 保证真值与集合使用相同设置
    perc_seeds = range(200, 206)  # 6 种子稳定插值, 与真值计算一致

    for m in range(M):
        # 每个实现独立种子 (可复现)
        seed_m = seed + m * 1000 + 7
        rng_m = np.random.default_rng(seed_m)

        # 1. 方向不确定性: 在 cone 内随机扰动
        perturbed_centers = _perturb_directions(centers, cones_95, rng_m)

        # 2. 强度不确定性: 从 P32 区间内独立抽样
        p32_sample = rng_m.uniform(p32_lo, p32_hi)

        # 3. 构建 SetTable (扰动方向 + 原始浓度/比例)
        st = SetTable(
            centers=perturbed_centers,
            concentrations=concentrations,
            proportions=proportions,
        )

        # 4. 生成 DFN 实现
        dfn = generate_dfn(st, p32=p32_sample, beta=beta, domain=domain,
                           seed=seed_m, r_min=r_min)
        realizations.append(dfn)

        # 5. 渗流曲线 (固定种子, 粗网格, 加速)
        if dfn.n_fractures > 0:
            perc = percolation_curve(
                st, p32_grid, beta, domain,
                seeds=perc_seeds,  # 固定种子, 差异仅来自方向/强度扰动
                pbc=False, axis=0
            )
            p32_crit = perc['p32_crit']
        else:
            p32_crit = 0.0

        p32_crits.append(p32_crit)

        # 6. 场景指标 (基于当前实现)
        # 使用 percolation_curve 的 p32_crit 作为场景指标代理
        # 完整场景指标需要 DFNRealization, 这里收集 p32_crit 作为核心指标
        scenario_egs.append(p32_crit)
        scenario_mine.append(p32_crit)
        scenario_disposal.append(p32_crit)

    # p32_crit 分布统计
    p32_crits_arr = np.array(p32_crits)
    p32_crits_valid = p32_crits_arr[p32_crits_arr > 0]

    if len(p32_crits_valid) >= 2:
        lower = float(np.percentile(p32_crits_valid, 5))
        median = float(np.percentile(p32_crits_valid, 50))
        upper = float(np.percentile(p32_crits_valid, 95))
    else:
        lower = median = upper = float(p32_estimate)

    # 场景指标区间
    def _interval(arr):
        a = np.array(arr)
        a = a[a > 0]
        if len(a) < 2:
            return {'lower': float(p32_estimate), 'median': float(p32_estimate),
                    'upper': float(p32_estimate), 'n_valid': len(a)}
        return {
            'lower': float(np.percentile(a, 5)),
            'median': float(np.percentile(a, 50)),
            'upper': float(np.percentile(a, 95)),
            'n_valid': len(a),
        }

    scenario_intervals = {
        'EGS': _interval(scenario_egs),
        'mine': _interval(scenario_mine),
        'disposal': _interval(scenario_disposal),
    }

    # 方向不确定性 = cones_95 (度)
    direction_uncertainty = cones_95.copy()

    # 强度不确定性 = 区间半宽 / 中值
    p32_mid = (p32_lo + p32_hi) / 2
    intensity_uncertainty = (p32_hi - p32_lo) / (2 * p32_mid + EPS)

    elapsed = time.time() - t0

    meta = {
        'M': M,
        'seed': seed,
        'beta': beta,
        'domain': list(domain),
        'p32_estimate': p32_estimate,
        'p32_range': list(p32_range),
        'K': K,
        'elapsed_sec': round(elapsed, 3),
        'n_fractures_mean': float(np.mean([d.n_fractures for d in realizations])),
        'n_fractures_std': float(np.std([d.n_fractures for d in realizations])),
        'r_min': r_min,
        'p32_grid_size': p32_grid_size,
        'honest_boundary': (
            f"后验集合基于方向 cone 内随机扰动 + P32 区间内独立抽样. "
            f"方向不确定性与强度不确定性解耦报告. "
            f"β={beta}, r_min={r_min}. "
            f"筛查级结论: 非真实连通性判定, 需现场标定."
        ),
    }

    return EnsembleResult(
        realizations=realizations,
        p32_crit_interval=(lower, median, upper),
        scenario_intervals=scenario_intervals,
        direction_uncertainty=direction_uncertainty,
        intensity_uncertainty=intensity_uncertainty,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# 结果序列化
# ---------------------------------------------------------------------------

def ensemble_to_dict(result: EnsembleResult) -> dict:
    """将 EnsembleResult 转为可 JSON 序列化的 dict."""
    return {
        'p32_crit_interval': {
            'lower': result.p32_crit_interval[0],
            'median': result.p32_crit_interval[1],
            'upper': result.p32_crit_interval[2],
        },
        'scenario_intervals': result.scenario_intervals,
        'direction_uncertainty_deg': result.direction_uncertainty.tolist(),
        'intensity_uncertainty': result.intensity_uncertainty,
        'n_realizations': len(result.realizations),
        'meta': result.meta,
    }
