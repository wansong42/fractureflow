# -*- coding: utf-8 -*-
"""进阶市场模块: 同一张组系表, 多回答客户几个问题.

全部是确定性几何/线性代数, 禁止上学习类模型.
每个函数 ≤50 行核心代码, 纯 numpy.

模块:
    - oda_permeability_tensor: Oda 渗透张量 (T8)
    - slope_kinematic_screening: 边坡运动学筛查 Markland 准则 (T10)
    - egs_fracture_orientation: EGS 压裂方位建议 (T11)
    - p10_p32_intensity: P10/P32 强度产品化 (T9, 包装已有函数)
"""
import numpy as np
from typing import Optional, Tuple


# ============================================================
# T8: Oda 渗透张量
# ============================================================

def oda_permeability_tensor(set_table: dict, p32_total: float,
                             aperture_mm: float = 0.1) -> dict:
    """Oda 渗透张量 (T8).

    公式 (写死, 不要自己发明):
        k_ij ∝ Σ_s w_s · (δ_ij − n_i · n_j)
        w_s = P32_s · b³ / 12
        P32_s = 该组条数占比 × 总 P32
        b = 裂隙开度 (mm, 缺省 0.1, 写入 assumptions)

    参数:
        set_table: {
            "centers": (K, 3) 组模态法向 (单位向量),
            "n_points": (K,) 每组条数,
        }
        p32_total: 总 P32 (裂隙数/m²)
        aperture_mm: 裂隙开度 (mm), 缺省 0.1

    返回:
        tensor: (3, 3) 渗透张量 (未归一化, 量级 ∝ P32·b³)
        eigenvalues: (3,) 主值 (从大到小)
        eigenvectors: (3, 3) 主轴 (列向量, 每列一个主方向)
        anisotropy_ratio: 各向异性比 k_max / k_min
        principal_directions: [{"trend": 倾向方位(°), "dip": 倾角(°)}, ...]
        assumptions: list
    """
    centers = np.asarray(set_table["centers"], dtype=float)
    n_points = np.asarray(set_table["n_points"], dtype=float)
    K = len(centers)

    # 归一化法向
    nrm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)

    # 各组 P32 占比
    frac = n_points / (n_points.sum() + 1e-12)
    p32_s = frac * p32_total

    # 开度 (m)
    b_m = aperture_mm * 1e-3

    # 权重 w_s = P32_s · b³ / 12
    w_s = p32_s * (b_m ** 3) / 12.0

    # 张量求和
    tensor = np.zeros((3, 3))
    for s in range(K):
        n = nrm[s]
        tensor += w_s[s] * (np.eye(3) - np.outer(n, n))

    # 特征分解
    eigvals, eigvecs = np.linalg.eigh(tensor)
    # eigh 返回从小到大, 我们需要从大到小
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 归一化: 绝对值太小 (b³ 量级), 返回相对张量 + 量级因子
    scale_factor = float(np.max(np.abs(eigvals)))
    if scale_factor > 1e-30:
        tensor_normalized = tensor / scale_factor
        eigvals_normalized = eigvals / scale_factor
    else:
        tensor_normalized = tensor
        eigvals_normalized = eigvals

    # 各向异性比
    anisotropy = eigvals_normalized[0] / (eigvals_normalized[2] + 1e-30)

    # 主轴方向 (倾向/倾角)
    principal_dirs = []
    for i in range(3):
        v = eigvecs[:, i]
        v = v / (np.linalg.norm(v) + 1e-12)
        # 特征向量无向 (±v 等价), 保证 z > 0 使倾角 ∈ [0, 90]
        # 法向无向, 倾角定义为与水平面夹角, 必须取 abs (与 T10/T11 同口径)
        if v[2] < 0:
            v = -v
        dip = np.degrees(np.arccos(np.clip(v[2], 0, 1)))
        az = np.degrees(np.arctan2(v[0], v[1])) % 360
        principal_dirs.append({"trend": round(az, 2), "dip": round(dip, 2)})

    return {
        "tensor": tensor_normalized,
        "tensor_scale_factor": scale_factor,
        "eigenvalues": eigvals_normalized.tolist(),
        "eigenvectors": eigvecs.tolist(),
        "anisotropy_ratio": round(anisotropy, 3),
        "principal_directions": principal_dirs,
        "assumptions": [
            f"裂隙开度 b = {aperture_mm} mm (缺省假设, 用户提供可替换)",
            "Oda 张量基于贯通裂隙假设, 未考虑连通性与粗糙度",
            f"总 P32 = {p32_total} m^-2",
            f"张量已归一化 (量级因子 {scale_factor:.2e})",
        ],
    }


# ============================================================
# T10: 边坡运动学筛查 (Markland 准则)
# ============================================================

def slope_kinematic_screening(set_table: dict,
                               slope_trend: float, slope_dip: float,
                               friction_angle_deg: float = 30.0) -> dict:
    """边坡运动学筛查 (T10, Markland 准则).

    平面滑动判据:
        |组走向 − 坡面走向| < (90° − φ)  且  φ < 组倾角 < 坡角

    楔形破坏判据:
        交线倾向与坡向差 < (90° − φ)  且  交线倾角 > φ
        (交线 = 两组法向叉积)

    参数:
        set_table: {"centers": (K, 3) 法向 (单位向量)}
        slope_trend: 坡面倾向方位角 (°)
        slope_dip: 坡面倾角 (°)
        friction_angle_deg: 内摩擦角 φ (°), 缺省 30

    返回:
        results: list of dict, 每组/每组对 {
            "type": "planar"/"wedge"/"stable",
            "detail": str
        }
        assumptions: list
    """
    centers = np.asarray(set_table["centers"], dtype=float)
    nrm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    phi = friction_angle_deg
    tolerance = 90.0 - phi

    # 组倾角 (与水平面夹角, 0-90°, 取 abs 法向 z 分量)
    group_dips = np.degrees(np.arccos(np.clip(np.abs(nrm[:, 2]), 0, 1)))
    group_strikes = (np.degrees(np.arctan2(nrm[:, 0], nrm[:, 1])) - 90) % 360

    # 坡面走向 = 坡面倾向 − 90° (Markland 准则用走向比较)
    slope_strike = (slope_trend - 90) % 360

    results = []

    # 平面滑动: 逐组判断
    for k in range(len(nrm)):
        strike_diff = abs(group_strikes[k] - slope_strike)
        strike_diff = min(strike_diff, 360 - strike_diff)
        dip_k = group_dips[k]

        if strike_diff < tolerance and phi < dip_k < slope_dip:
            results.append({
                "type": "planar",
                "set_id": k,
                "detail": f"组{k}: 走向差={strike_diff:.1f}° < {tolerance:.1f}°, "
                          f"倾角={dip_k:.1f}° ∈ ({phi:.1f}°, {slope_dip:.1f}°)",
            })

    # 楔形破坏: 两两组合
    for i in range(len(nrm)):
        for j in range(i + 1, len(nrm)):
            # 交线 = n_i × n_j
            intersection = np.cross(nrm[i], nrm[j])
            int_len = np.linalg.norm(intersection)
            if int_len < 1e-10:
                continue  # 平行, 无交线
            intersection /= int_len

            # 交线俯角 (与水平面夹角, 0-90°, 用 abs(z) 取俯角)
            int_plunge = np.degrees(np.arcsin(np.clip(abs(intersection[2]), 0, 1)))
            int_trend = (np.degrees(np.arctan2(intersection[0], intersection[1]))) % 360

            # 交线走向 = 交线倾向 − 90°
            int_strike = (int_trend - 90) % 360
            strike_diff = abs(int_strike - slope_strike)
            strike_diff = min(strike_diff, 360 - strike_diff)

            # 楔形: 走向差 < tol 且 φ < 俯角 < 坡角
            if strike_diff < tolerance and phi < int_plunge < slope_dip:
                results.append({
                    "type": "wedge",
                    "set_ids": [i, j],
                    "intersection_trend": round(int_trend, 2),
                    "intersection_plunge": round(int_plunge, 2),
                    "detail": f"组{i}-组{j}交线: 倾向={int_trend:.1f}°, "
                              f"俯角={int_plunge:.1f}°, 走向差={strike_diff:.1f}°",
                })

    if not results:
        results.append({"type": "stable", "detail": "无滑动/楔形风险"})

    return {
        "results": results,
        "slope": {"trend": slope_trend, "dip": slope_dip},
        "friction_angle": phi,
        "assumptions": [
            f"内摩擦角 φ = {phi}° (缺省, 用户提供可替换)",
            "筛查级判定, 非稳定性计算",
            "未考虑裂隙连通性、充填物、孔隙水压力",
        ],
    }


# ============================================================
# T11: EGS 压裂方位建议 (几何版 v1)
# ============================================================

def egs_fracture_orientation(set_table: dict, shmax_trend: float,
                               shmax_source: str = "未知") -> dict:
    """EGS 压裂方位建议 (T11, 几何版 v1).

    只做方位几何: 对每组算「走向与 SHmax 夹角 θ」, 按 |θ| 分类:
        ≈平行 (|θ| < 30°): 应力压闭合, 导流差
        ≈垂直 (|θ| > 60°): 易剪切开启, 甜点
        中间: 过渡

    参数:
        set_table: {"centers": (K, 3) 法向}
        shmax_trend: SHmax 方位角 (°)
        shmax_source: SHmax 数据来源标注

    返回:
        ranking: list, 按 |θ| 从大到小排序 (甜点优先)
        assumptions: list
    """
    centers = np.asarray(set_table["centers"], dtype=float)
    nrm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)

    # 组走向 = 倾向 − 90°
    group_trends = (np.degrees(np.arctan2(nrm[:, 0], nrm[:, 1])) - 90) % 360
    # 法向无向: 倾角 = arccos(|nz|) ∈ [0, 90], 与 T10 第 151 行同口径
    group_dips = np.degrees(np.arccos(np.clip(np.abs(nrm[:, 2]), 0, 1)))

    ranking = []
    for k in range(len(nrm)):
        trend_k = group_trends[k]
        theta = abs(trend_k - shmax_trend)
        theta = min(theta, 360 - theta)

        if theta < 30:
            cls = "平行 (导流差)"
            sweet = 0
        elif theta > 60:
            cls = "垂直 (甜点)"
            sweet = 2
        else:
            cls = "过渡"
            sweet = 1

        ranking.append({
            "set_id": k,
            "trend": round(trend_k, 2),
            "dip": round(group_dips[k], 2),
            "theta_to_shmax": round(theta, 2),
            "classification": cls,
            "sweet_score": sweet,
        })

    # 按 sweet_score 降序 (甜点优先)
    ranking.sort(key=lambda x: x["sweet_score"], reverse=True)

    return {
        "shmax_trend": shmax_trend,
        "shmax_source": shmax_source,
        "ranking": ranking,
        "assumptions": [
            f"SHmax 方位 = {shmax_trend}° (来源: {shmax_source})",
            "v1 仅做方位几何, 未含应力大小与孔隙压",
            "完整莫尔-库仑滑动趋势 (Ts=τ/σn) 需真实应力数据, 记入数据升档清单",
        ],
    }


# ============================================================
# T9: P10/P32 强度产品化 (包装已有函数)
# ============================================================

def p10_p32_intensity(net: dict, domain: tuple,
                       trace_lengths: Optional[np.ndarray] = None,
                       spacing: Optional[np.ndarray] = None) -> dict:
    """P10/P32 强度产品化 (T9).

    已有函数:
        - estimate_p32_from_spacing (间距 → P32)
        - fit_beta_from_tracelength (迹长 MLE → β)
        - estimate_p32_interval (编录表区间估计)

    本函数做统一入口 + P10 直接口径 (条数/井段长度).

    参数:
        net: net dict (含 nrm_full, depth_m)
        domain: (Lx, Ly, Lz) 域尺寸 (m)
        trace_lengths: 迹长数组 (可选, 触发 β 拟合)
        spacing: 间距数组 (可选, 触发 P32 直估)

    返回:
        result: {p10, p32, method, assumptions}
    """
    from .dfn import estimate_p32_from_spacing, fit_beta_from_tracelength

    pos = np.asarray(net["pos"], dtype=float)
    n = len(pos)

    # P10 = 条数 / 井段长度 (优先 depth_m, 否则用 pos 最大范围)
    if "depth_m" in net:
        depths = np.asarray(net["depth_m"], dtype=float)
        hole_length = float(depths.max() - depths.min())
    else:
        # pos 可能是 (深度, 0, 0) 一维近似, 用 x 范围; 否则用三轴最大范围
        x_range = pos[:, 0].max() - pos[:, 0].min()
        y_range = pos[:, 1].max() - pos[:, 1].min()
        z_range = pos[:, 2].max() - pos[:, 2].min()
        hole_length = float(max(x_range, y_range, z_range))
    p10 = n / (hole_length + 1e-12) if hole_length > 0 else 0.0

    # P32 估计
    if spacing is not None:
        r = estimate_p32_from_spacing(net, domain)
        p32 = r.get("p32", 0.0)
        method = "spacing→P32"
    elif trace_lengths is not None:
        r_b = fit_beta_from_tracelength(trace_lengths)
        beta = r_b.get("beta", 3.5)
        # 用 β 拟合结果做 P32 区间估计
        from .dfn import estimate_p32_interval
        r_p = estimate_p32_interval(net, domain)
        p32 = r_p.get("p32_p50", 0.0)
        method = f"trace→β={beta:.2f}→P32"
    else:
        from .dfn import estimate_p32_interval
        r_p = estimate_p32_interval(net, domain)
        p32 = r_p.get("p32_p50", 0.0)
        method = "count→P32_interval"

    return {
        "p10_per_m": round(p10, 4),
        "p32_per_m2": round(p32, 4),
        "hole_length_m": round(hole_length, 2),
        "n_fractures": n,
        "method": method,
        "assumptions": [
            "P10 = 条数 / 井段长度 (编录表直接口径)",
            "P32 估计方法: " + method,
            "未考虑迹长/尺寸效应时, 默认 β=3.5",
        ],
    }
