# -*- coding: utf-8 -*-
"""露头裂隙网络 v1: 点云 → 面片集合 + 拓扑 + 密度/间距统计.

输入 : 带 seg_id 的点云 (RANSAC 段号, 由 label_free_dirs 输出; 可选 set_id 组标记).
输出 : OutcropNetwork 对象, 含:
  - surfaces: 每段的等效圆盘 (中心 / 法向 / 面积 / 等效半径 / 内点数 / 面内残差)
  - set_table: K 组模态方向 + 各组总面积点数
  - topology: 组间法向角矩阵 + 交线方位
  - spacing: 组内面片质心沿裂面走向投影距离的分布 (组内面 >= 2 时有效)
  - p32: 面总面积 / 包围盒体积

诚实边界:
  - "面片" = RANSAC 统一段, 非真实单条裂隙. 真实露头同组内可能由多条近平行裂隙合并.
  - 等效半径由内点凸包投影估算, 不可直接当迹长.
  - 点云覆盖范围外的延展不可知.

入口:
  build_outcrop_network(pos, seg_id, set_id=None, domain=None) -> OutcropNetwork

验收 (四期任务书 §T31):
  - 合成基准面数召回 >= 90%
  - 端到端出《数字露头裂隙网络报告》(HTML + JSON)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .inference import _unit
from .geometry import dip_dir_to_strike_vector


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class FracSurface:
    """单个 RANSAC 裂隙面 (等效圆盘)."""
    center: np.ndarray          # (3,) 内点质心
    normal: np.ndarray          # (3,) 单位法向 (符号与组均值一致)
    area: float                 # 等效面积 (m²), = π r_eq²
    r_eq: float                 # 等效半径 (m)
    n_points: int               # 内点数
    residual_rms: float         # 内点残差 (点到平面距离 RMS, m)
    set_id: int = -1            # 组标记 (有 set_id 时)
    seg_id: int = -1            # 原始 seg_id


@dataclass
class SetSummary:
    """一组裂面的聚合统计."""
    set_id: int
    n_surfaces: int
    normal: np.ndarray          # (3,) 组内圆均值 (Frechet)
    total_area: float           # 各面面积和
    total_points: int
    dip_deg: float = 0.0
    dip_dir_deg: float = 0.0


@dataclass
class OutcropNetwork:
    """露头裂隙网络."""
    surfaces: List[FracSurface]
    set_summaries: List[SetSummary]
    set_ids_unique: np.ndarray                  # (K,) 升序
    pairwise_angle_deg: np.ndarray              # (K, K) 组间法向角
    p32: float = 0.0
    domain: Optional[Tuple[float, float, float]] = None
    spacing_stats: dict = field(default_factory=dict)
    input_meta: dict = field(default_factory=dict)

    @property
    def n_surfaces(self) -> int:
        return len(self.surfaces)

    @property
    def n_sets(self) -> int:
        return len(self.set_summaries)


# ---------------------------------------------------------------------------
# 面片提取
# ---------------------------------------------------------------------------
def extract_frac_surfaces(pos: np.ndarray, seg_id: np.ndarray,
                          set_id: Optional[np.ndarray] = None,
                          v_min: float = 1e-6) -> List[FracSurface]:
    """从带 seg_id 的点云提 FracSurface 列表.

    参数:
      pos:   (N,3) 坐标
      seg_id:(N,)  RANSAC 段号 (-1 = 未归段 / fallback, 跳过)
      set_id:(N,) 可选组标记, 传入时赋值 .set_id
      v_min: 最小包围盒体积 (m³), 防退化

    返回:
      FracSurface 列表 (按 seg_id 升序)
    """
    pos = np.asarray(pos, dtype=np.float64)
    seg_id = np.asarray(seg_id, dtype=np.int64)
    if set_id is not None:
        set_id = np.asarray(set_id, dtype=np.int64)

    seg_unique = np.unique(seg_id[seg_id >= 0])
    out: List[FracSurface] = []
    for s in seg_unique:
        m = seg_id == s
        pts = pos[m]
        n_pts = int(m.sum())
        if n_pts < 3:
            continue
        c = pts.mean(0)
        X = pts - c
        # SVD 法向 (最小主轴)
        try:
            S = X.T @ X
            w, v = np.linalg.eigh(S)
            normal = v[:, 0]
            if normal[2] < 0:
                normal = -normal
            normal = normal / (np.linalg.norm(normal) + 1e-15)
            # 残差 RMS
            d = X @ normal
            rms = float(np.sqrt(np.mean(d * d)))
        except Exception:
            continue

        # 等效半径: 投影到平面, 离心 → 凸包面积近似
        # 构造平面内正交基
        if abs(normal[2]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(normal, ref); e1 /= (np.linalg.norm(e1) + 1e-15)
        e2 = np.cross(normal, e1); e2 /= (np.linalg.norm(e2) + 1e-15)
        u = X @ e1  # 面内坐标
        v = X @ e2
        # 面积 = 2 * 2D 凸包面积的近似: 用 2*std(u)*2*std(v)*π/ (椭圆面积)
        # 更简洁: 用最大距离近似 r_eq = sqrt(area/pi)
        # 保守估计: 用点云 2D 分布范围
        rng_u = u.max() - u.min()
        rng_v = v.max() - v.min()
        area_approx = np.pi * (rng_u * rng_v) / 4  # 椭圆面积近似
        r_eq = float(np.sqrt(max(area_approx / np.pi, 1e-6)))

        sid = -1
        if set_id is not None:
            sids_m = set_id[m]
            sids_valid = sids_m[sids_m >= 0]   # -1=未标注点 (API 契约允许), bincount 不收负值
            if len(sids_valid) > 0:
                sid = int(np.bincount(sids_valid.astype(int)).argmax())

        out.append(FracSurface(
            center=c.copy(), normal=normal.copy(),
            area=float(area_approx), r_eq=r_eq,
            n_points=n_pts, residual_rms=float(rms),
            set_id=sid, seg_id=int(s),
        ))
    return out


# ---------------------------------------------------------------------------
# 组间拓扑
# ---------------------------------------------------------------------------
def _compute_set_summaries(surfaces: List[FracSurface],
                           set_ids_unique: np.ndarray) -> List[SetSummary]:
    """聚合组统计."""
    sums: List[SetSummary] = []
    for k in set_ids_unique:
        surfs = [s for s in surfaces if s.set_id == k]
        if not surfs:
            continue
        # 面加权圆均值
        normals = np.array([s.normal for s in surfs])
        areas = np.array([s.area for s in surfs])
        areas_p = areas / max(areas.sum(), 1e-15)
        # Frechet 均值初始化 (加权算术 + 迭代)
        mu = (normals.T @ areas_p)
        if np.linalg.norm(mu) < 1e-8:
            # ±n 抵消时回退最大面法向, 避免噪声方向锁死迭代极点
            mu = normals[int(np.argmax(areas))]
        mu = mu / (np.linalg.norm(mu) + 1e-15)
        for _ in range(8):
            dots = np.clip(normals @ mu, -1, 1)
            signs = np.sign(dots)
            adj = normals * signs[:, None]
            mu2 = (adj.T @ areas_p)
            mu2_norm = np.linalg.norm(mu2)
            if mu2_norm < 1e-12:
                break
            mu = mu2 / mu2_norm
        if mu[2] < 0:
            mu = -mu

        total_area = float(sum(s.area for s in surfs))
        total_pts = sum(s.n_points for s in surfs)
        dip = float(np.degrees(np.arccos(np.clip(abs(mu[2]), 0, 1))))
        dip_dir = float(np.degrees(np.arctan2(mu[0], mu[1])) % 360)
        sums.append(SetSummary(
            set_id=int(k), n_surfaces=len(surfs), normal=mu,
            total_area=total_area, total_points=total_pts,
            dip_deg=dip, dip_dir_deg=dip_dir,
        ))
    return sums


def _compute_pairwise_angle(summaries: List[SetSummary]) -> Tuple[np.ndarray, np.ndarray]:
    """组间法向角矩阵."""
    K = len(summaries)
    M = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            d = abs(np.clip(np.dot(summaries[i].normal, summaries[j].normal), -1, 1))
            ang = float(np.degrees(np.arccos(d)))
            M[i, j] = M[j, i] = ang
    return M, np.array([s.set_id for s in summaries])


def _compute_spacing_stats(surfaces: List[FracSurface],
                           set_ids_unique: np.ndarray,
                           pos: np.ndarray, seg_id: np.ndarray,
                           decile: int = 1000) -> dict:
    """组内面片间距统计.

    组内 >= 2 面时才计算. 方法: 每面质心沿组平均走向 (水平投影方向) 投影,
    取排序后相邻差 = 间距. 返回每组的 {median, mean, p10, p90, n}.
    """
    stats = {}
    # 构造组法向映射
    set_normals = {}
    for k in set_ids_unique:
        surfs = [s for s in surfaces if s.set_id == k]
        if len(surfs) >= 2:
            # 组均值法向: 必须符号对齐 (项目铁律) —— 直接算术均值在组内 ±n
            # 混合时互相抵消, 归一化后是噪声方向, 走向投影随之全错.
            nn = np.array([s.normal for s in surfs])
            mu = nn.mean(0)
            if np.linalg.norm(mu) < 1e-8:
                mu = nn[0]
            mu = mu / (np.linalg.norm(mu) + 1e-15)
            for _ in range(8):
                dots = np.clip(nn @ mu, -1, 1)
                signs = np.sign(dots)
                signs[signs == 0] = 1
                mu2 = (nn * signs[:, None]).mean(0)
                n2 = np.linalg.norm(mu2)
                if n2 < 1e-12:
                    break
                mu = mu2 / n2
            if mu[2] < 0:
                mu = -mu
            set_normals[k] = mu
            # 走向 = 水平方向 ⊥ 倾向 (走向是 dip_dir - 90)
            # BUG-A 修复: 走向单位向量必须用集中函数 [sin,cos,0] (B3-4 集中化)
            dip_dir = float(np.degrees(np.arctan2(mu[0], mu[1])))
            strike_dir = dip_dir_to_strike_vector(dip_dir)
            # 各质心投影
            ctrs = np.array([s.center for s in surfs])
            proj = ctrs @ strike_dir
            if len(proj) < 2:
                continue
            proj_sorted = np.sort(proj)
            gaps = np.diff(proj_sorted)
            gaps = gaps[gaps > 1e-3]  # 过滤数值噪声
            if len(gaps) == 0:
                continue
            stats[str(k)] = {
                "n_surfaces": len(surfs),
                "n_gaps": int(len(gaps)),
                "mean_m": round(float(gaps.mean()), 3),
                "median_m": round(float(np.median(gaps)), 3),
                "p10_m": round(float(np.percentile(gaps, 10)), 3),
                "p90_m": round(float(np.percentile(gaps, 90)), 3),
            }
    stats["_method"] = "沿裂面走向投影质心距, 仅组内 >= 2 面"
    return stats


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def build_outcrop_network(pos: np.ndarray, seg_id: np.ndarray,
                          set_id: Optional[np.ndarray] = None,
                          domain: Optional[Tuple[float, float, float]] = None,
                          min_surface_points: int = 10) -> OutcropNetwork:
    """从带 seg_id 的点云构造 OutcropNetwork."""
    pos = np.asarray(pos, dtype=np.float64)
    seg_id = np.asarray(seg_id, dtype=np.int64)

    surfs = extract_frac_surfaces(pos, seg_id, set_id)

    # 去掉极小面
    surfs = [s for s in surfs if s.n_points >= min_surface_points]

    if set_id is not None:
        set_id_arr = np.asarray(set_id, dtype=np.int64)
        set_ids_unique = np.unique(set_id_arr[set_id_arr >= 0])
    else:
        set_ids_unique = np.array([-1])

    set_sums = _compute_set_summaries(surfs, set_ids_unique)
    pw_ang, set_id_order = _compute_pairwise_angle(set_sums)
    spacing = _compute_spacing_stats(surfs, set_ids_unique, pos, seg_id)

    # P32: 总面面积 / 包围盒体积
    total_area = sum(s.area for s in surfs)
    if domain is not None:
        vol = float(domain[0] * domain[1] * domain[2])
    else:
        # pos 包围盒
        lo = pos.min(0); hi = pos.max(0)
        vol = max(float(np.prod(hi - lo)), 1e-9)
    p32 = total_area / vol

    net = OutcropNetwork(
        surfaces=surfs,
        set_summaries=set_sums,
        set_ids_unique=set_id_order,
        pairwise_angle_deg=pw_ang,
        p32=p32,
        domain=domain,
        spacing_stats=spacing,
        input_meta={
            "n_points": int(pos.shape[0]),
            "n_RANSAC_segments_raw": int(np.unique(seg_id[seg_id >= 0]).shape[0]),
            "n_surfaces_filtered": len(surfs),
            "vol_m3": round(vol, 3),
        },
    )
    return net


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------
def to_dict(net: OutcropNetwork) -> dict:
    """OutcropNetwork → 可 JSON 序列化 dict."""
    return {
        "input_meta": net.input_meta,
        "p32_m2_per_m3": round(net.p32, 5),
        "domain_m": list(net.domain) if net.domain else None,
        "n_surfaces": net.n_surfaces,
        "n_sets": net.n_sets,
        "surfaces": [
            {
                "seg_id": s.seg_id,
                "set_id": s.set_id,
                "center_m": [round(float(x), 4) for x in s.center],
                "normal": [round(float(x), 6) for x in s.normal],
                "area_m2": round(s.area, 3),
                "r_eq_m": round(s.r_eq, 4),
                "n_points": s.n_points,
                "residual_rms_m": round(s.residual_rms, 5),
            } for s in net.surfaces
        ],
        "set_summaries": [
            {
                "set_id": s.set_id,
                "n_surfaces": s.n_surfaces,
                "dip_deg": round(s.dip_deg, 2),
                "dip_dir_deg": round(s.dip_dir_deg, 2),
                "normal": [round(float(x), 6) for x in s.normal],
                "total_area_m2": round(s.total_area, 3),
                "total_points": s.total_points,
            } for s in net.set_summaries
        ],
        "pairwise_angle_deg": [
            [round(float(net.pairwise_angle_deg[i, j]), 2)
             for j in range(net.pairwise_angle_deg.shape[1])]
            for i in range(net.pairwise_angle_deg.shape[0])
        ],
        "set_ids": [int(x) for x in net.set_ids_unique],
        "spacing_stats": net.spacing_stats,
        "_honest_boundary": [
            "面片 = RANSAC 统一段, 非真实单裂隙. 同组内多条近平行裂隙可能被 RANSAC 合并.",
            "等效半径由内点凸包投影估算, 不可直接当迹长 / 开度.",
            "P32 仅基于可见面片, 点云覆盖范围外延展不可知.",
            "组内间距仅当组内 >= 2 面时计算, 单面组无间距.",
        ],
    }
