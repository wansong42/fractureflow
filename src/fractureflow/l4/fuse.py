# -*- coding: utf-8 -*-
"""T44: 冲突门控融合引擎 —— 多源组系审计 + 加权 Fréchet 融合 + 置信锥.

流程:
  1. 各源观测-only 球面 k-means 出各自组系 (centers, assign, κ)
  2. 源间组配对用匈牙利算法 (代价 = acos|<ci, cj>|)
  3. 按 T34 决策规则分档 (<20° 一致 / 20–35° 部分 / >=35° 冲突):
     - 一致 → 加权 Fréchet 均值合并 (w = √n_i × 源质量先验)
     - 部分 → 仅合并最一致对
     - 冲突 → 保持分立并标注
  4. 合并观测重估浓度 κ + bootstrap 95% 置信锥

输出: UnifiedSetTable = 方向 + κ + 95% 置信锥半角 + 贡献源 + 冲突标记.

诚实边界:
  - 同源池化 (多井联合) 在诚实口径下已证无增益 (P0 审计). 本模块只融合**异源**.
  - 融合增益仅在跨源互补时成立; 合成孪生验收见 scripts/fusion_synthetic_twin.py
  - 注意: `_merge_sources` 按 `source_type` 查找 source_table, 同类型多井需先聚合 (如 fusion_pipeline.py:_aggregate_entries).
  - L2 走向约束 (strike_constraint): 虚拟法向仅用于冲突审计匹配, 不参与均值融合 (entry.normals=None → _merge_two_groups 返回 None).

用法:
    from fractureflow.l4.fuse import fuse_bundle, unified_set_table_to_df
    result = fuse_bundle(bundle, K=4)
    # result.unified_set_table = UnifiedSetTable
    # result.conflict_report = {...}
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..setlabel import spherical_kmeans
from ..dfn import SetTable
from .constants import (
    ANGLE_CONSISTENT, ANGLE_CONFLICT, SOURCE_QUALITY_INITIAL,
    N_BOOTSTRAP, CONFIDENCE_LEVEL, EPS, STRUCTURE_TYPE_UNKNOWN,
)
from .evidence import EvidenceBundle, EvidenceEntry


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class SourceSetTable:
    """单源组系解析结果."""
    entry: EvidenceEntry                   # 来源证据条目
    centers: np.ndarray                    # (K, 3) 组模态方向
    assign: np.ndarray                     # (N,) 逐点组指派
    concentrations: np.ndarray             # (K,) 组内 vMF 浓度 κ
    group_n: np.ndarray                    # (K,) 每组观测点数
    K: int

    @property
    def source_type(self) -> str:
        return self.entry.source_type

    @property
    def source_name(self) -> str:
        return self.entry.source_name


@dataclass
class ConflictMatch:
    """两源组匹配对."""
    source_a: str                          # "L0 / L1 / L2 / L3"
    source_b: str
    group_a: int                           # 源 A 的组号
    group_b: int                           # 源 B 的组号
    angle_deg: float                       # 组心角距 (度)
    decision: str                          # "consistent" / "partial" / "conflict"
    # 匹配所属的具体 SourceSetTable 在 _merge_sources 收到的列表中的位置。
    # BUG-5 修复 (2026-08): 匹配在具体 entry i,j 的 centers 上算出, 合并必须用
    # 同一个 entry; 只按 source_type 查表时, 同类型多井 (如两口 L0/L1 井) 会把
    # A 表的组号套到 B 表的 assign 上 (张冠李戴)。idx 缺失时才回退类型查找。
    idx_a: Optional[int] = None
    idx_b: Optional[int] = None


@dataclass
class MergedGroup:
    """合并后的融合组."""
    group_id: int
    normal: np.ndarray                     # (3,) 融合方向 (Fréchet 均值)
    concentration: float                   # κ (合并观测重估)
    cone_95_deg: float                     # 95% 置信锥半角
    source_contributions: List[str]        # 贡献源列表
    # T50: 贡献结构面类型集合 (同向不同类型组跨子空间合并时保留各类型标签)
    structure_types: List[str] = field(default_factory=list)
    n_total: int = 0                       # 合并后总点数
    n_per_source: Dict[str, int] = field(default_factory=dict)


@dataclass
class UnifiedSetTable:
    """统一组系表 —— 融合输出."""
    groups: List[MergedGroup]
    # 与 dfn.SetTable 兼容的接口
    centers: np.ndarray = None             # (K_fused, 3)
    concentrations: np.ndarray = None      # (K_fused,)
    proportions: np.ndarray = None         # (K_fused,)
    # L4 扩展字段
    cones_95: np.ndarray = None            # (K_fused,) 置信锥半角
    source_map: List[List[str]] = None     # 各组的贡献源
    conflict_flags: List[str] = None       # 各组冲突标记
    structure_types: List[List[str]] = None  # (T50) 各组贡献结构面类型

    def __post_init__(self):
        if self.centers is None and self.groups:
            self.centers = np.stack([g.normal for g in self.groups])
            self.concentrations = np.array([g.concentration for g in self.groups])
            total_n = sum(g.n_total for g in self.groups)
            self.proportions = np.array([g.n_total / max(total_n, 1) for g in self.groups])
            self.cones_95 = np.array([g.cone_95_deg for g in self.groups])
            self.source_map = [g.source_contributions for g in self.groups]
            self.conflict_flags = [""] * len(self.groups)
            # T50: structure_types per group
            self.structure_types = [g.structure_types for g in self.groups]

    @property
    def K(self) -> int:
        return len(self.groups)

    def to_settable(self) -> SetTable:
        """转换为 dfn.SetTable (向后兼容 DFN 生成)."""
        return SetTable(
            centers=self.centers,
            concentrations=self.concentrations,
            proportions=self.proportions,
        )


@dataclass
class FuseResult:
    """融合结果 (统一组系表 + 冲突报告)."""
    unified_set_table: UnifiedSetTable
    source_tables: List[SourceSetTable]
    conflict_matches: List[ConflictMatch]
    conflicts_report: dict                  # 可落盘的审计 JSON
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------

def _unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + EPS)


def weighted_frechet_mean(normals: np.ndarray,
                          weights: Optional[np.ndarray] = None,
                          max_iter: int = 100,
                          tol: float = 1e-6) -> np.ndarray:
    """加权 Fréchet (intrinsic) 均值 on S², 含符号对齐.

    normals: (N, 3) 单位向量 (无向法向)
    weights: (N,) 非负权重, 默认等权

    算法: Weiszfeld-like iteration on sphere.
    """
    pts = _unit(np.asarray(normals, dtype=np.float64))
    N = pts.shape[0]
    if N == 0:
        return np.array([0.0, 0.0, 1.0])
    if weights is None:
        weights = np.ones(N, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    weights = weights / (weights.sum() + EPS)

    # 初始化: 符号对齐后的加权均值
    ref = pts[0]
    sgn = np.sign((pts * ref).sum(-1, keepdims=True))
    sgn[sgn == 0] = 1
    aligned = pts * sgn
    mu = _unit((aligned * weights[:, None]).sum(0))

    for _ in range(max_iter):
        # 投影到切空间: 对每个点, 计算 log_map at mu
        # log_mu(p) = (theta / sin(theta)) * (p - cos(theta)*mu)
        # where theta = arccos(<p, mu>), cos(theta) = |<p, mu>| due to undirected
        # 先符号对齐到 mu
        dots = (pts * mu).sum(-1, keepdims=True)  # (N, 1)
        sgn = np.sign(dots)
        sgn[sgn == 0] = 1
        aligned = pts * sgn

        cos_th = np.clip((aligned * mu).sum(-1), 0, 1)  # (N,) — 符号对齐后应 ∈[0,1], clip 防数值误差
        theta = np.arccos(cos_th)
        # 切向量
        sin_th = np.sin(theta)
        # 避免 theta=0 除零
        safe = sin_th > 1e-8
        log_vecs = np.zeros_like(aligned)
        log_vecs[safe] = (theta[safe, None] / sin_th[safe, None]) * (
            aligned[safe] - cos_th[safe, None] * mu
        )
        # 加权切均值
        weighted_tangent = (log_vecs * weights[:, None]).sum(0)
        tangent_norm = np.linalg.norm(weighted_tangent)

        if tangent_norm < tol:
            break
        # 指数映射回球面
        mu = _unit(np.cos(tangent_norm) * mu + np.sin(tangent_norm) * _unit(weighted_tangent))

    return mu.flatten()


def estimate_kappa(normals: np.ndarray, center: np.ndarray) -> float:
    """从观测法向 + 组心估计 vMF 浓度 κ (无向 |cos| 口径).

    简化 estimator: κ ≈ (R * (d - R²)) / (1 - R²), d=3, R = |mean resultant length|.
    对无向数据: R = |<normals_refaligned, center>| 的均值.
    """
    pts = _unit(np.asarray(normals, dtype=np.float64))
    center = _unit(np.asarray(center, dtype=np.float64)).flatten()
    # 符号对齐
    dots = pts @ center
    sgn = np.sign(dots)
    sgn[sgn == 0] = 1
    aligned = pts * sgn[:, None]
    # 沿 center 方向的平均投影
    R = np.mean(aligned @ center)
    R = np.clip(R, 0, 1)
    if R > 0.999:
        return 100.0
    if R < 0.01:
        return 0.01
    d = 3.0
    kappa = (R * (d - R ** 2)) / (1 - R ** 2)
    return float(np.clip(kappa, 0.01, 200.0))


def bootstrap_confidence_cone(normals: np.ndarray,
                              weights: Optional[np.ndarray] = None,
                              n_bootstrap: int = N_BOOTSTRAP,
                              ci_level: float = CONFIDENCE_LEVEL,
                              seed: int = 42) -> float:
    """Bootstrap 置信锥半角 (度).

    重采样观测条目 (保持组内相关结构), 每次算加权 Fréchet 均值,
    返回覆盖 ci_level bootstrap 均值的锥半角.
    """
    pts = _unit(np.asarray(normals, dtype=np.float64))
    N = pts.shape[0]
    if N < 3:
        return 180.0  # 不确定度极大

    rng = np.random.default_rng(seed)
    # 中心 (参考)
    center = weighted_frechet_mean(pts, weights)

    # bootstrap
    angles = np.empty(n_bootstrap)
    for bi in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        sample = pts[idx]
        if weights is not None:
            w = np.asarray(weights)[idx]
        else:
            w = None
        mu_b = weighted_frechet_mean(sample, w)
        # 角距 (含符号对齐)
        cos_val = abs(float(np.clip(mu_b @ center, -1, 1)))
        angles[bi] = np.degrees(np.arccos(np.clip(cos_val, 0, 1)))

    # 锥半角: 覆盖 ci_level 的角度分位数
    cone = float(np.percentile(angles, ci_level * 100))
    return cone


# ---------------------------------------------------------------------------
# 组间匹配
# ---------------------------------------------------------------------------

def hungarian_match_groups(centers_a: np.ndarray, centers_b: np.ndarray,
                           cost_threshold: float = 90.0) -> List[Tuple[int, int, float]]:
    """匈牙利匹配两组方向, 代价 = acos|<ci, cj>| (度).

    返回 [(i, j, angle_deg), ...]. 裁剪 cost > threshold 的匹配 (极差匹配).
    """
    Ka = centers_a.shape[0]
    Kb = centers_b.shape[0]
    if Ka == 0 or Kb == 0:
        return []
    cost = np.zeros((Ka, Kb))
    for i in range(Ka):
        for j in range(Kb):
            cos_val = abs(float(np.clip(centers_a[i] @ centers_b[j], -1, 1)))
            cost[i, j] = np.degrees(np.arccos(np.clip(cos_val, 0, 1)))
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(row_ind, col_ind):
        angle = cost[i, j]
        if angle < cost_threshold:
            matches.append((int(i), int(j), float(angle)))
    return matches


def classify_consistency(angle_deg: float) -> str:
    """根据角距判定一致性."""
    if angle_deg < ANGLE_CONSISTENT:
        return "consistent"
    elif angle_deg < ANGLE_CONFLICT:
        return "partial"
    else:
        return "conflict"


# ---------------------------------------------------------------------------
# 源内组系解析
# ---------------------------------------------------------------------------

def parse_source_table(entry: EvidenceEntry, K: int, seed: int = 42) -> Optional[SourceSetTable]:
    """对单条证据跑观测-only 球面 k-means, 输出 SourceSetTable."""
    if entry.normals is None or entry.is_constraint_only:
        # L2 约束条目: 单独处理, 不用 k-means
        if entry.is_constraint_only and entry.strikes is not None:
            return _parse_strike_constraint(entry, K, seed)
        return None

    nrm = entry.normals
    N = nrm.shape[0]
    if N < K * 2:  # 点太少, 退化为较少组
        K_eff = max(1, N // 2)
    else:
        K_eff = K

    centers, assign = spherical_kmeans(nrm, K_eff, seed=seed)
    # 每组的 κ 与点数
    concentrations = np.zeros(K_eff)
    group_n = np.zeros(K_eff, dtype=int)
    for k in range(K_eff):
        mask = assign == k
        group_n[k] = int(mask.sum())
        if group_n[k] >= 3:
            concentrations[k] = estimate_kappa(nrm[mask], centers[k])
        else:
            concentrations[k] = SOURCE_QUALITY_INITIAL  # 默认低浓度

    return SourceSetTable(
        entry=entry,
        centers=centers,
        assign=assign,
        concentrations=concentrations,
        group_n=group_n,
        K=K_eff,
    )


def _parse_strike_constraint(entry: EvidenceEntry, K: int, seed: int) -> SourceSetTable:
    """L2 走向约束 → 虚拟组系表 (用于匹配验证, 不参与均值融合)."""
    strikes = entry.strikes  # (M,) 走向方位
    # 走向聚类 (一维 circular k-means on 0-180)
    # 简化: 用 bin 分组
    nrm_count = len(strikes)
    centers = np.zeros((0, 3))
    assign = np.zeros(nrm_count, dtype=int)

    if nrm_count > 0:
        # 用走向近似法向 (假设垂直倾角)
        from .evidence import dip_dipdir_to_normal
        fake_dip = np.full(nrm_count, 90.0)  # 垂直
        fake_dd = strikes % 360  # 走向+90 ≈ dip direction (近似)
        fake_dd = (fake_dd + 90) % 360
        normals = dip_dipdir_to_normal(fake_dip, fake_dd)
        K_eff = min(K, max(1, nrm_count // 10))
        if K_eff >= 2:
            centers, assign = spherical_kmeans(normals, K_eff, seed=seed)
        else:
            centers = weighted_frechet_mean(normals).reshape(1, 3)
            assign = np.zeros(nrm_count, dtype=int)
            K_eff = 1

    return SourceSetTable(
        entry=entry,
        centers=centers,
        assign=assign,
        concentrations=np.ones(max(1, len(centers))) * 0.1,  # 约束源低浓度
        group_n=np.bincount(assign, minlength=max(1, len(centers))),
        K=max(1, len(centers)),
    )


# ---------------------------------------------------------------------------
# 融合主流程
# ---------------------------------------------------------------------------

def fuse_bundle(bundle: EvidenceBundle, K: int, seed: int = 42,
                type_aware: bool = False) -> FuseResult:
    """融合 EvidenceBundle → UnifiedSetTable.

    主流程:
      1. 各源解析为 SourceSetTable
      2. 跨源 Hungarian 匹配
      3. 一致性门控 + 加权 Fréchet 合并
      4. 冲突保持分立

    T50: type_aware=True 时, 先按 (source_type, structure_type) 划分子空间,
    子空间内做既有 k-means + 匈牙利匹配 + 三线决策, 再跨子空间合并同向组
    (合并时类型不同但方向一致的组保留各自 type 标签).
    structure_type=unknown 的条目**混入各子空间** (不划独立空间), 保证向后兼容.
    """
    # Step 1: 源内解析
    source_tables: List[SourceSetTable] = []
    for entry in bundle.entries:
        st = parse_source_table(entry, K, seed=seed)
        if st is not None and st.K > 0:
            source_tables.append(st)

    if not source_tables:
        # 空 bundle → 返回空调表
        return FuseResult(
            unified_set_table=UnifiedSetTable(groups=[]),
            source_tables=[],
            conflict_matches=[],
            conflicts_report={"status": "empty_bundle", "site": bundle.site_name},
        )

    if not type_aware:
        # 六期标准模式 (无类型隔离)
        all_matches: List[ConflictMatch] = []
        for i in range(len(source_tables)):
            for j in range(i + 1, len(source_tables)):
                sa = source_tables[i]
                sb = source_tables[j]
                if sa.source_type == sb.source_type:
                    continue
                matches = hungarian_match_groups(sa.centers, sb.centers)
                for (gi, gj, angle) in matches:
                    cm = ConflictMatch(
                        source_a=sa.source_type,
                        source_b=sb.source_type,
                        group_a=gi,
                        group_b=gj,
                        angle_deg=angle,
                        decision=classify_consistency(angle),
                        idx_a=i, idx_b=j,
                    )
                    all_matches.append(cm)
        merged_groups, conflict_report = _merge_sources(source_tables, all_matches, seed)
        # T61 修复: 标准分支此前把匹配明细丢在局部变量里, fuse_bundle 的
        # conflict_matches 恒为空, T44 Gate3 (冲突植入检出) 必挂.
        matches_for_result = all_matches
    else:
        # T50: 类型隔离模式
        merged_groups, conflict_report = _fuse_type_aware(source_tables, K, seed)
        matches_for_result = conflict_report.get("all_matches", [])

    # Step 4: 构造 UnifiedSetTable
    ust = UnifiedSetTable(groups=merged_groups)

    return FuseResult(
        unified_set_table=ust,
        source_tables=source_tables,
        conflict_matches=matches_for_result,
        conflicts_report=conflict_report,
        meta={
            "site_name": bundle.site_name,
            "K_input": K,
            "type_aware": type_aware,
            "n_source_tables": len(source_tables),
            "n_consistent_pairs": conflict_report.get("n_consistent_merged", 0),
            "n_partial_pairs": conflict_report.get("n_partial_merged", 0),
            "n_conflict_pairs": conflict_report.get("n_conflict_kept", 0),
        },
    )


def _resolve_table(source_tables: List[SourceSetTable],
                   source_type: str, idx: Optional[int]) -> Optional[SourceSetTable]:
    """解析匹配对所属的具体 SourceSetTable.

    优先用匹配时记录的条目索引 (idx); 索引缺失或类型不符才回退 source_type
    查找, 且回退路径遇到同类型多表时必须显式报错 (否则会静默张冠李戴)。
    """
    if idx is not None and 0 <= idx < len(source_tables):
        st = source_tables[idx]
        if st.source_type == source_type:
            return st
    same_type = [s for s in source_tables if s.source_type == source_type]
    if len(same_type) == 1:
        return same_type[0]
    if len(same_type) > 1:
        raise ValueError(
            f"_merge_sources: 存在 {len(same_type)} 个同类型 ({source_type}) 源表 "
            f"且 ConflictMatch 未携带条目索引 (idx_a/idx_b), 无法确定合并对象. "
            f"请先聚合同类型多井 (如 fusion_pipeline 的预聚合), 或使用携带 idx 的"
            f"fuse_bundle 主流程.")
    return None


def _merge_sources(source_tables: List[SourceSetTable],
                   matches: List[ConflictMatch],
                   seed: int = 42) -> Tuple[List[MergedGroup], dict]:
    """核心合并逻辑: consistent 合并, partial 仅最一致对, conflict 保持."""
    consistent_sets = [cm for cm in matches if cm.decision == "consistent"]
    partial_pairs = [cm for cm in matches if cm.decision == "partial"]
    conflict_pairs = [cm for cm in matches if cm.decision == "conflict"]

    merged: List[MergedGroup] = []
    gid = 0
    used_entries: set = set()
    n_consistent_merged = 0
    n_partial_merged = 0

    for cm in consistent_sets:
        sa = _resolve_table(source_tables, cm.source_a, cm.idx_a)
        sb = _resolve_table(source_tables, cm.source_b, cm.idx_b)
        if sa is None or sb is None:
            continue
        # L2 约束源 (normals=None) 不参与融合, 仅作审计
        if sa.entry.normals is None or sb.entry.normals is None:
            continue
        key_a = (sa.source_name, cm.group_a)
        key_b = (sb.source_name, cm.group_b)
        if key_a in used_entries or key_b in used_entries:
            continue
        mg = _merge_two_groups(sa, cm.group_a, sb, cm.group_b, gid, seed)
        if mg is not None:
            merged.append(mg)
            used_entries.add(key_a)
            used_entries.add(key_b)
            gid += 1
            n_consistent_merged += 1

    if partial_pairs:
        partial_pairs.sort(key=lambda cm: cm.angle_deg)
        best = partial_pairs[0]
        sa = _resolve_table(source_tables, best.source_a, best.idx_a)
        sb = _resolve_table(source_tables, best.source_b, best.idx_b)
        if sa is not None and sb is not None:
            if sa.entry.normals is not None and sb.entry.normals is not None:
                key_a = (sa.source_name, best.group_a)
                key_b = (sb.source_name, best.group_b)
                if key_a not in used_entries and key_b not in used_entries:
                    mg = _merge_two_groups(sa, best.group_a, sb, best.group_b, gid, seed)
                    if mg is not None:
                        merged.append(mg)
                        used_entries.add(key_a)
                        used_entries.add(key_b)
                        gid += 1
                        n_partial_merged += 1

    # conflict 对中的组保持独立输出
    for st in source_tables:
        for k in range(st.K):
            if (st.source_name, k) in used_entries:
                continue
            nrm_entry = st.entry.normals
            if nrm_entry is None:
                continue
            mask = st.assign == k
            if mask.sum() == 0:
                continue  # B5(T61): 空簇直接跳过, 不产出全点幻影组
            pts = nrm_entry[mask]
            w = np.ones(pts.shape[0]) * st.entry.weight
            mu = weighted_frechet_mean(pts, w)
            cone = bootstrap_confidence_cone(pts, w, seed=seed)
            kappa = estimate_kappa(pts, mu)
            # T50: propagate structure_type
            stype = st.entry.structure_type
            struct_types = [stype] if stype != STRUCTURE_TYPE_UNKNOWN else []
            mg = MergedGroup(
                group_id=gid,
                normal=mu,
                concentration=kappa,
                cone_95_deg=cone,
                source_contributions=[f"{st.source_type}:{st.source_name}"],
                structure_types=struct_types,
                n_total=int(st.group_n[k]),
                n_per_source={st.source_type: int(st.group_n[k])},
            )
            merged.append(mg)
            gid += 1

    # 冲突报告
    conflict_report = {
        "n_consistent_total": len(consistent_sets),
        "n_consistent_merged": n_consistent_merged,
        "n_partial_total": len(partial_pairs),
        "n_partial_merged": n_partial_merged,
        "n_conflict_kept": len(conflict_pairs),
        "conflicts": [
            {
                "source_a": cm.source_a, "source_b": cm.source_b,
                "angle": round(cm.angle_deg, 2), "decision": cm.decision,
            }
            for cm in conflict_pairs
        ],
        "n_output_groups": len(merged),
    }

    return merged, conflict_report


def _merge_two_groups(sa: SourceSetTable, ga: int,
                      sb: SourceSetTable, gb: int,
                      gid: int, seed: int) -> Optional[MergedGroup]:
    """合并两组 (来自不同源)."""
    # 提取观测
    nrm_a = sa.entry.normals
    nrm_b = sb.entry.normals
    if nrm_a is None or nrm_b is None:
        return None

    mask_a = sa.assign == ga
    mask_b = sb.assign == gb
    pts_a = nrm_a[mask_a]
    pts_b = nrm_b[mask_b]
    n_a = pts_a.shape[0]
    n_b = pts_b.shape[0]

    # 权重 = √n × 源质量先验
    w_a = np.sqrt(n_a) * sa.entry.weight
    w_b = np.sqrt(n_b) * sb.entry.weight

    all_pts = np.concatenate([pts_a, pts_b], axis=0)
    all_w = np.concatenate([np.full(n_a, w_a), np.full(n_b, w_b)])

    # 加权 Fréchet 均值
    mu = weighted_frechet_mean(all_pts, all_w)

    # 合并 κ
    kappa = estimate_kappa(all_pts, mu)

    # bootstrap 置信锥
    cone = bootstrap_confidence_cone(all_pts, all_w, seed=seed)

    # T50: 收集结构面类型 (去重)
    types_a = [sa.entry.structure_type] if sa.entry.structure_type != STRUCTURE_TYPE_UNKNOWN else []
    types_b = [sb.entry.structure_type] if sb.entry.structure_type != STRUCTURE_TYPE_UNKNOWN else []
    structure_types = sorted(set(types_a + types_b))

    return MergedGroup(
        group_id=gid,
        normal=mu,
        concentration=kappa,
        cone_95_deg=cone,
        source_contributions=[f"{sa.source_type}:{sa.source_name}", f"{sb.source_type}:{sb.source_name}"],
        structure_types=structure_types,
        n_total=n_a + n_b,
        n_per_source={sa.source_type: n_a, sb.source_type: n_b},
    )


# ---------------------------------------------------------------------------
# T50: 类型隔离融合
# ---------------------------------------------------------------------------

def _fuse_type_aware(source_tables: List[SourceSetTable], K: int,
                     seed: int) -> Tuple[List[MergedGroup], dict]:
    """T50 类型隔离融合: 先按 (source_type, structure_type) 划分子空间,
    子空间内做既有 k-means + 匈牙利匹配 + 三线决策,
    再跨子空间合并同向组 (匈牙利全局匹配, 保留各类型标签).

    关键纪律:
      - structure_type=unknown 的条目**混入各子空间** (不划独立空间)
      - 跨子空间合并必须走匈牙利全局匹配, 不贪心
    """
    # Step A: 划分子空间
    # 子空间 key = (source_type, structure_type) — 但 unknown 混入所有空间
    # 实现: 对每个 source_type, 收集其所有 entries 的 structure_type 集合
    #       若只有 unknown → 单空间; 若有已知类型 → 各已知类型一个空间 + unknown 混入全部

    # 按 source_type 分组
    by_source_type: Dict[str, List[SourceSetTable]] = {}
    for st in source_tables:
        by_source_type.setdefault(st.source_type, []).append(st)

    # 子空间: 每个 (source_type, structure_type) 对应一组 SourceSetTable
    subspaces: Dict[Tuple[str, str], List[SourceSetTable]] = {}
    for stype, st_list in by_source_type.items():
        # 收集该 source_type 下所有已知类型
        known_types = set()
        for st in st_list:
            if st.entry.structure_type != STRUCTURE_TYPE_UNKNOWN:
                known_types.add(st.entry.structure_type)

        if not known_types:
            # 全 unknown → 单空间
            subspaces[(stype, STRUCTURE_TYPE_UNKNOWN)] = st_list
        else:
            # 有已知类型 → 各已知类型一个空间
            for kt in known_types:
                subspaces[(stype, kt)] = []
            # unknown 条目混入各已知类型空间
            for st in st_list:
                if st.entry.structure_type == STRUCTURE_TYPE_UNKNOWN:
                    for kt in known_types:
                        subspaces[(stype, kt)].append(st)
                else:
                    subspaces[(stype, st.entry.structure_type)].append(st)

    # Step B: 子空间内融合 (既有逻辑)
    all_matches: List[ConflictMatch] = []
    n_consistent_merged = 0
    n_partial_merged = 0
    n_conflict_kept = 0

    # 记录每个子空间的合并结果 (用于跨子空间合并)
    # subspace_results[sub_key] = list of MergedGroup (按追加顺序, local_idx = 在 list 中的位置)
    subspace_results: Dict[Tuple[str, str], List[MergedGroup]] = {}

    for sub_key, sub_tables in subspaces.items():
        if not sub_tables:
            continue
        # 子空间内跨源匹配
        sub_matches: List[ConflictMatch] = []
        for i in range(len(sub_tables)):
            for j in range(i + 1, len(sub_tables)):
                sa = sub_tables[i]
                sb = sub_tables[j]
                if sa.source_type == sb.source_type:
                    continue
                matches = hungarian_match_groups(sa.centers, sb.centers)
                for (gi, gj, angle) in matches:
                    cm = ConflictMatch(
                        source_a=sa.source_type,
                        source_b=sb.source_type,
                        group_a=gi,
                        group_b=gj,
                        angle_deg=angle,
                        decision=classify_consistency(angle),
                        idx_a=i, idx_b=j,
                    )
                    sub_matches.append(cm)
        all_matches.extend(sub_matches)

        # 子空间内合并
        sub_merged, sub_report = _merge_sources(sub_tables, sub_matches, seed)
        # 暂存 (group_id 稍后统一重编)
        subspace_results[sub_key] = sub_merged
        n_consistent_merged += sub_report.get("n_consistent_merged", 0)
        n_partial_merged += sub_report.get("n_partial_merged", 0)
        n_conflict_kept += sub_report.get("n_conflict_kept", 0)

    # Step C: 跨子空间合并同向组 (匈牙利全局匹配, 不贪心)
    sub_keys = list(subspace_results.keys())
    used_groups: set = set()  # (sub_key, local_idx)
    cross_merged: List[MergedGroup] = []  # 跨子空间合并产生的新组

    for i in range(len(sub_keys)):
        for j in range(i + 1, len(sub_keys)):
            key_a = sub_keys[i]
            key_b = sub_keys[j]
            groups_a = subspace_results.get(key_a, [])
            groups_b = subspace_results.get(key_b, [])
            if not groups_a or not groups_b:
                continue

            centers_a = np.stack([g.normal for g in groups_a])
            centers_b = np.stack([g.normal for g in groups_b])

            # 匈牙利全局匹配
            matches = hungarian_match_groups(centers_a, centers_b)
            for (gi, gj, angle) in matches:
                if angle >= ANGLE_CONSISTENT:
                    continue  # 仅合并一致对 (<20°)
                idx_a = (key_a, gi)
                idx_b = (key_b, gj)
                if idx_a in used_groups or idx_b in used_groups:
                    continue
                ga = groups_a[gi]
                gb = groups_b[gj]
                # 跨子空间合并: 保留各类型标签
                merged_mg = _merge_cross_subspace_groups(ga, gb, -1)
                if merged_mg is not None:
                    cross_merged.append(merged_mg)
                    used_groups.add(idx_a)
                    used_groups.add(idx_b)
                    n_consistent_merged += 1

    # Step D: 组装最终输出
    # 未被子空间间合并的组 + 跨子空间合并产生的新组
    final_merged: List[MergedGroup] = []
    for sub_key, groups in subspace_results.items():
        for li, mg in enumerate(groups):
            if (sub_key, li) not in used_groups:
                final_merged.append(mg)
    # 追加跨子空间合并的组
    final_merged.extend(cross_merged)
    # 统一重编 group_id
    for i, mg in enumerate(final_merged):
        mg.group_id = i

    conflict_report = {
        "n_consistent_merged": n_consistent_merged,
        "n_partial_merged": n_partial_merged,
        "n_conflict_kept": n_conflict_kept,
        "n_output_groups": len(final_merged),
        "n_subspaces": len(subspaces),
        "n_cross_subspace_merged": len(cross_merged),
        "subspace_keys": [f"{k[0]}:{k[1]}" for k in subspaces.keys()],
        "all_matches": all_matches,
    }

    return final_merged, conflict_report


def _merge_cross_subspace_groups(ga: MergedGroup, gb: MergedGroup,
                                  gid: int) -> Optional[MergedGroup]:
    """跨子空间合并两个 MergedGroup (类型不同但方向一致).
    保留各类型标签, 方向取加权平均.
    """
    # 权重按 n_total
    na = ga.n_total
    nb = gb.n_total
    wa = np.sqrt(na)
    wb = np.sqrt(nb)

    # 加权方向平均 (符号对齐)
    a = ga.normal
    b = gb.normal
    # 符号对齐
    if a @ b < 0:
        b = -b
    mu = _unit(wa * a + wb * b)

    # 合并类型标签
    all_types = sorted(set(ga.structure_types + gb.structure_types))

    # n_per_source 按 key 累加: 跨子空间合并常发生在同 source_type 不同
    # structure_type 之间 (如 L1-natural × L1-induced), dict 直接合并会
    # 覆盖同键计数, 导致 sum(values) != n_total, 审计自相矛盾.
    nps = dict(ga.n_per_source)
    for k, v in gb.n_per_source.items():
        nps[k] = nps.get(k, 0) + int(v)

    return MergedGroup(
        group_id=gid,
        normal=mu,
        concentration=max(ga.concentration, gb.concentration),
        cone_95_deg=max(ga.cone_95_deg, gb.cone_95_deg),
        source_contributions=ga.source_contributions + gb.source_contributions,
        structure_types=all_types,
        n_total=na + nb,
        n_per_source=nps,
    )


# ---------------------------------------------------------------------------
# 序列化辅助
# ---------------------------------------------------------------------------

def unified_set_table_to_dict(ust: UnifiedSetTable) -> dict:
    """转为可 JSON 序列化的 dict."""
    return {
        "K": ust.K,
        "centers": ust.centers.tolist() if ust.centers is not None else [],
        "concentrations": ust.concentrations.tolist() if ust.concentrations is not None else [],
        "proportions": ust.proportions.tolist() if ust.proportions is not None else [],
        "cones_95_deg": ust.cones_95.tolist() if ust.cones_95 is not None else [],
        "source_map": ust.source_map if ust.source_map else [],
        "conflict_flags": ust.conflict_flags if ust.conflict_flags else [],
        "structure_types": ust.structure_types if ust.structure_types else [],
    }
