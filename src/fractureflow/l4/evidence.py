# -*- coding: utf-8 -*-
"""T43: 统一证据模型 —— L0–L3 异构数据进同一框架.

统一证据条目 = (观测法向数组 | 走向约束, source_type L0–L3, n_obs, weight, provenance).

四源适配器:
  - L0: 钻孔编录表 (beishan npz / 客户 CSV depth,dip,dip_direction) → 法向观测
  - L1: FMI 迹线 (forge_fmi_2wells.pt) → 法向观测 (ENU, 与统一约定一致)
  - L2: Åland 露头迹线 → 走向约束条目 (法向垂直于迹线走向, 非直接观测)
  - L3: 稠密点云 (Pontrelli/HRF) → RANSAC 面法向 (带面内点数与残差)

坐标约定 (ENU 唯一化, 轴序 E,N,U; 与 read_forge_las/site_model 一致):
  - dip: 倾角 (0–90°, 从水平面起算)
  - dip_direction: 倾向 (0–360°, 从北顺时针)
  - 法向 n = [sin(dip)·sin(dd), sin(dip)·cos(dd), cos(dip)]
  - nz > 0 = 法向指上 (与 dip 0–90° 一致; 全项目实测数据均为 nz>0,
    血统仲裁见 docs/血统审计与数据血统仲裁_T58T60.md)

诚实边界:
  - L2 约束 ≠ 观测: 走向约束不参与均值投票, 只用于 T44 匹配验证与 T46 覆盖率检查
  - FORGE 坐标是局部 ENU, 与统一约定轴序一致 (血统仲裁 T58 已验证一致), provenance 记录来源

用法:
    from fractureflow.l4.evidence import build_bundle, from_beishan_npz, from_aland_traces
    bundle = build_bundle(
        from_beishan_npz("data/real/beishan_wells.npz"),
        from_aland_traces("data/.../getaberget_20m_1_traces.geojson"),
        site_name="synthetic_twin",
    )
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .constants import SOURCE_QUALITY_INITIAL, EPS, STRUCTURE_TYPE_UNKNOWN, STRUCTURE_TYPES


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class EvidenceEntry:
    """单条证据条目."""
    source_type: str                     # "L0" / "L1" / "L2" / "L3"
    source_name: str                     # 井名 / 场地名 / 窗口名 / 面编号
    data_type: str                       # "normal_obs" / "strike_constraint"
    # 法向观测 (L0/L1/L3), shape (N, 3) 单位向量
    normals: Optional[np.ndarray] = None
    # 走向约束 (L2 only), shape (M,) 走向方位角 (度, 从北顺时针 0–180)
    strikes: Optional[np.ndarray] = None
    # 结构面类型 (T50): ISRM 类型族 + 数据驱动类型; 缺省 unknown (向后兼容)
    structure_type: str = STRUCTURE_TYPE_UNKNOWN
    # 附加属性 (L3 per-surface metadata / L1 per-fracture attrs)
    meta: dict = field(default_factory=dict)
    # 来源权重 (初始等权, T46 回测可修订)
    weight: float = SOURCE_QUALITY_INITIAL
    # 来源元数据 (坐标约定 / 处理步骤 / 原始文件路径)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float64)
            # 单位化
            norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
            self.normals = self.normals / np.clip(norms, EPS, None)
        if self.strikes is not None:
            self.strikes = np.asarray(self.strikes, dtype=np.float64)
        # 结构面类型校验 (T50): 非法值回退 unknown
        if self.structure_type not in STRUCTURE_TYPES:
            self.structure_type = STRUCTURE_TYPE_UNKNOWN

    @property
    def n_obs(self) -> int:
        if self.normals is not None:
            return self.normals.shape[0]
        if self.strikes is not None:
            return self.strikes.shape[0]
        return 0

    @property
    def is_constraint_only(self) -> bool:
        """L2 走向约束 = True, 仅作匹配验证用, 不可进入均值投票."""
        return self.data_type == "strike_constraint"


@dataclass
class EvidenceBundle:
    """多源证据集合 —— L4 融合的入口."""
    entries: List[EvidenceEntry] = field(default_factory=list)
    site_name: str = ""
    bundle_meta: dict = field(default_factory=dict)

    def add(self, entry: EvidenceEntry) -> None:
        self.entries.append(entry)

    @property
    def n_sources(self) -> int:
        return len(self.entries)

    @property
    def source_types(self) -> List[str]:
        return sorted(set(e.source_type for e in self.entries))

    def get_observed_normals(self) -> List[Tuple[np.ndarray, EvidenceEntry]]:
        """返回所有法向观测 (不含 L2 约束) 的列表."""
        result = []
        for e in self.entries:
            if e.normals is not None and not e.is_constraint_only:
                result.append((e.normals, e))
        return result

    def summary(self) -> dict:
        """摘要: 各类型证据数量与观测数."""
        by_type: dict = {}
        for e in self.entries:
            by_type.setdefault(e.source_type, {"n_entries": 0, "n_obs": 0})
            by_type[e.source_type]["n_entries"] += 1
            by_type[e.source_type]["n_obs"] += e.n_obs
        return {
            "site_name": self.site_name,
            "n_sources": self.n_sources,
            "source_types": self.source_types,
            "by_type": by_type,
        }


# ---------------------------------------------------------------------------
# 坐标转换工具 (ENU 约定, 轴序 E,N,U)
# ---------------------------------------------------------------------------

def dip_dipdir_to_normal(dip_deg: np.ndarray, dip_dir_deg: np.ndarray) -> np.ndarray:
    """倾角倾向 → 单位法向 (ENU 约定: nz>0 指上, 轴序 E,N,U).

    dip: 0–90° 从水平起算
    dip_dir: 0–360° 从北顺时针
    """
    dip = np.radians(np.asarray(dip_deg, dtype=np.float64))
    dd = np.radians(np.asarray(dip_dir_deg, dtype=np.float64))
    nx = np.sin(dip) * np.sin(dd)
    ny = np.sin(dip) * np.cos(dd)
    nz = np.cos(dip)
    nrm = np.stack([nx, ny, nz], axis=-1)
    norms = np.linalg.norm(nrm, axis=-1, keepdims=True)
    return nrm / np.clip(norms, EPS, None)


def normal_to_dip_dipdir(nrm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """单位法向 → 倾角倾向 (ENU 约定)."""
    nrm = np.asarray(nrm, dtype=np.float64)
    norms = np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = nrm / np.clip(norms, EPS, None)
    # |nz| → dip (无向法向)
    dip = np.degrees(np.arccos(np.clip(np.abs(nrm[..., 2]), 0, 1)))
    # dip_dir from nx, ny (horizontal projection)
    dd = np.degrees(np.arctan2(nrm[..., 0], nrm[..., 1])) % 360
    return dip, dd


# ---------------------------------------------------------------------------
# L0 适配器: 钻孔编录表
# ---------------------------------------------------------------------------

def from_beishan_npz(path: str) -> List[EvidenceEntry]:
    """从 beishan_wells.npz 加载 L0 证据 (22 井, 每井预设法向).

    npz 中 wells[wi] 是 (Ni, 3) 单位法向量 (ENU 约定 nz>0, 与 FORGE 同帧, 血统仲裁已验证).
    注意: npz 内已是法向而非 dip/dip_dir, 不需转换.
    """
    path = os.path.abspath(path)
    d = np.load(path, allow_pickle=True)
    wells = d["wells"]  # (n_wells, max_per_well, 3)
    entries = []
    for wi in range(wells.shape[0]):
        nrm = np.asarray(wells[wi], dtype=np.float64)
        # 跳过全零行 (填充)
        valid = np.linalg.norm(nrm, axis=1) > 0.5
        nrm = nrm[valid]
        if len(nrm) == 0:
            continue
        # 单位化
        nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
        entries.append(EvidenceEntry(
            source_type="L0",
            source_name=f"beishan_w{wi:02d}",
            data_type="normal_obs",
            normals=nrm,
            weight=SOURCE_QUALITY_INITIAL,
            provenance={
                "source_file": path,
                "well_index": wi,
                "coordinate_system": "ENU",
                "n_raw": int(wells.shape[1]),
                "n_valid": int(valid.sum()),
                "note": "法向已预计算 (nz>0), 原始列顺序非 dip/dip_dir",
            },
        ))
    return entries


def from_borehole_csv(path: str, well_name: Optional[str] = None) -> EvidenceEntry:
    """从客户 CSV 加载单井 L0 证据 (列: depth, dip, dip_direction)."""
    import csv
    import warnings

    path = os.path.abspath(path)
    dips, dip_dirs = [], []
    n_raw = 0
    n_dropped = 0
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_raw += 1
            try:
                dips.append(float(row["dip"]))
                dip_dirs.append(float(row["dip_direction"]))
            except (KeyError, ValueError):
                n_dropped += 1
                continue
    if n_dropped:
        warnings.warn(
            f"[evidence.from_borehole_csv] {path} 丢弃 {n_dropped}/{n_raw} 行 "
            f"(列缺失或非法数值); 若比例异常请检查列名 (应为 dip/dip_direction).",
            stacklevel=2,
        )
    if n_raw - n_dropped == 0:
        raise ValueError(
            f"[evidence.from_borehole_csv] {path} 无有效行 (丢弃 {n_dropped}/{n_raw}); "
            f"列名应为 dip/dip_direction, 且为合法数值."
        )
    nrm = dip_dipdir_to_normal(np.array(dips), np.array(dip_dirs))
    if well_name is None:
        well_name = os.path.splitext(os.path.basename(path))[0]
    return EvidenceEntry(
        source_type="L0",
        source_name=well_name,
        data_type="normal_obs",
        normals=nrm,
        weight=SOURCE_QUALITY_INITIAL,
        provenance={
            "source_file": path,
            "coordinate_system": "ENU",
            "n_raw": len(dips),
        },
    )


# ---------------------------------------------------------------------------
# L1 适配器: FMI 迹线 (forge_fmi_2wells.pt)
# ---------------------------------------------------------------------------

def from_forge_fmi_pt(path: str) -> List[EvidenceEntry]:
    """从 forge_fmi_2wells.pt 加载 L1 证据.

    pt 结构: {"nets": [...], "widx": {...}, "summary": {...}}
    每个 net: dict with "nrm" (N,3), "src"="forge_fmi", "wid"=井名.

    ⚠️ FMI 法向坐标: FORGE 井迹积分输出局部 ENU. 本适配器直接取 net["nrm"]
    (well-local ENU; 2026-08-28 血统仲裁验证与 beishan/loaded_real 同帧, 无需旋转).

    T50: 按 ftype (natural/induced) 拆分子入口,结构类型记录在 structure_type 字段.
    """
    import torch
    path = os.path.abspath(path)
    data = torch.load(path, weights_only=False)
    nets = data["nets"]
    entries = []
    for i, net in enumerate(nets):
        nrm = np.asarray(net["nrm"], dtype=np.float64)
        # 零行防护 (与 from_beishan_npz 同姿态): 零向量除法产生 NaN 行会
        # 静默流入 spherical_kmeans, NaN argmax 指派不可控.
        row_norm = np.linalg.norm(nrm, axis=1, keepdims=True)
        keep = row_norm[:, 0] > 1e-12
        if not bool(keep.all()):
            nrm = nrm[keep]
        ftype_all = net.get("ftype")
        if ftype_all is not None:
            ftype_all = np.asarray(ftype_all)[keep]
        if nrm.shape[0] == 0:
            continue  # 整网全零法向, 跳过而非产出空条目
        nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
        wid = net.get("wid", f"forge_{i}")
        # T50: 按 ftype 拆分 (natural → structure_type="natural", induced → "induced")
        ftype = ftype_all
        if ftype is not None:
            ftype_arr = np.asarray(ftype)
            unique_types = np.unique(ftype_arr)
            for stype in unique_types:
                mask = ftype_arr == stype
                nrm_t = nrm[mask]
                if len(nrm_t) == 0:
                    continue
                entries.append(EvidenceEntry(
                    source_type="L1",
                    source_name=f"{wid}_{stype}",
                    data_type="normal_obs",
                    normals=nrm_t,
                    structure_type=str(stype),
                    weight=SOURCE_QUALITY_INITIAL,
                    provenance={
                        "source_file": path,
                        "coordinate_system": "ENU (well-local ENU, 血统仲裁 T58 验证与统一约定一致)",
                        "n_fractures": int(mask.sum()),
                        "well_id": wid,
                        "src": net.get("src", "unknown"),
                        "ftype": str(stype),
                        "ftype_counts": _count_ftypes(ftype),
                    },
                ))
        else:
            # 无 ftype 字段 → 回退单一条目 (缺省 unknown)
            entries.append(EvidenceEntry(
                source_type="L1",
                source_name=wid,
                data_type="normal_obs",
                normals=nrm,
                structure_type=STRUCTURE_TYPE_UNKNOWN,
                weight=SOURCE_QUALITY_INITIAL,
                provenance={
                    "source_file": path,
                    "coordinate_system": "ENU (well-local ENU, 血统仲裁 T58 验证与统一约定一致)",
                    "n_fractures": int(len(nrm)),
                    "well_id": wid,
                    "src": net.get("src", "unknown"),
                },
            ))
    return entries


def _count_ftypes(ftype) -> dict:
    """统计 ftype 分布 (用于 provenance)."""
    if ftype is None:
        return {}
    return dict(Counter(str(f) for f in np.asarray(ftype).ravel()))


# ---------------------------------------------------------------------------
# L2 适配器: Åland 露头迹线 (走向约束, 非法向观测)
# ---------------------------------------------------------------------------

def from_aland_traces(path_or_dir: str, vertical_dip: bool = True) -> List[EvidenceEntry]:
    """从 Åland 迹线 GeoJSON 加载 L2 证据.

    关键: L2 只提供走向方位 (strike azimuth), 非法向观测.
    走向约束记录在 strikes 字段, data_type="strike_constraint".
    """
    from ..aland import load_trace_window, aland_networks

    path_or_dir = os.path.abspath(path_or_dir)
    if os.path.isdir(path_or_dir):
        nets = aland_networks(traces_dir=path_or_dir, vertical_dip=vertical_dip)
    else:
        net = load_trace_window(path_or_dir, vertical_dip=vertical_dip)
        nets = [net] if net is not None else []

    entries = []
    for net in nets:
        if net is None:
            continue
        az = net.get("az")
        if az is None or len(az) == 0:
            continue
        strike_deg = np.asarray(az, dtype=np.float64) % 180
        name = net.get("name", "aland_window")
        entries.append(EvidenceEntry(
            source_type="L2",
            source_name=name,
            data_type="strike_constraint",
            strikes=strike_deg,
            weight=SOURCE_QUALITY_INITIAL,
            meta={
                "n_traces": len(strike_deg),
                "vertical_dip_assumed": vertical_dip,
                "set_ids": net.get("set_ids").tolist() if net.get("set_ids") is not None else None,
            },
            provenance={
                "source_file": path_or_dir,
                "coordinate_system": "local_2D (azimuth from map north)",
                "constraint_type": "normal perpendicular to trace strike",
                "note": "法向仅知在垂直于走向的大圆上, 不直接观测",
            },
        ))
    return entries


# ---------------------------------------------------------------------------
# L3 适配器: 稠密点云 (RANSAC 面法向)
# ---------------------------------------------------------------------------

def from_point_cloud(pos: np.ndarray,
                     normals: Optional[np.ndarray] = None,
                     seg_id: Optional[np.ndarray] = None,
                     source_name: str = "point_cloud",
                     run_ransac: bool = False,
                     ransac_kwargs: Optional[dict] = None) -> List[EvidenceEntry]:
    """从稠密点云加载 L3 证据.

    两种模式:
      - (a) 已分割: 传入 normals[N,3] + seg_id[N] (每点所属面号)
      - (b) run_ransac=True: 对 pos 跑 RANSAC 分割 (耗时 ~30s, 需较大内存)

    每面一个 EvidenceEntry, 面法向取 seg_id==si 点的均值法向 (符号对齐).
    """
    from ..segmentation import segment_planes_ransac

    pos = np.asarray(pos, dtype=np.float64)

    if normals is None and run_ransac:
        rkw = ransac_kwargs or {}
        seg_id_out, planes = segment_planes_ransac(pos, **rkw)
        # planes: list of plane objects with .normal
        surface_normals = []
        surface_npts = []
        for si, plane in enumerate(planes):
            mask = seg_id_out == si
            surface_normals.append(np.asarray(plane.normal).flatten())
            surface_npts.append(int(mask.sum()))
        return _build_l3_entries(surface_normals, surface_npts, source_name, "ransac_fit")

    if normals is not None and seg_id is not None:
        seg_id = np.asarray(seg_id, dtype=int)
        normals = np.asarray(normals, dtype=np.float64)
        unique_segs = np.unique(seg_id[seg_id >= 0])
        surface_normals = []
        surface_npts = []
        for si in unique_segs:
            mask = seg_id == si
            pts = normals[mask]
            # 符号对齐 + 均值 = 面法向
            ref = pts[0]
            sgn = np.sign((pts * ref).sum(-1, keepdims=True))
            sgn[sgn == 0] = 1
            pts = pts * sgn
            mean_n = pts.mean(0)
            mean_n = mean_n / (np.linalg.norm(mean_n) + EPS)
            surface_normals.append(mean_n)
            surface_npts.append(int(mask.sum()))
        return _build_l3_entries(surface_normals, surface_npts, source_name, "pre_segmented")

    raise ValueError("Must provide (normals + seg_id) OR run_ransac=True")


def _build_l3_entries(normals_list: list, npts_list: list,
                      source_name: str, method: str) -> List[EvidenceEntry]:
    """为每面构建一个 EvidenceEntry."""
    entries = []
    for si, (nrm, npts) in enumerate(zip(normals_list, npts_list)):
        nrm = np.asarray(nrm, dtype=np.float64).reshape(1, 3)
        nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + EPS)
        entries.append(EvidenceEntry(
            source_type="L3",
            source_name=f"{source_name}_s{si}",
            data_type="normal_obs",
            normals=nrm,
            weight=SOURCE_QUALITY_INITIAL,
            meta={
                "seg_id": si,
                "n_points": npts,
                "method": method,
            },
            provenance={
                "coordinate_system": "scan_local (客户需提供配准信息)",
                "normal_source": "SVD of inlier points (RANSAC plane)",
            },
        ))
    return entries


def from_outcrop_network(network, source_name: str = "outcrop") -> List[EvidenceEntry]:
    """从 OutcropNetwork 对象加载 L3 证据 (每面对应一个 entry)."""
    entries = []
    for s in network.surfaces:
        n = np.asarray(s.normal, dtype=np.float64).reshape(1, 3)
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + EPS)
        entry = EvidenceEntry(
            source_type="L3",
            source_name=f"{source_name}_s{s.seg_id}",
            data_type="normal_obs",
            normals=n,
            weight=SOURCE_QUALITY_INITIAL,
            meta={
                "seg_id": int(s.seg_id),
                "set_id": int(s.set_id),
                "n_points": int(s.n_points),
                "area_m2": float(s.area),
                "r_eq_m": float(s.r_eq),
                "residual_rms_m": float(s.residual_rms),
            },
            provenance={
                "coordinate_system": "scan_local",
                "normal_source": "RANSAC plane fit",
            },
        )
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# EvidenceBundle 便捷构造
# ---------------------------------------------------------------------------

def build_bundle(*source_entries_list: List[EvidenceEntry],
                 site_name: str = "",
                 bundle_meta: Optional[dict] = None) -> EvidenceBundle:
    """将多个 EvidenceEntry 列表合并为一个 EvidenceBundle."""
    bundle = EvidenceBundle(site_name=site_name, bundle_meta=bundle_meta or {})
    for entries in source_entries_list:
        for e in entries:
            bundle.add(e)
    return bundle
