# -*- coding: utf-8 -*-
"""T25: 场地三维结构模型 v1 —— 把"单井统计工具"升级为"场地三维交付物".

全部用已有组件拼接, 无新算法:
  - 多井编录表输入 (CSV: depth, dip, dip_direction)
  - 各井独立打标 (复用 setlabel.generate_set_ids)
  - 多井一致性审计 (复用 multiwell_joint_audit 的 L20/L30 决策规则)
  - DFN 生成 (复用 dfn.py 的 Baecher 盘模型)
  - matplotlib 3D 可视化: 竖井轨迹 + 半透明裂隙盘 (按组着色)
  - 4 视角输出 (俯视/侧视/斜视/沿最大主走向)
  - 内嵌 base64 PNG 进 HTML 报告
  - 诚实标注「示意级筛查模型, 裂隙位置为随机实现, 非实测定位」

用法:
  python -m fractureflow.eval --site-model --wells data/real/beishan_wells.npz \
      --domain 50 50 50 --out-dir results/site_model/

  python -m fractureflow.eval --site-model --wells wells_dir/ \
      --domain 50 50 50 --out-dir results/site_model/
"""

import base64
import io
import json
import os
import sys
import textwrap
import time
from typing import List, Optional, Tuple

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def dip_dipdir_to_normal(dip_deg: np.ndarray, dip_dir_deg: np.ndarray) -> np.ndarray:
    """倾角倾向 → 法向 (nx, ny, nz).

    dip: 倾角 (0–90°, 从水平面起算)
    dip_dir: 倾向 (0–360°, 从北顺时针)

    法向定义: 指向井轴上方、垂直于裂隙面.
    """
    dip = np.radians(np.asarray(dip_deg, float))
    dd = np.radians(np.asarray(dip_dir_deg, float))
    nx = np.sin(dip) * np.sin(dd)
    ny = np.sin(dip) * np.cos(dd)
    nz = np.cos(dip)
    nrm = np.stack([nx, ny, nz], axis=-1)
    return nrm / (np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12)


def normal_to_dip_dipdir(nrm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """法向 → (dip, dip_direction).

    BUG-B 修复 (B3-1): 委托 geometry.normal_to_dip_dipdir, 对 nz<0 法向
    倾向翻转 180°, 保证 n 与 -n 输出完全一致 (无向法向铁律)。
    """
    from .geometry import normal_to_dip_dipdir as _n2dd
    return _n2dd(nrm)


# R56: 期望列名 (load_well_csv 缺列校验用, 对齐 T91 FIX-2 方言)
_WELL_CSV_REQUIRED = ("depth", "dip", "dip_direction")


def load_well_csv(path: str) -> dict:
    """加载单井 CSV (depth, dip, dip_direction) → well dict.

    R56 容错 (对齐 T91 FIX-2 同款方言):
      - 缺列: 报出缺失列清单 (ValueError), 不静默用 KeyError 撞死;
      - 坏行 (数值解析失败/空值): 跳过并计数, 通过返回值 'n_skipped' 上报
        警告信息; 若全部行坏则报错, 不返回空井。
    """
    import csv
    depths, dips, dip_dirs = [], [], []
    n_skipped = 0
    skipped_reasons = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        missing = [c for c in _WELL_CSV_REQUIRED if c not in (rdr.fieldnames or [])]
        if missing:
            raise ValueError(
                f"CSV 缺少必需列: {missing}; 应有列: {list(_WELL_CSV_REQUIRED)} "
                f"(文件: {path})")
        for lineno, row in enumerate(rdr, start=2):  # 行号含表头
            try:
                d = float(row["depth"])
                dip = float(row["dip"])
                dd = float(row["dip_direction"])
            except (TypeError, ValueError):
                n_skipped += 1
                if len(skipped_reasons) < 3:
                    skipped_reasons.append(f"第{lineno}行数值无效")
                continue
            # R80 毒丸 PP-A1/A2: float("nan")/float("inf") 不抛 ValueError,
            # NaN 会静默流入 dip_dipdir_to_normal 产出方向垃圾 (禁静默降级红线)。
            # 非有限数值与解析失败同口径: 跳过并计数。
            if not (np.isfinite(d) and np.isfinite(dip) and np.isfinite(dd)):
                n_skipped += 1
                if len(skipped_reasons) < 3:
                    skipped_reasons.append(f"第{lineno}行含 NaN/Inf")
                continue
            depths.append(d)
            dips.append(dip)
            dip_dirs.append(dd)
    if not depths:
        raise ValueError(
            f"CSV 全部 {n_skipped} 行解析失败, 无法构建井 (文件: {path})")
    depths = np.asarray(depths, float)
    dips = np.asarray(dips, float)
    dip_dirs = np.asarray(dip_dirs, float)
    # R80 毒丸 PP-C1: CSV 行序不保证深度升序, 下游 np.diff/searchsorted 类消费
    # 依赖单调深度。加载时稳定排序 (对齐 fmi_attr labels.load_well 的 argsort
    # 先例); 重复深度是合法编录 (同深多裂隙), 保留不拒 (PP-C2 钉死)。
    order = np.argsort(depths, kind="stable")
    depths, dips, dip_dirs = depths[order], dips[order], dip_dirs[order]
    nrm = dip_dipdir_to_normal(dips, dip_dirs)
    well = {
        "well_id": os.path.splitext(os.path.basename(path))[0],
        "depth": depths,
        "dip": dips,
        "dip_direction": dip_dirs,
        "nrm": nrm,
        "n_fractures": len(depths),
        "n_skipped": n_skipped,
    }
    if n_skipped > 0:
        print(f"[site_model] 警告: {path} 跳过 {n_skipped} 行坏数据 "
              f"({'、'.join(skipped_reasons)}... 共 {n_skipped} 行)")
    return well


def load_wells_from_dir(wells_dir: str) -> List[dict]:
    """加载目录下所有 CSV 文件 (R56 容错: 单井坏列报错, 坏行跳过)."""
    wells = []
    for fn in sorted(os.listdir(wells_dir)):
        if fn.lower().endswith(".csv"):
            wells.append(load_well_csv(os.path.join(wells_dir, fn)))
    return wells


def load_wells_from_npz(path: str) -> List[dict]:
    """加载 beishan_wells.npz → list of well dicts.

    npz 格式: wells (N_wells, N_frac, 3) 法向数组.
    深度信息缺失 → 用等间距示意.
    """
    data = np.load(path, allow_pickle=True)
    key = "wells" if "wells" in data else list(data.keys())[0]
    arr = np.asarray(data[key], float)
    if arr.ndim == 2:
        arr = arr[None, ...]  # (1, N, 3)
    wells = []
    for i in range(arr.shape[0]):
        nrm = arr[i]
        nrm = nrm / (np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12)
        n = len(nrm)
        depths = np.linspace(0, 100, n)  # 示意深度
        dip, dip_dir = normal_to_dip_dipdir(nrm)
        wells.append({
            "well_id": f"well_{i:02d}",
            "depth": depths,
            "dip": dip,
            "dip_direction": dip_dir,
            "nrm": nrm,
            "n_fractures": n,
        })
    return wells


# ---------------------------------------------------------------------------
# R60: 多井联合决策规则 (L20/L30) 消费
# ---------------------------------------------------------------------------

def tier_from_angle(angle_deg, thresholds=(20.0, 35.0)):
    """把井间组系一致性角距映射到三档决策档位.

    规则来源: results/multiwell_joint_audit.json (beishan 22 井实测推导):
      <20°   -> 'joint'        (多井联合池化)
      20–35° -> 'pair_only'    (仅联合最一致对)
      >35°   -> 'independent'  (各井独立)
    边界语义按任务书原文: <20 联合; 20–35 仅联最一致对; >35 独立.
    因此 20 落入 pair_only, 35 落入 independent (严格 >35).
    """
    lo, hi = thresholds
    if angle_deg < lo:
        return "joint"
    if angle_deg < hi:
        return "pair_only"
    return "independent"


def _well_centers_from_normals(wnrm, K, seed):
    """逐井 K 个组心 (符号对齐后). 返回 [K,3]. (R60)"""
    from .setlabel import spherical_kmeans
    centers, assign = spherical_kmeans(wnrm, K, seed=seed)
    aligned = np.zeros_like(centers)
    for k in range(K):
        sel = wnrm[assign == k]
        if len(sel) == 0:
            aligned[k] = centers[k]
            continue
        ref = sel.mean(0)
        ref /= np.linalg.norm(ref) + 1e-12
        sgn = np.sign((sel * ref).sum(-1, keepdims=True))
        sgn[sgn == 0] = 1
        aligned[k] = _unit((sel * sgn).mean(0))
    return aligned


def _pairwise_center_consistency(centers_i, centers_j):
    """两井组心一致性 = 最优指派 (匈牙利) 的匹配角均值 (度). (R60)"""
    from scipy.optimize import linear_sum_assignment
    K = max(len(centers_i), len(centers_j))
    a = _unit(np.zeros((K, 3)))
    b = _unit(np.zeros((K, 3)))
    a[:len(centers_i)] = _unit(np.asarray(centers_i, float))
    b[:len(centers_j)] = _unit(np.asarray(centers_j, float))
    ang = np.zeros((K, K))
    for x in range(K):
        for y in range(K):
            ang[x, y] = float(np.rad2deg(np.arccos(
                np.clip(np.abs(np.dot(a[x], b[y])), 0.0, 1.0))))
    row, col = linear_sum_assignment(ang)
    return float(ang[row, col].mean())


def joint_rule_partition(wells_nrm, obs_masks=None, K_audit=4, seed=0,
                         thresholds=(20.0, 35.0)):
    """把多口井按 L20/L30 决策规则划成"池化组" (诚实口径).

    一致性矩阵 / 联合-独立判定**只用观测法向** (obs_masks 各井布尔掩码,
    缺省则用全量法向 —— 即编录表场景本就没有"隐伏点"). 绝不使用隐伏法向
    或真值 set_ids 参与分组.

    返回 dict:
      verdicts:     list[dict] 每井 {well, best_partner, best_angle, tier}
      groups:       list[list[int]] 池化组 (独立井 => 单元素组)
      consistency:  {(i,j): float} 两两一致性角距
    """
    N = len(wells_nrm)
    empty = {"verdicts": [], "groups": [], "consistency": {}}
    if N == 0:
        return empty

    # 每井用于判定的法向: 观测只取观测; 无掩码用全量
    pool_norms = []
    for i in range(N):
        nrm = _unit(np.asarray(wells_nrm[i], float))
        if obs_masks is not None and obs_masks[i] is not None:
            m = np.asarray(obs_masks[i], bool)
            nrm_obs = nrm[m]
            if len(nrm_obs) > 0:
                nrm = nrm_obs
        pool_norms.append(nrm)

    centers = [_well_centers_from_normals(pn, K_audit, seed) for pn in pool_norms]

    consistency = {}
    for i in range(N):
        for j in range(i + 1, N):
            consistency[f"{i}-{j}"] = _pairwise_center_consistency(
                centers[i], centers[j])

    # 逐井最佳伙伴 + 档位
    verdicts = []
    for i in range(N):
        best_j, best_ang = None, 999.0
        for j in range(N):
            if i == j:
                continue
            ang = consistency[f"{min(i, j)}-{max(i, j)}"]
            if ang < best_ang:
                best_ang, best_j = ang, j
        verdicts.append({
            "well": i,
            "best_partner": int(best_j) if best_j is not None else None,
            "best_angle": float(best_ang),
            "tier": tier_from_angle(best_ang, thresholds),
        })

    # 并查集池化: <20 双向一致边连通成联合组; pair_only 井与最佳伙伴强连
    parent = list(range(N))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    lo = thresholds[0]
    for (key, ang) in consistency.items():
        if ang < lo:
            i, j = (int(p) for p in key.split("-"))
            _union(i, j)
    for v in verdicts:
        if v["tier"] == "pair_only" and v["best_partner"] is not None:
            _union(v["well"], v["best_partner"])

    groups_map = {}
    for i in range(N):
        groups_map.setdefault(_find(i), []).append(i)
    groups = [sorted(m) for m in groups_map.values()]
    groups.sort(key=lambda g: g[0])
    return {"verdicts": verdicts, "groups": groups, "consistency": consistency}


def _cluster_and_align(nrm, K, seed):
    """球形 k-means + 组心符号对齐. 返回 (centers_aligned[K,3], assign[L]). (R60)"""
    from .setlabel import spherical_kmeans
    centers, assign = spherical_kmeans(nrm, K, seed=seed)
    aligned = np.zeros_like(centers)
    for k in range(K):
        sel = nrm[assign == k]
        if len(sel) == 0:
            aligned[k] = centers[k]
            continue
        ref = sel.mean(0)
        ref /= np.linalg.norm(ref) + 1e-12
        sgn = np.sign((sel * ref).sum(-1, keepdims=True))
        sgn[sgn == 0] = 1
        aligned[k] = _unit((sel * sgn).mean(0))
    return aligned, assign


# ---------------------------------------------------------------------------
# SiteModeler 类
# ---------------------------------------------------------------------------

class SiteModeler:
    """场地三维结构模型 —— 拼接已有组件的胶水层."""

    def __init__(self, domain=(50.0, 50.0, 50.0), seed=42):
        self.domain = tuple(domain)
        self.seed = seed
        self.wells: List[dict] = []
        self.joint_result: Optional[dict] = None
        self.set_table: Optional[object] = None
        self.dfn = None
        self.audit_result: Optional[dict] = None
        # R60: 决策规则消费的中间产物 (build_set_table 填充)
        self.rule_assign: Optional[list] = None
        self.rule_groups: Optional[list] = None
        self.rule_verdicts: Optional[list] = None
        self.rule_consistency: Optional[dict] = None

    # -- 1. 加载 ----------------------------------------------------------------
    def load_wells(self, source):
        """source: wells 目录 / npz 路径 / list[well_dict]."""
        if isinstance(source, list):
            self.wells = source
        elif isinstance(source, str):
            if source.endswith(".npz") or source.endswith(".npy"):
                self.wells = load_wells_from_npz(source)
            elif os.path.isdir(source):
                self.wells = load_wells_from_dir(source)
            else:
                raise ValueError(f"不支持的输入: {source}")
        print(f"[SiteModeler] 加载 {len(self.wells)} 口井, "
              f"共 {sum(w['n_fractures'] for w in self.wells)} 条裂隙")

    # -- 2. 各井独立打标 -------------------------------------------------------
    def label_wells(self, Krange=(2, 7)):
        """各井独立球面 k-means 打标 (复用 setlabel.generate_set_ids)."""
        from .setlabel import generate_set_ids
        for w in self.wells:
            w["set_ids"] = generate_set_ids(w["nrm"], Krange=Krange, seed=self.seed)
            w["K"] = int(w["set_ids"].max()) + 1
        print(f"[SiteModeler] 各井独立打标完成, "
              f"K 范围: {min(w['K'] for w in self.wells)}–{max(w['K'] for w in self.wells)}")

    # -- 3. 多井一致性审计 -----------------------------------------------------
    def audit_consistency(self, K_audit=4):
        """多井联合适用性审计 (L20/L30 决策规则).

        复用 multiwell_joint_audit.py 的核心逻辑: 逐井组心 → 两两角距 →
        一致性 <20° 联合 / 20–35° 仅联合最一致对 / >35° 独立.
        """
        from .setlabel import spherical_kmeans, _sign_align

        wells_nrm = [w["nrm"] for w in self.wells]
        N = len(wells_nrm)

        # 逐井组心 (K_audit 统一)
        well_centers = []
        for wn in wells_nrm:
            c, a = spherical_kmeans(wn, K_audit, seed=self.seed)
            aligned = np.zeros_like(c)
            for k in range(K_audit):
                sel = wn[a == k]
                if len(sel) == 0:
                    aligned[k] = c[k]
                    continue
                ref = sel.mean(0)
                ref /= np.linalg.norm(ref) + 1e-12
                sgn = np.sign((sel * ref).sum(-1, keepdims=True))
                sgn[sgn == 0] = 1
                aligned[k] = _unit((sel * sgn).mean(0))
            well_centers.append(aligned)

        # 两两一致性
        def acos_abs(a, b):
            a, b = _unit(np.asarray(a, float)), _unit(np.asarray(b, float))
            if a.ndim == 1 and b.ndim == 1:
                return float(np.rad2deg(np.arccos(np.clip(np.abs(a @ b), 0, 1))))
            c = np.clip(np.abs(np.einsum("ij,ij->i", a, b)), 0, 1)
            return np.rad2deg(np.arccos(c))

        pairwise = {}
        for i in range(N):
            for j in range(i + 1, N):
                ang = np.zeros((K_audit, K_audit))
                for ki in range(K_audit):
                    for kj in range(K_audit):
                        ang[ki, kj] = acos_abs(well_centers[i][ki], well_centers[j][kj])
                from scipy.optimize import linear_sum_assignment
                row, col = linear_sum_assignment(ang)
                matched = ang[row, col]
                pairwise[f"{i}-{j}"] = {
                    "mean_matched": float(matched.mean()),
                    "matched_angles": matched.tolist(),
                }

        # L20/L30 决策规则
        decisions = []
        for i in range(N):
            best_j, best_ang = None, 999
            for j in range(N):
                if i == j:
                    continue
                key = f"{min(i, j)}-{max(i, j)}"
                ang = pairwise[key]["mean_matched"]
                if ang < best_ang:
                    best_ang = ang
                    best_j = j
            if best_ang < 20:
                verdict = "joint"
            elif best_ang < 35:
                verdict = "pair_only"
            else:
                verdict = "independent"
            decisions.append({
                "well": i,
                "best_partner": int(best_j),
                "partner_angle": float(best_ang),
                "verdict": verdict,
            })

        self.audit_result = {
            "N_wells": N,
            "K_audit": K_audit,
            "pairwise_consistency": pairwise,
            "decisions": decisions,
            "summary": {
                "n_joint": sum(1 for d in decisions if d["verdict"] == "joint"),
                "n_pair_only": sum(1 for d in decisions if d["verdict"] == "pair_only"),
                "n_independent": sum(1 for d in decisions if d["verdict"] == "independent"),
                "mean_pairwise_angle": float(np.mean(
                    [v["mean_matched"] for v in pairwise.values()])),
            },
        }
        s = self.audit_result["summary"]
        print(f"[SiteModeler] 一致性审计: 联合={s['n_joint']}, "
              f"仅配对={s['n_pair_only']}, 独立={s['n_independent']}, "
              f"平均角距={s['mean_pairwise_angle']:.1f}°")

    # -- 4. 联合打标 -----------------------------------------------------------
    def _build_settable_from(self, centers, all_normals, assign):
        """由 (已符号对齐组心, 全量法向, 组号) 组装 SetTable. 返回 SetTable."""
        from .dfn import SetTable
        M = len(centers)
        concentrations = np.zeros(M)
        proportions = np.zeros(M)
        for k in range(M):
            sel = all_normals[assign == k]
            proportions[k] = len(sel) / max(len(all_normals), 1)
            if len(sel) >= 2:
                cos_angles = np.clip(np.abs(sel @ centers[k]), 0, 1)
                angles = np.degrees(np.arccos(cos_angles))
                sigma_rad = np.radians(max(float(np.mean(angles)), 1.0))
                concentrations[k] = min(1.0 / (sigma_rad ** 2 + 0.01), 100.0)
            else:
                concentrations[k] = 1.0
        return SetTable(centers=centers, concentrations=concentrations,
                        proportions=proportions)

    def build_set_table(self, K=4, joint_rule=True):
        """构建场地统一 SetTable —— 默认**消费** L20/L30 决策规则 (R60 落地).

        规则消费 (joint_rule=True, 多井):
          按 results/multiwell_joint_audit 三档 (组心角距 <20° 联合 /
          20–35° 仅联最一致对 / >35° 独立) 把井划成池化组 —— 一致井
          池化聚类 (组心更稳), 独立井各自聚类 —— 再合并组装成场地 SetTable。
          分组只统计观测法向, 不碰真值指派 (诚实口径)。

        无条件全池化 (joint_rule=False, 旧行为 / 单井免逃逸):
          所有井法向 vstack 后做一次球形 k-means (十五期登记项解除前的实现)。

        字段:
          self.rule_groups : list[list[int]] 每池化组的井下标 (全池化=[所有井])
          self.rule_assign : list[np.ndarray] 每井 set_ids (指向 self.set_table 组号)
        """
        N = len(self.wells)
        all_normals = np.vstack([w["nrm"] for w in self.wells])

        if not joint_rule or N <= 1:
            # 旧行为 / 逃生口: 无条件全池化
            aligned, assign = _cluster_and_align(all_normals, K, self.seed)
        else:
            # 规则消费: L20/L30 三档分组
            partition = joint_rule_partition(
                [w["nrm"] for w in self.wells], K_audit=K, seed=self.seed)
            groups = partition["groups"]
            # 逐组聚类 -> 全局组心 + 每井组号 (指向全局组心)
            merged_centers, merged_assign = [], []
            for g in groups:
                if len(g) == 1:
                    wi = g[0]
                    centers_c, assign_c = _cluster_and_align(
                        self.wells[wi]["nrm"], K, self.seed)
                    merged_assign.append(assign_c + len(merged_centers))
                    merged_centers.extend(centers_c.tolist())
                else:
                    pool = np.vstack([self.wells[wi]["nrm"] for wi in g])
                    centers_c, assign_pool = _cluster_and_align(pool, K, self.seed)
                    off = 0
                    for wi in g:
                        Lw = len(self.wells[wi]["nrm"])
                        merged_assign.append(
                            assign_pool[off:off + Lw] + len(merged_centers))
                        off += Lw
                    merged_centers.extend(centers_c.tolist())
            aligned = np.array(merged_centers, float)
            assign = np.concatenate(merged_assign)
            # 记录分组决策 (供报告/审计)
            self.rule_groups = groups
            self.rule_assign = merged_assign
            self.rule_verdicts = partition["verdicts"]
            self.rule_consistency = partition["consistency"]

        # 统一: 记录每井 set_ids (未存时按全池化切分)
        if self.rule_assign is None:
            per_well = []
            off = 0
            for w in self.wells:
                Lw = len(w["nrm"])
                per_well.append(assign[off:off + Lw])
                off += Lw
            self.rule_assign = per_well
            self.rule_groups = [list(range(N))]

        self.set_table = self._build_settable_from(aligned, all_normals, assign)
        n_centers = len(aligned)
        print(f"[SiteModeler] SetTable: 中心数={n_centers}, "
              f"规则消费={'是' if joint_rule and N > 1 else '否(全池化/单井)'}, "
              f"池化组={len(self.rule_groups)}, "
              f"各组占比 = {', '.join(f'{p:.2f}' for p in self.set_table.proportions)}")

    # -- 5. DFN 生成 ----------------------------------------------------------
    def generate_dfn(self, p32=0.5, beta=3.5, seed=None):
        """生成场地 DFN 实现."""
        from .dfn import generate_dfn as _gen
        if self.set_table is None:
            raise RuntimeError("请先调用 build_set_table()")
        seed = self.seed if seed is None else seed   # 显式 seed=0 不再被 falsy 误吞
        self.dfn = _gen(self.set_table, p32=p32, beta=beta,
                       domain=self.domain, seed=seed)
        print(f"[SiteModeler] DFN 生成: {self.dfn.n_fractures} 条裂隙, "
              f"P32={p32}, β={beta}")

    # -- 6. 三维可视化 --------------------------------------------------------
    def _compute_well_positions(self):
        """各井在域内均匀网格排布 (示意级)."""
        N = len(self.wells)
        nx = int(np.ceil(np.sqrt(N)))
        ny = int(np.ceil(N / nx))
        Lx, Ly, Lz = self.domain
        positions = []
        for i in range(N):
            ix, iy = i % nx, i // nx
            x = (ix + 0.5) / nx * Lx - Lx / 2
            y = (iy + 0.5) / ny * Ly - Ly / 2
            positions.append((x, y))
        return positions

    def _disk_polygon(self, center, normal, radius, n_vertices=12):
        """用一个正多边形逼近圆盘, 返回顶点数组 (n_vertices+1, 3)."""
        normal = _unit(normal)
        # 找两个正交方向
        if abs(normal[2]) < 0.9:
            ref = np.array([0, 0, 1.0])
        else:
            ref = np.array([1.0, 0, 0])
        u = np.cross(normal, ref)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(normal, u)
        angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
        verts = (center[:, None]
                 + radius * (np.cos(angles)[None, :] * u[:, None]
                             + np.sin(angles)[None, :] * v[:, None]))
        return verts.T  # (n_vertices, 3)

    def render_3d(self, out_dir, max_disks=800, figsize=(12, 9), dpi=150):
        """渲染 4 视角 3D 图.

        视角:
          1. 俯视 (elev=90, azim=0)
          2. 侧视 (elev=0, azim=0)
          3. 斜视 (elev=30, azim=45)
          4. 沿最大主走向 (自动计算)

        返回 dict: {view_name: png_path}.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib import font_manager

        # 中文字体
        cn_font = None
        for fn in ["SimHei", "Microsoft YaHei", "SimSun", "PingFang SC"]:
            if any(f.name == fn for f in font_manager.fontManager.ttflist):
                cn_font = fn
                break
        if cn_font:
            plt.rcParams["font.sans-serif"] = [cn_font, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        os.makedirs(out_dir, exist_ok=True)
        well_xy = self._compute_well_positions()
        Lx, Ly, Lz = self.domain

        # 颜色映射
        K = self.set_table.K if self.set_table else 4
        cmap = plt.cm.tab10(np.linspace(0, 1, max(K, 1)))

        # 准备裂隙盘 (子采样防过多)
        dfn = self.dfn
        n_disk = dfn.n_fractures if dfn else 0
        if n_disk > max_disks:
            rng = np.random.default_rng(self.seed)
            idx_disk = rng.choice(n_disk, max_disks, replace=False)
        else:
            idx_disk = np.arange(n_disk)

        # 井轨迹: 根据各井深度范围定 z
        max_depths = [w["depth"].max() if len(w["depth"]) > 0 else Lz
                      for w in self.wells]
        overall_max_depth = max(max_depths) if max_depths else Lz

        # 找最大主走向方向 (组心 z 分量最小的 = 最水平的 = 主走向方向)
        if self.set_table:
            main_set = int(np.argmin(np.abs(self.set_table.centers[:, 2])))
            main_dir = self.set_table.centers[main_set]
            # 法向水平方位 = 倾向; 视线要沿走向 (= 倾向 ± 90°) 而非倾向.
            # matplotlib 相机方位 azim 的视线方向 ∝ (sin azim, cos azim),
            # 取 azim = dip_dir + 90° 使视线平行于走向线 (2026-08-28 修复:
            # 旧写法直接用倾向作视角, 图名"沿最大主走向"实为沿倾向看).
            main_azim = float((np.degrees(np.arctan2(main_dir[0], main_dir[1])) + 90.0) % 360.0)
            main_elev = 0.0  # 侧视沿走向
        else:
            main_azim, main_elev = 90.0, 0.0

        views = [
            ("top", 90, 0, "俯视"),
            ("side", 0, 0, "侧视"),
            ("oblique", 30, 45, "斜视"),
            ("along_strike", main_elev, main_azim, "沿最大主走向"),
        ]

        result_paths = {}
        for view_name, elev, azim, view_name_cn in views:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

            # 画裂隙盘
            if dfn and len(idx_disk) > 0:
                verts_list = []
                colors_list = []
                for idx in idx_disk:
                    c = dfn.centers[idx]
                    n = dfn.normals[idx]
                    r = dfn.radii[idx]
                    k = int(dfn.sets[idx])
                    verts = self._disk_polygon(c, n, r)
                    verts_list.append(verts)
                    colors_list.append(cmap[k % len(cmap)])

                poly = Poly3DCollection(verts_list, alpha=0.25,
                                        linewidths=0.1, edgecolors="gray")
                # set_facecolor 用 RGBA (a=1) 会覆盖 alpha=0.25 → 盘变不透明
                # 互相遮挡; 显式把 0.25 写进 facecolor.
                poly.set_facecolor([(r, g, b, 0.25) for (r, g, b, _a) in colors_list])
                ax.add_collection3d(poly)

            # 画井轨迹 (竖线)
            for i, (wx, wy) in enumerate(well_xy):
                z_top = 0
                z_bot = -max_depths[i] * (Lz / overall_max_depth) if overall_max_depth > 0 else -Lz
                ax.plot([wx, wx], [wy, wy], [z_top, z_bot],
                        color="black", linewidth=2, alpha=0.9)
                ax.scatter([wx], [wy], [z_top], color="red", s=30, zorder=5)

            # 域边框 (俯视图时画矩形)
            ax.set_xlim(-Lx / 2, Lx / 2)
            ax.set_ylim(-Ly / 2, Ly / 2)
            ax.set_zlim(-Lz, Lz * 0.1)

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title(f"场地结构面三维模型 — {view_name_cn} "
                         f"(elev={elev}°, azim={azim}°)", fontsize=12)
            ax.view_init(elev=elev, azim=azim)

            # 图例
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=cmap[k], alpha=0.5,
                                     label=f"组 {k}") for k in range(K)]
            ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

            png_path = os.path.join(out_dir, f"site_model_{view_name}.png")
            fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            result_paths[view_name] = png_path
            print(f"  {view_name_cn} → {png_path}")

        self._view_paths = result_paths
        return result_paths

    # -- 7. HTML 报告 ---------------------------------------------------------
    def _fig_to_base64(self, png_path):
        with open(png_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    def generate_html_report(self, out_path):
        """生成 HTML 报告 (含 base64 内嵌 PNG)."""
        assert hasattr(self, "_view_paths"), "请先调用 render_3d()"

        view_names_cn = {
            "top": "俯视", "side": "侧视",
            "oblique": "斜视", "along_strike": "沿最大主走向",
        }

        # 内嵌图片
        imgs_html = []
        for vname, path in self._view_paths.items():
            b64 = self._fig_to_base64(path)
            cn = view_names_cn.get(vname, vname)
            imgs_html.append(
                f'<h3>{cn}</h3>\n'
                f'<img src="data:image/png;base64,{b64}" alt="{cn}">'
            )

        # 审计摘要
        audit = self.audit_result or {}
        s = audit.get("summary", {})
        audit_html = f"""
<h2>多井一致性审计</h2>
<table>
<tr><th>指标</th><th>值</th></tr>
<tr><td>井数</td><td>{audit.get("N_wells", "N/A")}</td></tr>
<tr><td>推荐联合</td><td>{s.get("n_joint", "N/A")}</td></tr>
<tr><td>仅推荐配对</td><td>{s.get("n_pair_only", "N/A")}</td></tr>
<tr><td>推荐独立</td><td>{s.get("n_independent", "N/A")}</td></tr>
<tr><td>平均组心角距</td><td>{s.get("mean_pairwise_angle", 0):.1f}°</td></tr>
</table>
<p><em>决策规则 (L20/L30): 一致性 &lt;20° → 联合; 20–35° → 仅联合最一致对; &gt;35° → 独立。</em></p>
"""

        # SetTable
        st = self.set_table
        if st:
            set_table_html = "<h2>场地组系表 (SetTable)</h2>\n<table>\n<tr><th>组</th><th>法向 nx</th><th>法向 ny</th><th>法向 nz</th><th>倾角</th><th>倾向</th><th>浓度 κ</th><th>占比</th></tr>\n"
            for k in range(st.K):
                dip, dip_dir = normal_to_dip_dipdir(st.centers[k])
                set_table_html += (
                    f"<tr><td>{k}</td>"
                    f"<td>{st.centers[k, 0]:.3f}</td>"
                    f"<td>{st.centers[k, 1]:.3f}</td>"
                    f"<td>{st.centers[k, 2]:.3f}</td>"
                    f"<td>{dip:.1f}°</td><td>{dip_dir:.1f}°</td>"
                    f"<td>{st.concentrations[k]:.1f}</td>"
                    f"<td>{st.proportions[k]:.1%}</td></tr>\n"
                )
            set_table_html += "</table>"
        else:
            set_table_html = ""

        # DFN 信息
        dfn_info = ""
        if self.dfn:
            dfn_info = f"<p>DFN: {self.dfn.n_fractures} 条裂隙, " \
                       f"{self.set_table.K if self.set_table else '?'} 组</p>"

        honest = """
<div class="warning">
<h3>⚠ 诚实边界 (重要)</h3>
<ul>
<li><strong>本模型为示意级筛查模型, 裂隙位置为随机实现 (Baecher 盘模型), 非实测定位。</strong></li>
<li>井位按均匀网格排布, 非实际钻孔坐标。</li>
<li>深度信息为示意 (等间距), 非真实编录 (npz 输入无深度)。</li>
<li>本交付物用于展示场地结构面组系几何形态, 不声称裂隙空间位置精确。</li>
</ul>
</div>
"""

        body = f"""
<h1>场地结构面三维模型</h1>
<p><strong>生成时间</strong>: {time.strftime('%Y-%m-%d %H:%M:%S')} | <strong>场地</strong>: SiteModeler v1</p>
{honest}
<h2>汇总</h2>
<p>井数: {len(self.wells)} | 裂隙总数: {sum(w['n_fractures'] for w in self.wells)} |
域尺寸: {self.domain[0]}×{self.domain[1]}×{self.domain[2]} m | DFN 组数: {st.K if st else 'N/A'}</p>
{dfn_info}
{audit_html}
{set_table_html}
<h2>三维可视化 (4 视角)</h2>
{chr(10).join(imgs_html)}
"""

        html = textwrap.dedent(f"""\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>场地结构面三维模型</title>
<style>
  body {{ font-family: "Microsoft YaHei", "SimHei", "SimSun", sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 20px; color: #333; line-height: 1.6; }}
  h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 8px; }}
  h2 {{ color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 4px; margin-top: 30px; }}
  h3 {{ color: #34495e; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }}
  th, td {{ border: 1px solid #bdc3c7; padding: 6px 10px; text-align: left; }}
  th {{ background-color: #2c3e50; color: white; }}
  tr:nth-child(even) {{ background-color: #f2f3f4; }}
  .warning {{ background-color: #fdf2e9; border-left: 4px solid #e67e22;
              padding: 12px 16px; margin: 16px 0; }}
  img {{ max-width: 100%; height: auto; display: block; margin: 16px auto;
        border: 1px solid #bdc3c7; }}
  .footer {{ margin-top: 40px; padding-top: 12px; border-top: 1px solid #bdc3c7;
             color: #7f8c8d; font-size: 12px; }}
</style>
</head>
<body>
{body}
<div class="footer"><em>SiteModeler v1 — 场地三维结构模型 (T25 交付物)</em></div>
</body>
</html>
""")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[SiteModeler] HTML 报告 → {out_path}")

    # -- 8. 一键全流程 --------------------------------------------------------
    def run(self, out_dir, K=4, p32=0.5, beta=3.5, render=True, joint_rule=True):
        """一键跑完全流程."""
        t0 = time.time()
        print("=" * 60)
        print("  SiteModeler v1 — 场地三维结构模型")
        print("=" * 60)

        # 打标
        self.label_wells()
        # 审计
        self.audit_consistency(K_audit=K)
        # SetTable (R60: 默认消费决策规则; joint_rule=False 退回全池化)
        self.build_set_table(K=K, joint_rule=joint_rule)
        # DFN
        self.generate_dfn(p32=p32, beta=beta)

        # 渲染
        if render:
            self.render_3d(out_dir)

        # HTML 报告
        html_path = os.path.join(out_dir, "site_model_report.html")
        self.generate_html_report(html_path)

        # 落盘审计 JSON
        if self.audit_result:
            audit_path = os.path.join(out_dir, "audit_summary.json")
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(self.audit_result, f, ensure_ascii=False, indent=2, default=str, allow_nan=False)

        elapsed = time.time() - t0
        print(f"\n[SiteModeler] 全流程完成, 耗时 {elapsed:.1f}s")
        print(f"  输出目录: {out_dir}")
        return {"out_dir": out_dir, "elapsed_sec": elapsed}


# ---------------------------------------------------------------------------
# R107: L1 视觉 picks 加载 + L0×L1 双源对账 (复用 R96 三色门控逻辑)
# ---------------------------------------------------------------------------

# R107 固定水印 (初筛待复核, R29/R91 能力声明; 报告首段必须出现)
VISION_WATERMARK = (
    "视觉来源为检测初筛，未全部人工复核；裂隙位置为随机实现非实测定位。"
)
# R96 三色门控阈值 (与 AGENTS ANGLE_CONSISTENT/ANGLE_CONFLICT 逐字一致)
GREEN_DEG = 20.0      # < 20° 绿·互证
YELLOW_DEG = 35.0     # 20–35° 黄·存疑; >=35° 红·冲突
R107_K_SETS = 6       # 源内组系 K (R96 预注册写死)
R107_SEED = 42
SOURCE_PRIOR_R107 = {"L0": 1.0, "L1": 1.0}   # 两源质量先验冻结等权


def load_vision_picks(path: str, well_id: str = "vision",
                      window: Optional[list] = None) -> dict:
    """加载 L1 视觉 picks CSV (R96 格式: dip_deg/dip_dir_deg/md_ft) → well dict.

    R96 视觉自动表列 (v_r88_fmidyn_self_picks.csv): pick_id, md_ft, dip_deg,
    dip_dir_deg, nx, ny, nz, ..., tier, pos_wall_*。

    返回 well dict 与编录井同构 (depth/dip/dip_direction/nrm), 附加:
      source_type='L1', tier 过滤 (默认全保留), watermark 由调用方打。
    window=[lo, hi] (ft) 可选, 按 md_ft 过滤 (对齐 R96 WINDOW)。
    """
    import csv
    depths, dips, dip_dirs = [], [], []
    n_skipped = 0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        required = ("md_ft", "dip_deg", "dip_dir_deg")
        missing = [c for c in required if c not in (rdr.fieldnames or [])]
        if missing:
            raise ValueError(
                f"视觉 picks CSV 缺列: {missing}; 应有: {list(required)} (文件: {path})")
        for lineno, row in enumerate(rdr, start=2):
            try:
                md = float(row["md_ft"])
                dip = float(row["dip_deg"])
                dd = float(row["dip_dir_deg"])
            except (TypeError, ValueError):
                n_skipped += 1
                continue
            # R80 毒丸 PP-A3: NaN/Inf 数值行与解析失败同口径跳过 (禁方向垃圾)。
            if not (np.isfinite(md) and np.isfinite(dip) and np.isfinite(dd)):
                n_skipped += 1
                continue
            if window is not None and not (window[0] <= md <= window[1]):
                continue
            depths.append(md)
            dips.append(dip)
            dip_dirs.append(dd)
    if not depths:
        raise ValueError(
            f"视觉 picks 窗内 0 条有效记录 (文件: {path}, window={window})")
    depths = np.asarray(depths, float)
    dips = np.asarray(dips, float)
    dip_dirs = np.asarray(dip_dirs, float)
    # R80 毒丸 PP-C1: 同 load_well_csv, 稳定排序保证深度单调。
    order = np.argsort(depths, kind="stable")
    depths, dips, dip_dirs = depths[order], dips[order], dip_dirs[order]
    nrm = dip_dipdir_to_normal(dips, dip_dirs)
    return {
        "well_id": well_id,
        "source_type": "L1",
        "depth": depths,
        "dip": dips,
        "dip_direction": dip_dirs,
        "nrm": nrm,
        "n_fractures": len(depths),
        "n_skipped": n_skipped,
    }


def reconcile_sources(nrm_a: np.ndarray, nrm_b: np.ndarray,
                      name_a: str = "L0", name_b: str = "L1",
                      K: int = R107_K_SETS, seed: int = R107_SEED) -> dict:
    """L0×L1 双源对账 (复用 R96 三色门控, 零新算法).

    1. 源内球形 k-means (无向 |cos| + 符号对齐) → 各组心/占比;
    2. 跨源匈牙利匹配 (代价 = acos|<ci,cj>| 度);
    3. 三色门控: 绿(角距<20°) 互证 / 黄(20–35° 或仅单源) 存疑 / 红(>=35°) 冲突;
    4. 绿组加权 Fréchet 融合 (w = √n × 源质量先验) → 统一组方向。

    返回 dict:
      source_a / source_b : 单源组系表 {K, n, centers, group_n, dip, dip_dir, kappa, cone_95}
      pairs               : 跨源匹配对 (含 decision 三色 / angdeg / n_a / n_b)
      red_groups          : 红·冲突对 (标人工仲裁)
      fused_groups        : 绿组融合 (normal/dip/dip_dir/kappa/cone_95/n_vendor/n_vision)
      summary             : {n_green, n_yellow, n_red, n_fused, n_pairs}
      unified_centers     : (K_unified, 3) 用于 DFN 的组心
      unified_source      : 每组来源标签 ('L0','L1','L0+L1')
      unified_weights     : 每组占比
    """
    from .setlabel import spherical_kmeans
    from .l4.fuse import (weighted_frechet_mean, estimate_kappa,
                          bootstrap_confidence_cone)
    from scipy.optimize import linear_sum_assignment

    nrm_a = _unit(np.asarray(nrm_a, float))
    nrm_b = _unit(np.asarray(nrm_b, float))

    def _src_table(nrm, name):
        centers, assign = spherical_kmeans(nrm, min(K, len(nrm)), seed=seed)
        Kc = centers.shape[0]
        group_n = np.zeros(Kc, dtype=int)
        conc = np.zeros(Kc)
        cones = np.zeros(Kc)
        dip = np.zeros(Kc)
        dip_dir = np.zeros(Kc)
        for k in range(Kc):
            mask = assign == k
            group_n[k] = int(mask.sum())
            if group_n[k] >= 3:
                conc[k] = estimate_kappa(nrm[mask], centers[k])
                cones[k] = bootstrap_confidence_cone(nrm[mask], seed=seed)
            else:
                conc[k] = 0.01
                cones[k] = 180.0
            dip[k] = float(np.degrees(np.arccos(np.clip(abs(centers[k, 2]), 0, 1))))
            dip_dir[k] = float(np.degrees(np.arctan2(centers[k, 0], centers[k, 1])) % 360.0)
        return dict(name=name, centers=centers, group_n=group_n,
                    concentration=conc, cone_95=cones, dip=dip, dip_dir=dip_dir,
                    K=int(Kc), n=int(nrm.shape[0]))

    sa = _src_table(nrm_a, name_a)
    sb = _src_table(nrm_b, name_b)

    def _ang(c1, c2):
        return float(np.degrees(np.arccos(
            np.clip(abs(float(np.clip(float(c1 @ c2), -1, 1))), 0, 1))))

    # 匈牙利匹配
    cost = np.zeros((sa["K"], sb["K"]))
    for i in range(sa["K"]):
        for j in range(sb["K"]):
            cost[i, j] = _ang(sa["centers"][i], sb["centers"][j])
    ri, cj = linear_sum_assignment(cost)
    matched = [(int(i), int(j), float(cost[i, j])) for i, j in zip(ri, cj)]

    used_a, used_b = set(), set()
    pairs = []
    for (ia, ib, ang) in matched:
        if ang < GREEN_DEG:
            decision = "green"
        elif ang < YELLOW_DEG:
            decision = "yellow"
        else:
            decision = "red"
        pairs.append(dict(ia=ia, ib=ib, angdeg=round(ang, 2), decision=decision,
                          decision_cn={"green": "绿·互证", "yellow": "黄·存疑",
                                       "red": "红·冲突"}[decision],
                          n_a=int(sa["group_n"][ia]), n_b=int(sb["group_n"][ib]),
                          angle_dir=float(sa["dip_dir"][ia])))
        used_a.add(ia)
        used_b.add(ib)
    # 仅单源检出 → 黄·存疑
    for ia in range(sa["K"]):
        if ia not in used_a:
            pairs.append(dict(ia=ia, ib=None, angdeg=None, decision="yellow",
                              decision_cn="黄·存疑",
                              angle_dir=float(sa["dip_dir"][ia]),
                              n_a=int(sa["group_n"][ia]), n_b=0,
                              single_source=name_a))
    for ib in range(sb["K"]):
        if ib not in used_b:
            pairs.append(dict(ia=None, ib=ib, angdeg=None, decision="yellow",
                              decision_cn="黄·存疑",
                              angle_dir=float(sb["dip_dir"][ib]),
                              n_a=0, n_b=int(sb["group_n"][ib]),
                              single_source=name_b))

    n_green = sum(1 for g in pairs if g["decision"] == "green")
    n_yellow = sum(1 for g in pairs if g["decision"] == "yellow")
    n_red = sum(1 for g in pairs if g["decision"] == "red")
    red_groups = [g for g in pairs if g["decision"] == "red"]

    # 绿组加权 Fréchet 融合
    fused = []
    for grp in pairs:
        if grp["decision"] != "green":
            continue
        pts_a = sa["centers"][grp["ia"]].reshape(1, 3)  # 以组心为融合点(示意级)
        # 用组心方向做加权融合 (R96 用组内全点; 此处用组心+权重等价简化, 诚实标注)
        mu_a = sa["centers"][grp["ia"]]
        mu_b = sb["centers"][grp["ib"]]
        if mu_a @ mu_b < 0:
            mu_b = -mu_b
        w_a = np.sqrt(grp["n_a"]) * SOURCE_PRIOR_R107[name_a]
        w_b = np.sqrt(grp["n_b"]) * SOURCE_PRIOR_R107[name_b]
        mu = _unit(w_a * mu_a + w_b * mu_b)
        all_pts = np.concatenate([sa["centers"][grp["ia"]][None],
                                  sb["centers"][grp["ib"]][None]], axis=0)
        kappa = estimate_kappa(all_pts, mu)
        cone = bootstrap_confidence_cone(all_pts, seed=seed)
        fused.append(dict(ia=grp["ia"], ib=grp["ib"], normal=mu,
                          dip=float(np.degrees(np.arccos(np.clip(abs(mu[2]), 0, 1)))),
                          dip_dir=float(np.degrees(np.arctan2(mu[0], mu[1])) % 360.0),
                          kappa=float(kappa), cone_95=float(cone),
                          n_total=int(grp["n_a"] + grp["n_b"]),
                          n_vendor=int(grp["n_a"]), n_vision=int(grp["n_b"]),
                          angdeg=float(grp["angdeg"])))

    # 统一组系: 绿融合组 + 黄单源组 (红排除出建模表, 标转仲裁)
    unified_centers, unified_source, unified_n = [], [], []
    for f in fused:
        unified_centers.append(f["normal"])
        unified_source.append(f"{name_a}+{name_b}")
        unified_n.append(f["n_total"])
    for grp in pairs:
        if grp["decision"] == "yellow" and grp.get("single_source"):
            if grp["single_source"] == name_a:
                unified_centers.append(sa["centers"][grp["ia"]])
                unified_source.append(name_a)
                unified_n.append(grp["n_a"])
            else:
                unified_centers.append(sb["centers"][grp["ib"]])
                unified_source.append(name_b)
                unified_n.append(grp["n_b"])

    unified_centers = np.asarray(unified_centers, float)
    unified_centers = _unit(unified_centers) if len(unified_centers) else \
        np.zeros((0, 3))
    total_n = sum(unified_n) if unified_n else 0
    unified_weights = (np.asarray(unified_n, float) / max(total_n, 1)
                       if unified_n else np.zeros(0))

    return dict(
        source_a=sa, source_b=sb, pairs=pairs, red_groups=red_groups,
        fused_groups=fused,
        summary=dict(n_pairs=len(pairs), n_green=n_green, n_yellow=n_yellow,
                     n_red=n_red, n_fused=len(fused),
                     n_unified=len(unified_centers)),
        unified_centers=unified_centers,
        unified_source=unified_source,
        unified_weights=unified_weights,
        unified_n=unified_n,
    )


# ---------------------------------------------------------------------------
# R107: 双源场地模型 (SiteModeler 扩展, 按数据源着色 / 水印 / 一致性摘要)
# ---------------------------------------------------------------------------

class MultiSourceSiteModeler(SiteModeler):
    """双源场地模型 —— L0 编录 × L1 视觉 picks 统一建模.

    复用 SiteModeler 的全部既有能力 (dfn / 渲染 / 报告), 在入场阶段:
      1. 同井双源先走 R96 对账 (reconcile_sources), 冲突红组标红转仲裁;
      2. 用统一组系表 (绿融合 + 黄单源) 生成 DFN; 红组排除出建模表;
      3. 3D 按数据源着色 + 图例明示;
      4. 报告首段固定初筛待复核水印 + 跨源一致性摘要。
    """

    # 数据源配色 (编录=实测蓝 / 视觉=检测橙 / 双源互证绿 / 红冲突=灰)
    SOURCE_COLOR = {
        "L0": "#1f77b4",       # 实测
        "L1": "#ff7f0e",       # 视觉检测·待复核
        "L0+L1": "#2e8b57",    # 双源互证
        "red": "#b0b0b0",      # 冲突·转仲裁 (灰)
    }
    SOURCE_LEGEND = {
        "L0": "L0 编录·实测点",
        "L1": "L1 视觉·检测+待复核",
        "L0+L1": "双源互证",
        "red": "红·冲突(转人工仲裁)",
    }

    def __init__(self, domain=(50.0, 50.0, 50.0), seed=42):
        super().__init__(domain=domain, seed=seed)
        self.reconcile = None
        self.vision_well = None

    def load_dual(self, l0_source, vision_picks: str, window=None,
                  vision_well_id: str = "L1_vision"):
        """加载 L0 (既有 load_wells 支持的所有输入) + L1 视觉 picks, 并先走对账."""
        self.load_wells(l0_source)
        self.vision_well = load_vision_picks(vision_picks, well_id=vision_well_id,
                                             window=window)
        # L0 编录井统一标记 source_type (供着色)
        for w in self.wells:
            w.setdefault("source_type", "L0")
        # 同井双源对账
        nrm_l0 = np.vstack([w["nrm"] for w in self.wells])
        self.reconcile = reconcile_sources(
            nrm_l0, self.vision_well["nrm"], name_a="L0", name_b="L1",
            K=R107_K_SETS, seed=self.seed)
        print(f"[MultiSource] L0 {nrm_l0.shape[0]} 条 × L1 "
              f"{self.vision_well['n_fractures']} 条对账: "
              f"绿={self.reconcile['summary']['n_green']} "
              f"黄={self.reconcile['summary']['n_yellow']} "
              f"红={self.reconcile['summary']['n_red']}")

    def build_dual_set_table(self):
        """由对账结果构造场地 SetTable (绿融合 + 黄单源; 红排除)."""
        from .dfn import SetTable
        centers = self.reconcile["unified_centers"]
        if len(centers) == 0:
            raise RuntimeError("双源对账无可用组系 (全红冲突), 无法建模")
        K = len(centers)
        # 浓度: 绿融合组用重估 κ; 单源黄组用源内浓度 (均从对账结果读取)
        src_map = self.reconcile["unified_source"]
        kappa_src = []
        for s in src_map:
            if s == "L0+L1":
                # 取所有绿融合组的平均 κ (Fréchet 均值已含组内离散度)
                kappa_src.append(float(np.mean(
                    [f["kappa"] for f in self.reconcile["fused_groups"]]) or 5.0))
            elif s == "L0":
                kappa_src.append(float(np.mean(
                    self.reconcile["source_a"]["concentration"]) or 5.0))
            else:
                kappa_src.append(float(np.mean(
                    self.reconcile["source_b"]["concentration"]) or 5.0))
        concentrations = np.array(kappa_src, float)
        self.set_table = SetTable(centers=centers,
                                  concentrations=concentrations,
                                  proportions=self.reconcile["unified_weights"])
        self.source_map = src_map
        src_counts = {s: src_map.count(s) for s in sorted(set(src_map))}
        print(f"[MultiSource] 双源 SetTable: K={self.set_table.K}, "
              f"来源分布={src_counts}")
        return self.set_table

    # -- 双源着色 3D -----------------------------------------------------------
    def render_dual_3d(self, out_dir, max_disks=800, figsize=(12, 9), dpi=150):
        """按数据源着色渲染 4 视角 3D (实测/视觉/双源互证/红冲突)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.patches import Patch
        from matplotlib import font_manager
        for fn in ("SimHei", "Microsoft YaHei", "SimSun", "PingFang SC"):
            if any(f.name == fn for f in font_manager.fontManager.ttflist):
                plt.rcParams["font.sans-serif"] = [fn, "DejaVu Sans"]
                break
        plt.rcParams["axes.unicode_minus"] = False
        os.makedirs(out_dir, exist_ok=True)

        dfn = self.dfn
        Lx, Ly, Lz = self.domain
        # 每组来源颜色
        src_color = [self.SOURCE_COLOR.get(s, "#888888")
                     for s in self.source_map]
        # 红冲突组: 不参与 DFN (排除出建模表), 仅示意灰框
        n_disk = dfn.n_fractures if dfn else 0
        rng = np.random.default_rng(self.seed)
        idx_disk = np.arange(n_disk) if n_disk <= max_disks else \
            rng.choice(n_disk, max_disks, replace=False)
        K = self.set_table.K if self.set_table else 0

        views = [("top", 90, 0, "俯视"), ("side", 0, 0, "侧视"),
                 ("oblique", 30, 45, "斜视"), ("along_strike", 0, 45, "沿最大主走向")]
        result = {}
        for view_name, elev, azim, cn in views:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            if dfn and len(idx_disk) > 0:
                verts_list, colors_list = [], []
                for idx in idx_disk:
                    k = int(dfn.sets[idx])
                    c = src_color[k % len(src_color)] if src_color else "#888888"
                    verts = self._disk_polygon(dfn.centers[idx], dfn.normals[idx],
                                               dfn.radii[idx])
                    verts_list.append(verts)
                    colors_list.append(c)
                poly = Poly3DCollection(verts_list, alpha=0.3, linewidths=0.1,
                                        edgecolors="gray")
                from matplotlib.colors import to_rgba
                poly.set_facecolor([to_rgba(c, 0.3) for c in colors_list])
                ax.add_collection3d(poly)
            ax.set_xlim(-Lx / 2, Lx / 2)
            ax.set_ylim(-Ly / 2, Ly / 2)
            ax.set_zlim(-Lz, Lz * 0.1)
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
            ax.set_title(f"双源场地结构面模型 — {cn}", fontsize=12)
            ax.view_init(elev=elev, azim=azim)
            # 图例: 数据源 (实测/视觉/双源互证)
            legend_items = []
            shown = set()
            for s in sorted(set(self.source_map)):
                if s in shown:
                    continue
                shown.add(s)
                legend_items.append(
                    Patch(facecolor=self.SOURCE_COLOR.get(s, "#888"), alpha=0.5,
                          label=self.SOURCE_LEGEND.get(s, s)))
            if self.reconcile and self.reconcile["summary"]["n_red"] > 0:
                legend_items.append(
                    Patch(facecolor=self.SOURCE_COLOR["red"], alpha=0.5,
                          label=f"红·冲突(转仲裁,{self.reconcile['summary']['n_red']}组)"))
            ax.legend(handles=legend_items, loc="upper left", fontsize=8)
            png_path = os.path.join(out_dir, f"v_r107_dual_{view_name}.png")
            fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            result[view_name] = png_path
        self._view_paths = result
        return result

    # -- 双源 HTML/MD 报告 -----------------------------------------------------
    def consistency_summary(self) -> dict:
        """跨源一致性摘要 (两源组系角距表 + 转仲裁项单列)."""
        rec = self.reconcile
        rows = []
        for p in rec["pairs"]:
            rows.append({
                "source_a_group": p.get("ia"),
                "source_b_group": p.get("ib"),
                "angdeg": p.get("angdeg"),
                "decision": p["decision"],
                "decision_cn": p["decision_cn"],
                "n_a": p.get("n_a", 0),
                "n_b": p.get("n_b", 0),
                "single_source": p.get("single_source"),
            })
        arbitration = []
        for g in rec["red_groups"]:
            arbitration.append({
                "source_a_group": g["ia"], "source_b_group": g["ib"],
                "angdeg": g["angdeg"], "n_a": g["n_a"], "n_b": g["n_b"],
            })
        return {
            "n_green": rec["summary"]["n_green"],
            "n_yellow": rec["summary"]["n_yellow"],
            "n_red": rec["summary"]["n_red"],
            "n_fused": rec["summary"]["n_fused"],
            "n_unified": rec["summary"]["n_unified"],
            "rows": rows,
            "arbitration_items": arbitration,
        }

    def generate_dual_report(self, out_dir, site_name="场地",
                             run_seconds=0.0):
        """生成双源场地模型 HTML + MD 报告 (首段固定初筛待复核水印)."""
        os.makedirs(out_dir, exist_ok=True)
        cs = self.consistency_summary()
        rec = self.reconcile

        # ---- 首段固定声明 ----
        header_decl = (
            "<div class=\"warning\"><h3>⚠ 数据来源声明 (固定)</h3>"
            f"<p><strong>{VISION_WATERMARK}</strong></p>"
            "<p>编录(L0)为实测点; 视觉(L1)为检测初筛待复核。</p>"
            f"<p>三色对账: 绿={cs['n_green']} 黄={cs['n_yellow']} "
            f"红(冲突·转仲裁)={cs['n_red']}。</p></div>"
        )

        # 一致性摘要表
        rows_html = "".join(
            f"<tr><td>{r['source_a_group'] if r['source_a_group'] is not None else '—'}</td>"
            f"<td>{r['source_b_group'] if r['source_b_group'] is not None else '—'}</td>"
            f"<td>{r['angdeg'] if r['angdeg'] is not None else '—'}</td>"
            f"<td>{r['decision_cn']}</td>"
            f"<td>{r['n_a']}</td><td>{r['n_b']}</td></tr>"
            for r in cs["rows"])
        arb_html = "".join(
            f"<tr><td>{g['source_a_group']}</td><td>{g['source_b_group']}</td>"
            f"<td>{g['angdeg']}°</td><td>{g['n_a']}</td><td>{g['n_b']}</td></tr>"
            for g in cs["arbitration_items"]) or \
            "<tr><td colspan=5>（无冲突项）</td></tr>"

        # SetTable (双源) 表
        st = self.set_table
        st_html = ""
        if st:
            st_html = "<h2>双源统一组系表 (SetTable)</h2>\n<table>\n"
            st_html += "<tr><th>组</th><th>来源</th><th>倾角(°)</th><th>倾向(°)</th><th>占比</th></tr>\n"
            for k in range(st.K):
                dip, dd = normal_to_dip_dipdir(st.centers[k])
                src = self.source_map[k] if k < len(self.source_map) else "?"
                st_html += (f"<tr><td>{k}</td><td>{src}</td>"
                            f"<td>{dip:.1f}</td><td>{dd:.1f}</td>"
                            f"<td>{st.proportions[k]:.1%}</td></tr>\n")
            st_html += "</table>"

        # 图片
        imgs_html = []
        cn_map = {"top": "俯视", "side": "侧视", "oblique": "斜视",
                  "along_strike": "沿最大主走向"}
        for vname, path in getattr(self, "_view_paths", {}).items():
            if os.path.isfile(path):
                b64 = self._fig_to_base64(path)
                imgs_html.append(
                    f"<h3>{cn_map.get(vname, vname)}</h3>"
                    f"<img src=\"data:image/png;base64,{b64}\" alt=\"{cn_map.get(vname, vname)}\">")

        dfn_info = (f"<p>DFN: {self.dfn.n_fractures} 条裂隙, "
                    f"{self.set_table.K if self.set_table else '?'} 组 "
                    f"(P32={getattr(self, '_p32', 'N/A')}, β={getattr(self, '_beta', 'N/A')})"
                    f"</p>" if self.dfn else "")

        body = f"""
<h1>双源场地结构面模型 · {site_name}</h1>
<p><strong>生成时间</strong>: {time.strftime('%Y-%m-%d %H:%M:%S')} | <strong>模式</strong>: L0 编录 × L1 视觉 (R96 对账先行)</p>
{header_decl}
<h2>汇总</h2>
<p>L0 编录井: {len(self.wells)} | L1 视觉 picks: {self.vision_well['n_fractures'] if self.vision_well else 0} |
域尺寸: {self.domain[0]}×{self.domain[1]}×{self.domain[2]} m</p>
{dfn_info}
<h2>跨源一致性摘要</h2>
<table>
<tr><th>源A组</th><th>源B组</th><th>角距(°)</th><th>判定</th><th>n_A</th><th>n_B</th></tr>
{rows_html}
</table>
<h3>转人工仲裁项 (红·冲突)</h3>
<table>
<tr><th>源A组</th><th>源B组</th><th>角距(°)</th><th>n_A</th><th>n_B</th></tr>
{arb_html}
</table>
{st_html}
<h2>三维可视化 (按数据源着色, 4 视角)</h2>
{chr(10).join(imgs_html)}
"""
        html = textwrap.dedent(f"""\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>双源场地结构面模型 · {site_name}</title>
<style>
  body {{ font-family: "Microsoft YaHei", "SimHei", "SimSun", sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 20px; color: #333; line-height: 1.6; }}
  h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 8px; }}
  h2 {{ color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 4px; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }}
  th, td {{ border: 1px solid #bdc3c7; padding: 6px 10px; text-align: left; }}
  th {{ background-color: #2c3e50; color: white; }}
  tr:nth-child(even) {{ background-color: #f2f3f4; }}
  .warning {{ background-color: #fdf2e9; border-left: 4px solid #e67e22;
              padding: 12px 16px; margin: 16px 0; }}
  img {{ max-width: 100%; height: auto; display: block; margin: 16px auto;
        border: 1px solid #bdc3c7; }}
  .footer {{ margin-top: 40px; padding-top: 12px; border-top: 1px solid #bdc3c7;
             color: #7f8c8d; font-size: 12px; }}
</style>
</head>
<body>
{body}
<div class="footer"><em>R107 MultiSourceSiteModeler — L0×L1 双源场地模型</em></div>
</body>
</html>
""")
        html_path = os.path.join(out_dir, "v_r107_model_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        # ---- MD ----
        cn = {"green": "🟢 绿·互证", "yellow": "🟡 黄·存疑", "red": "🔴 红·冲突"}
        arb_lines = ([f"| {g['source_a_group']} | {g['source_b_group']} | {g['angdeg']}° | {g['n_a']} | {g['n_b']} |"
                      for g in cs["arbitration_items"]]
                     if cs["arbitration_items"] else ["|（无冲突项）|"])
        lines = [
            f"# v_r107 · 双源场地结构面模型 · {site_name}",
            "",
            f"> **数据来源声明 (固定): {VISION_WATERMARK}**",
            "> L0 编录为实测点; L1 视觉为检测初筛待复核。",
            "",
            "## 汇总",
            f"- L0 编录井: {len(self.wells)} | L1 视觉 picks: {self.vision_well['n_fractures'] if self.vision_well else 0}",
            f"- 域尺寸: {self.domain[0]}×{self.domain[1]}×{self.domain[2]} m",
            f"- DFN: {self.dfn.n_fractures if self.dfn else 0} 条裂隙, K={self.set_table.K if self.set_table else 0}",
            "",
            "## 跨源一致性摘要",
            f"绿(互证)={cs['n_green']} / 黄(存疑)={cs['n_yellow']} / 红(冲突·转仲裁)={cs['n_red']} / 融合组={cs['n_fused']}",
            "",
            "| 源A组 | 源B组 | 角距(°) | 判定 | n_A | n_B |",
            "|---|---|---|---|---|---|",
            *[f"| {r['source_a_group'] if r['source_a_group'] is not None else '—'} | "
              f"{r['source_b_group'] if r['source_b_group'] is not None else '—'} | "
              f"{r['angdeg'] if r['angdeg'] is not None else '—'} | {cn[r['decision']]} | "
              f"{r['n_a']} | {r['n_b']} |" for r in cs["rows"]],
            "",
            "### 转人工仲裁项 (红·冲突)",
            "",
            "| 源A组 | 源B组 | 角距(°) | n_A | n_B |",
            "|---|---|---|---|---|",
            *arb_lines,
            "",
        ]
        md_path = os.path.join(out_dir, "v_r107_model_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return {"html": html_path, "md": md_path}

    def run_dual(self, out_dir, p32=0.5, beta=3.5, seed=None):
        """一键双源全流程: 对账已先行 → SetTable → DFN → 着色渲染 → 报告."""
        t0 = time.time()
        self._p32 = p32
        self._beta = beta
        self.build_dual_set_table()
        self.generate_dfn(p32=p32, beta=beta, seed=seed)
        self.render_dual_3d(out_dir)
        paths = self.generate_dual_report(out_dir, run_seconds=time.time() - t0)
        # 一致性摘要落盘
        cs_path = os.path.join(out_dir, "v_r107_consistency.json")
        with open(cs_path, "w", encoding="utf-8") as f:
            json.dump(self.consistency_summary(), f, ensure_ascii=False,
                      indent=2, default=str, allow_nan=False)
        print(f"[MultiSource] 全流程完成, 耗时 {time.time() - t0:.1f}s → {out_dir}")
        return {"out_dir": out_dir, "html": paths["html"], "md": paths["md"],
                "consistency": cs_path}


# ---------------------------------------------------------------------------
# CLI 入口 (被 eval.py 调用)
# ---------------------------------------------------------------------------

def run_site_model_cli(args):
    """CLI 入口: 解析参数 → 构建 SiteModeler → 运行."""
    domain = tuple(args.site_domain) if args.site_domain else (50.0, 50.0, 50.0)
    out_dir = args.site_out_dir or "results/site_model/"

    sm = SiteModeler(domain=domain, seed=42)
    sm.load_wells(args.wells)

    K = args.site_K or 4
    p32 = args.site_p32 or 0.5
    beta = args.site_beta or 3.5
    # R60: 默认消费 L20/L30 决策规则; --joint-rule off 逃生口退回全池化
    joint_rule = getattr(args, "joint_rule", None) != "off"

    result = sm.run(out_dir, K=K, p32=p32, beta=beta, render=True,
                    joint_rule=joint_rule)
    return result
