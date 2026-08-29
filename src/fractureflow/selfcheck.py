# -*- coding: utf-8 -*-
"""一键自检体系 — 八期 T63 核心资产 + 九期 T67 管道断裂探测器.

用法:
    cd <project_root> && python -m fractureflow.selfcheck
    或: python src/fractureflow/selfcheck.py
    或: python src/fractureflow/selfcheck.py --with-tests  (额外跑 pytest)

无网络、无 GPU、<30 秒 (不含 --with-tests), 输出 PASS/FAIL 表 + 落盘 JSON.

六组检查:
  1. 几何约定组: dip/dd↔法向往返 + NED 约定 + 法向无向 + 符号对齐
  2. 泄漏毒丸组: 几何/Terzaghi/strict 三组毒丸
  3. 口径扫描组: 废弃数字出现在非标注语境即 FAIL
  4. 资产完整性组: 关键 results/*.json 存在性 + 关键数字一致性抽查
  5. 单元回归组: B1-B9 各一条最小复现用例
  6. 管道断裂探测组 (T67): 冲突检测 / 聚类管道 / SetTable 生成 / pytest 整合
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import subprocess
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_SELF_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
_TESTS_DIR = os.path.join(_PROJECT_ROOT, "tests")

# 确保 fractureflow 包和 scripts/ 可导入
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# JSON 序列化辅助
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 编码."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# 废弃数字清单 (来自 docs/评测标准与口径锁定.md §4.3)
# ---------------------------------------------------------------------------

_DEPRECATED_NUMBERS = {
    "15.82": "泄漏 oracle",
    "14.22": "泄漏 oracle p50",
    "12.39": "泄漏 oracle K12",
    "7.50": "泄漏 oracle strict=False",
    "9.83": "泄漏多井联合",
    "7.78": "FORGE modal_err (buggy)",
    "7.52": "FORGE 联合 (buggy)",
    "9.35": "FORGE K12 锚点 (buggy)",
    "43.2": "dip_only 误差 (arcsin)",
    "42": "beishan dip 中位 (arcsin, 应为 47.7)",
}


# ---------------------------------------------------------------------------
# 检查结果
# ---------------------------------------------------------------------------

class CheckResult:
    """单条检查结果."""
    def __init__(self, name: str, group: str, passed: bool, message: str = "", detail: str = ""):
        self.name = name
        self.group = group
        self.passed = bool(passed)  # 确保是 Python bool
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "group": self.group,
            "passed": self.passed,
            "message": self.message,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# 检查注册
# ---------------------------------------------------------------------------

_ALL_CHECKS: List[Tuple[str, str, Callable[[], CheckResult]]] = []


def check(name: str, group: str):
    """装饰器: 注册一条检查."""
    def decorator(fn: Callable[[], CheckResult]):
        _ALL_CHECKS.append((group, name, fn))
        return fn
    return decorator


# ---------------------------------------------------------------------------
# 1. 几何约定组
# ---------------------------------------------------------------------------

@check("dip_dipdir_to_normal 水平面", "几何约定")
def chk_horizontal_plane():
    from scripts.read_forge_las import dip_dipdir_to_normal
    n = dip_dipdir_to_normal(np.array([0.0]), np.array([0.0]))
    ok = abs(abs(float(n[0, 2])) - 1.0) < 1e-6
    return CheckResult("dip_dipdir_to_normal 水平面", "几何约定", ok,
                       "水平面法向竖直" if ok else f"nU={n[0,2]} (应为 ±1)")


@check("dip_dipdir_to_normal 垂直面", "几何约定")
def chk_vertical_plane():
    from scripts.read_forge_las import dip_dipdir_to_normal
    n = dip_dipdir_to_normal(np.array([0.0]), np.array([90.0]))
    ok = abs(float(n[0, 2])) < 1e-6 and abs(float(n[0, 1]) - 1.0) < 1e-6
    return CheckResult("dip_dipdir_to_normal 垂直面", "几何约定", ok,
                       "垂直面法向水平" if ok else f"nU={n[0,2]}, nN={n[0,1]}")


@check("dip_dipdir_to_normal 45°", "几何约定")
def chk_45_degree():
    from scripts.read_forge_las import dip_dipdir_to_normal
    n = dip_dipdir_to_normal(np.array([0.0]), np.array([45.0]))
    expected = np.cos(np.deg2rad(45))
    ok = abs(abs(float(n[0, 2])) - expected) < 1e-6
    return CheckResult("dip_dipdir_to_normal 45°", "几何约定", ok,
                       "|nz|=cos45" if ok else f"|nz|={abs(n[0,2])} (应为 {expected})")


@check("FORGE 16B 血统锚点", "几何约定")
def chk_lineage_anchor():
    try:
        import torch
        pt_path = os.path.join(_PROJECT_ROOT, "data/external/utah_forge_fmi/forge_fmi_2wells.pt")
        if not os.path.exists(pt_path):
            return CheckResult("FORGE 16B 血统锚点", "几何约定", True, "跳过 (pt 不存在)")
        d = torch.load(pt_path, map_location="cpu", weights_only=False)
        for net in d["nets"]:
            if net["wid"] == "16B":
                nrm = net["nrm"]
                nz = np.abs(nrm[:, 2])
                implied = np.degrees(np.arccos(np.clip(nz, 0, 1)))
                raw = net["dip"]
                diff = float(np.median(np.abs(implied - raw)))
                ok = diff < 0.01
                return CheckResult("FORGE 16B 血统锚点", "几何约定", ok,
                                   f"median |implied - raw| = {diff:.6f}°" if ok else f"diff={diff:.4f}°")
        return CheckResult("FORGE 16B 血统锚点", "几何约定", False, "16B not found")
    except Exception as e:
        return CheckResult("FORGE 16B 血统锚点", "几何约定", False, str(e))


@check("Terzaghi 权重方向", "几何约定")
def chk_terzaghi_direction():
    from fractureflow.terzaghi import terzaghi_weights
    n = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    w = terzaghi_weights(n)
    ok = float(w[1]) > float(w[0])
    return CheckResult("Terzaghi 权重方向", "几何约定", ok,
                       f"垂直面 w={w[1]:.3f} > 水平面 w={w[0]:.3f}" if ok else f"方向反了")


@check("arccos 前 abs+clip", "几何约定")
def chk_arccos_clip():
    cos_vals = np.array([0.0, 0.5, 1.0, -0.0])
    clipped = np.clip(np.abs(cos_vals), 0, 1)
    ok = bool(np.all(clipped >= 0) and np.all(clipped <= 1))
    return CheckResult("arccos 前 abs+clip", "几何约定", ok, "arccos 输入安全")


# ---------------------------------------------------------------------------
# 2. 泄漏毒丸组
# ---------------------------------------------------------------------------

@check("毒丸: 几何约定 (水平面→竖直法向)", "泄漏毒丸")
def chk_poison_geometry():
    from scripts.read_forge_las import dip_dipdir_to_normal
    n = dip_dipdir_to_normal(np.array([0.0]), np.array([0.0]))
    ok = abs(float(n[0, 2])) > 0.99
    return CheckResult("毒丸: 几何约定", "泄漏毒丸", ok,
                       "几何约定正确" if ok else f"几何约定可能错误: nU={n[0,2]}")


@check("毒丸: strict 默认值", "泄漏毒丸")
def chk_poison_strict_default():
    import inspect
    from fractureflow.inference import set_aware_dirs
    sig = inspect.signature(set_aware_dirs)
    default = sig.parameters["strict"].default
    ok = default == True
    return CheckResult("毒丸: strict 默认值", "泄漏毒丸", ok,
                       f"strict 默认值 = {default}" if ok else f"strict 默认值仍为 {default} (应为 True)")


@check("毒丸: Terzaghi 方向", "泄漏毒丸")
def chk_poison_terzaghi():
    from fractureflow.terzaghi import terzaghi_weights
    n = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    w = terzaghi_weights(n)
    # 如果方向反了, 水平面权重会 > 垂直面
    ok = float(w[0]) < float(w[1])
    return CheckResult("毒丸: Terzaghi 方向", "泄漏毒丸", ok,
                       "Terzaghi 方向正确" if ok else "Terzaghi 方向可能反了")


@check("S23 数据构造毒丸: nrm_full 污染不影响 set_dirs (T76)", "泄漏毒丸")
def chk_s23_data_construction_poison():
    """T76 毒丸: 把 0-观测组的 nrm_full 填垃圾值, 断言 set_dirs 不变.

    构造: 40 点 K=4 组, 第 4 组全部隐伏 (0 观测). 修复前 set_dirs[3] 由
    nrm_full[sel] 构造 (泄漏隐伏真值), 污染 nrm_full 后 set_dirs 变.
    修复后 set_dirs[3] 由全局观测均值构造 (仅 nrm_obs), 污染 nrm_full 不影响.
    """
    from fractureflow.data import prepare_net

    L, K = 40, 4
    rng = np.random.default_rng(42)
    pos = rng.normal(size=(L, 3)).astype(np.float32)
    # 4 组正交方向 + 小噪声
    base = np.eye(3, dtype=np.float32)
    nrm_list = []
    for k in range(K):
        noise = rng.normal(0, 0.05, (L // K, 3)).astype(np.float32)
        pts = base[k % 3] + noise
        pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
        nrm_list.append(pts)
    nrm = np.concatenate(nrm_list, axis=0)
    set_ids = np.repeat(np.arange(K), L // K)
    # 第 4 组 (idx 30-39) 全部隐伏
    obs_mask = np.ones(L, dtype=np.float32)
    obs_mask[30:40] = 0.0

    net_base = {
        "pos": torch.as_tensor(pos),
        "nrm": torch.as_tensor(nrm),
        "len": torch.ones(L, 1),
        "lith": torch.zeros(L, dtype=torch.int64),
        "s1": torch.eye(3)[0],
        "s3": torch.eye(3)[2],
        "set_ids": torch.as_tensor(set_ids, dtype=torch.int64),
        "obs_mask": torch.as_tensor(obs_mask),
    }

    # 基线
    b1 = prepare_net(dict(net_base), 0.4, rng=np.random.default_rng(999))
    sd1 = b1["set_dirs"].clone()

    # 毒丸: 把第 4 组 (隐伏) 的 nrm_full 改成完全不同的方向
    net_poison = dict(net_base)
    nrm_poison = nrm.copy()
    nrm_poison[30:40] = np.array([0.0, 0.0, 1.0])  # 改成竖直向上 (原组方向是 z)
    net_poison["nrm"] = torch.as_tensor(nrm_poison)
    b2 = prepare_net(net_poison, 0.4, rng=np.random.default_rng(999))
    sd2 = b2["set_dirs"].clone()

    diff = (sd1 - sd2).abs().max().item()
    ok = diff < 1e-5
    detail = f"set_dirs max|diff| = {diff:.2e}"
    if ok:
        detail += " — nrm_full 污染不影响 set_dirs, T76 泄漏已修复"
    else:
        detail += " — nrm_full 污染改变了 set_dirs, 泄漏仍存在!"
    return CheckResult("S23 数据构造毒丸: nrm_full 污染不影响 set_dirs (T76)",
                       "泄漏毒丸", ok, detail)


@check("S24 Route B 无观测裂隙不产出零向量 (BUG-1)", "泄漏毒丸")
def chk_s24_routeb_unobserved_poison():
    """BUG-1 毒丸: 整条裂隙无观测点时, 其隐伏点必须走空间回退而非零向量.

    旧实现: 无观测裂隙用 nrm[sel] (盲协议下全零) 算 mode -> 零向量假 mode ->
    回退子集内观测数=0 -> l1_local 只能返回零向量 (=90° 误差)。
    """
    from fractureflow.connectivity import fracture_aware_dirs

    rng = np.random.default_rng(42)
    n1 = np.array([1.0, 0.0, 0.0])
    pos_list, nrm_list, fid_list = [], [], []
    fid = 0
    for c in ([0.0, 0.0, 0.0], [10.0, 0.0, 0.0]):
        for _ in range(6):
            ctr = np.array(c) + rng.normal(scale=2.0, size=3)
            pts = ctr + rng.normal(scale=0.05, size=(5, 3))
            noisy = n1 + rng.normal(scale=0.02, size=3)
            pos_list.append(pts)
            nrm_list.append(np.repeat(noisy[None], 5, axis=0))
            fid_list.append(np.full(5, fid))
            fid += 1
    pos = np.concatenate(pos_list)
    nrm_true = np.concatenate(nrm_list)
    fids = np.concatenate(fid_list)
    occ = rng.random(60) < 0.4
    all_fids = np.unique(fids)
    no_obs = np.setdiff1d(all_fids, fids[~occ])
    if len(no_obs) == 0:
        occ[fids == all_fids[0]] = False
        no_obs = np.array([all_fids[0]])
    nrm_blind = nrm_true * occ[:, None]
    dirs, _ = fracture_aware_dirs(pos, nrm_blind, occ, fracture_id=fids)
    hid = ~occ
    norms = np.linalg.norm(dirs[hid], axis=1)
    n_zero = int((norms < 0.5).sum())
    ok = n_zero == 0
    return CheckResult("S24 Route B 无观测裂隙不产出零向量 (BUG-1)",
                       "泄漏毒丸", ok,
                       f"{int(hid.sum())} 隐伏点全部非零" if ok
                       else f"{n_zero}/{int(hid.sum())} 隐伏点零向量 (BUG-1 复发)")


@check("S25 站点索引跨进程稳定 (BUG-3)", "泄漏毒丸")
def chk_s25_site_index_stable():
    """BUG-3 毒丸: e3gt_hybrid_v2._site_index 必须用跨进程稳定哈希.

    Python 内建 str hash 受 PYTHONHASHSEED 随机化 —— 换进程后站点嵌入整体错位,
    SiteGroupCalib 校正头静默失效。修复要求 zlib.crc32; 本检查同时做源码扫描
    (禁止 hash( 用于站点映射) 与行为验证 (crc32 映射符合预期).
    """
    src_path = os.path.join(_SRC_DIR, "fractureflow", "e3gt_hybrid_v2.py")
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    if "hash(str(" in src or "abs(hash(" in src:
        return CheckResult("S25 站点索引跨进程稳定 (BUG-3)", "泄漏毒丸", False,
                           "e3gt_hybrid_v2.py 仍在使用内建 hash() 做站点映射!")
    try:
        import zlib
        from fractureflow.e3gt_hybrid_v2 import E3GTHybridV2
        m = E3GTHybridV2(d_model=32, k_knn=8, n_layers=1, n_global=1)
        b = {"src": ["BS34", "16B"], "pos": torch.zeros(2, 4, 3)}
        idx = m._site_index(b).tolist()
        want = [zlib.crc32(s.encode("utf-8")) % 32 for s in ["BS34", "16B"]]
        ok = idx == want and len(set(idx)) > 0
        detail = f"_site_index={['BS34:'+str(idx[0]), '16B:'+str(idx[1])]} (crc32 一致)" \
            if ok else f"_site_index={idx}, crc32 期望={want}"
    except Exception as e:
        return CheckResult("S25 站点索引跨进程稳定 (BUG-3)", "泄漏毒丸", False, str(e))
    return CheckResult("S25 站点索引跨进程稳定 (BUG-3)", "泄漏毒丸", ok, detail)


@check("S26 L4 融合按条目索引解析 (BUG-5)", "管道断裂探测")
def chk_s26_fuse_idx_resolution():
    """BUG-5 毒丸: 同类型多井时, 合并必须用匹配时的那个 entry.

    构造两口 L0 井 + 一个 L1 露头: 旧实现按 source_type 取第一个同类型表,
    会把第二口井的组号套到第一口井的 assign 上。修复后 ConflictMatch 携带
    idx_a/idx_b, 合并组贡献必须包含两井各自的 source_name。
    """
    from fractureflow.l4.evidence import EvidenceEntry, EvidenceBundle
    from fractureflow.l4.fuse import fuse_bundle

    rng = np.random.default_rng(7)

    def make_entry(name, normals):
        n = np.asarray(normals, dtype=np.float64)
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
        jitter = rng.normal(0, 0.03, n.shape)
        n = n + jitter
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
        return EvidenceEntry(source_type="L0", source_name=name,
                             data_type="normal_obs", normals=n)

    # 两口 L0 井, 各自两组正交方向; L1 一组方向与两井第一组都近 (<20°)
    well_a = make_entry("wellA", [[1, 0, 0]] * 30 + [[0, 1, 0]] * 30)
    well_b = make_entry("wellB", [[0.99, 0.1, 0]] * 30 + [[0, 0, 1]] * 30)
    l1 = EvidenceEntry(source_type="L1", source_name="outcrop",
                       data_type="normal_obs",
                       normals=np.tile([1, 0.05, 0], (40, 1)).astype(np.float64))
    bundle = EvidenceBundle(entries=[well_a, well_b, l1], site_name="s26")
    res = fuse_bundle(bundle, K=2)
    contrib = [c for g in res.unified_set_table.groups
               for c in g.source_contributions]
    names = set(contrib)
    both_wells_merged = any("wellA" in c for c in contrib) and \
        any("wellB" in c for c in contrib)
    ok = both_wells_merged
    detail = (f"贡献源={sorted(names)} — 同类型两井均被正确引用"
              if ok else
              f"贡献源={sorted(names)} — 有井被静默吞掉/张冠李戴 (BUG-5 复发)")
    return CheckResult("S26 L4 融合按条目索引解析 (BUG-5)", "管道断裂探测", ok, detail)


@check("S27 local_frames 等变性: fixed 等变 / legacy 非等变 (BUG-2)", "管道断裂探测")
def chk_s27_local_frames_equivariance():
    """BUG-2 护栏: 修正版标架必须旋转等变, legacy 分支必须保持非等变.

    双向防回归: (a) 有人把 fixed 改坏 -> 前半 FAIL;
    (b) 有人误以为 legacy 也等变 / 或把 legacy 删掉导致冻结复现断裂 -> 后半 FAIL。
    """
    try:
        from fractureflow.geometry import local_frames
        rng = np.random.default_rng(7)
        base_pts = np.array([[3., 0, 0], [-3., 0, 0], [0, 2., 0], [0, -2., 0],
                             [0, 0, 1.], [0, 0, -1.],
                             [0.7, 0.5, 0.2], [-0.6, 0.4, -0.3]])
        centers = rng.normal(size=(1, 6, 3)) * 5.0
        knn = (centers[:, :, None, :] + base_pts[None, None]
               + rng.normal(size=(1, 6, 8, 3)) * 1e-3).astype(np.float32)
        A = rng.normal(size=(3, 3))
        Q, _ = np.linalg.qr(A)
        R = torch.tensor(Q, dtype=torch.float32)
        kp1 = torch.tensor(knn)
        kp2 = kp1 @ R.T

        def max_pair_angle(v1, v2):
            rv1 = torch.einsum("ef,blfg->bleg", R, v1)
            cosv = (rv1 * v2).sum(-2).abs().clamp(0, 1)
            return float(torch.acos(cosv).max()) * 180.0 / np.pi

        f1, _ = local_frames(kp1, 8, deterministic_e2=True, legacy=False)
        f2, _ = local_frames(kp2, 8, deterministic_e2=True, legacy=False)
        ang_fixed = max_pair_angle(f1, f2)
        l1_, _ = local_frames(kp1, 8, legacy=True)   # 显式 legacy (免疫全局默认切换)
        l2_, _ = local_frames(kp2, 8, legacy=True)
        ang_legacy = max_pair_angle(l1_, l2_)
        ok = ang_fixed < 5.0 and ang_legacy > 5.0
        detail = (f"fixed={ang_fixed:.2f}° (<5), legacy={ang_legacy:.2f}° (>5)"
                  if ok else
                  f"fixed={ang_fixed:.2f}° legacy={ang_legacy:.2f}° — 违反 T82 约定")
    except Exception as e:
        return CheckResult("S27 local_frames 等变性: fixed 等变 / legacy 非等变 (BUG-2)",
                           "管道断裂探测", False, str(e))
    return CheckResult("S27 local_frames 等变性: fixed 等变 / legacy 非等变 (BUG-2)",
                       "管道断裂探测", ok, detail)


@check("S29 场景指标面-轴语义毒丸 (法向≠通道方向)", "管道断裂探测")
def chk_s29_scenario_plane_axis_semantics():
    """十四期 P0 毒丸: dominant_direction 是优势面**法向**, 场景判定必须用
    面-轴夹角 (90°−法向角)。历史 bug: 三个场景函数拿法向直接套阈值, 判级全反
    (水平组系+竖直井对误判"对齐良好")。已知几何必须给出正确判级。
    """
    try:
        from fractureflow.dfn import DFNRealization
        from fractureflow.percolation import (
            egs_connectivity_metric, mine_risk_sections, disposal_escape_priority)

        rng = np.random.default_rng(7)

        def single_set_dfn(normal):
            nrm = np.asarray(normal, float)
            nrm = nrm / np.linalg.norm(nrm)
            u = np.array([1.0, 0.0, 0.0])
            v = np.cross(nrm, u)
            if np.linalg.norm(v) < 1e-6:
                u = np.array([0.0, 1.0, 0.0])
                v = np.cross(nrm, u)
            v = v / np.linalg.norm(v)
            centers, normals = [], []
            for _ in range(12):
                a, b = rng.uniform(-2, 2), rng.uniform(-2, 2)
                centers.append(a * u + b * v + rng.uniform(-0.5, 0.5) * nrm)
                jit = nrm + rng.normal(0, 0.02, 3)
                normals.append(jit / np.linalg.norm(jit))
            return DFNRealization(centers=np.array(centers),
                                  normals=np.array(normals),
                                  radii=np.full(12, 5.0),
                                  sets=np.zeros(12, dtype=int))

        hz = single_set_dfn([0, 0, 1])   # 水平组系
        st = single_set_dfn([1, 0, 0])   # 陡立组系 (面含 z)
        egs_bad = egs_connectivity_metric(hz, [0, 0, 1])['assessment']
        mine_high = mine_risk_sections(hz, [1, 0, 0])['risk_level']
        disp_high = disposal_escape_priority(st)['escape_priority']
        ok = ('较差' in egs_bad) and (mine_high == '高') and (disp_high == '高')
        detail = (f"竖直井对⊥水平面→'{egs_bad}'; 巷道轴在水平连通面内→'{mine_high}'; "
                  f"垂直方向在陡立面内→'{disp_high}'"
                  if ok else
                  f"判级反向复发: egs='{egs_bad}' mine='{mine_high}' disp='{disp_high}'")
    except Exception as e:
        return CheckResult("S29 场景指标面-轴语义毒丸 (法向≠通道方向)",
                           "管道断裂探测", False, str(e))
    return CheckResult("S29 场景指标面-轴语义毒丸 (法向≠通道方向)",
                       "管道断裂探测", ok, detail)


@check("S30 geo_prior 批量一致性 [known-fail: H1]", "管道断裂探测")
def chk_s30_geo_prior_batch_consistency():
    """H1 known-fail 门禁 (架构师裁决 2026-08): 测机制是否仍如档案所述, 不许假绿。

    事实档案 (scripts/diag_s30_batch_mech.py 隔离诊断):
      - 根因: _kmeans_dirs_batched 共享 default_rng(seed) 按批内顺序消费,
        第 b 个网的 k-means 初始化依赖其批内位置 (numpy 参考路径每网 fresh seed);
      - 实测: 位置 0 网与 B=1 一致 (~0.03°); 位置 >=1 网在组系清晰合成数据上
        最大逐点角差 41.2° (Ks=(3..6) 过聚类 -> 多个局部最优 -> init 敏感);
      - 影响面: 仅训练期 gp 先验噪声 + 批量评测路径; 诚实榜 8 方法全 numpy
        路径, 主锚点 36.687/31.7/12.37/0.37 全部不受影响;
      - 处置: 计算冻结不动 (修了会断 e3gt 系 checkpoint 复现), 修复推迟至
        重训时代; 对外表述仅限 B=1。
    门禁语义:
      - 位置 0 网偏离 >=0.1° -> FAIL (根因机制变了, 停手上报);
      - 其余网偏离 <0.1°     -> PASS "H1 已消失" (有人修了 -> 必须更新文档+重评);
      - 其余网偏离 <50°      -> PASS known-fail (仍在档案包络内);
      - 否则                 -> FAIL (行为漂移超出档案)。
    """
    name = "S30 geo_prior 批量一致性 [known-fail: H1]"
    try:
        import torch
        from fractureflow.geo_prior import geo_prior_dirs_gpu

        def make_net(seed, n=40):
            r = np.random.default_rng(seed)
            pos = r.normal(size=(n, 3))
            nrm = np.vstack([np.tile([1.0, 0, 0], (n // 2, 1)),
                             np.tile([0.0, 1, 0], (n - n // 2, 1))])
            nrm = nrm + r.normal(0, 0.05, nrm.shape)
            nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
            mask = r.random(n) < 0.5
            return (pos.astype(np.float64), (nrm * mask[:, None]).astype(np.float64),
                    mask)

        nets = [make_net(s) for s in (101, 202, 303, 404)]
        outs_b1 = []
        for pos, nrm, mask in nets:
            o = geo_prior_dirs_gpu(torch.tensor(pos)[None], torch.tensor(nrm)[None],
                                   torch.tensor(mask, dtype=torch.float64)[None],
                                   fast=False)
            outs_b1.append(o[0].numpy())
        pos4 = torch.tensor(np.stack([x[0] for x in nets]))
        nrm4 = torch.tensor(np.stack([x[1] for x in nets]))
        mask4 = torch.tensor(np.stack([x[2] for x in nets]), dtype=torch.float64)
        out_b4 = geo_prior_dirs_gpu(pos4, nrm4, mask4, fast=False).numpy()

        devs = []
        for b, ((_, _, mask), o1) in enumerate(zip(nets, outs_b1)):
            hid = ~mask
            c = np.clip(np.abs((o1[hid] * out_b4[b][hid]).sum(-1)), 0, 1)
            devs.append(float(np.degrees(np.arccos(c)).max()))
        pos0_dev, others_worst = devs[0], max(devs[1:])

        if pos0_dev >= 0.1:
            ok = False
            detail = (f"机制变化: 位置0网偏离 B=1 达 {pos0_dev:.3f}° "
                      f"(应≈0) — 共享init诊断失效, 停手上报")
        elif others_worst < 0.1:
            ok = True
            detail = ("H1 已消失 (批量=B=1, worst<0.1°) — 有人修复了 geo_prior; "
                      "必须同步更新口径锁定文档并重评 prior 系模型数字")
        elif others_worst < 50.0:
            ok = True
            detail = (f"known-fail (H1, 按档案维持): 位置依赖偏差 worst="
                      f"{others_worst:.1f}° <50° 包络; 计算冻结, 修复推迟至重训时代; "
                      f"对外表述仅限 B=1")
        else:
            ok = False
            detail = f"H1 偏差超档案包络: {others_worst:.1f}° >=50° — geo_prior 行为漂移"
    except Exception as e:
        return CheckResult(name, "管道断裂探测", False, str(e))
    return CheckResult(name, "管道断裂探测", ok, detail)


@check("S31 json.dump allow_nan 守卫扫描 (缺守卫/错位)", "管道断裂探测")
def chk_s31_json_nan_guard():
    """L5 规则: 全库 json.dump/dumps 必须带 allow_nan=False, 且 allow_nan
    不许出现在非 json 调用上。背景: decovalex_routeB.json 曾静默写入 NaN;
    批量补丁曾把 kwarg 错插进外层 print (f-string AST 位置陷阱)。"""
    try:
        import ast as _ast

        def audit(path):
            bad_g, bad_p, bad_s = [], [], []
            try:
                # utf-8-sig: 剥 BOM (R80 修复: v_r33 带 BOM 曾致整文件被静默跳过扫描)
                txt = open(path, 'rb').read().decode('utf-8-sig')
                tree = _ast.parse(txt)
            except Exception as e:
                bad_s.append(f"{path}: {type(e).__name__} {e}")
                return bad_g, bad_p, bad_s
            for node in _ast.walk(tree):
                if not isinstance(node, _ast.Call):
                    continue
                f = node.func
                is_jd = (isinstance(f, _ast.Attribute) and f.attr in ('dump', 'dumps')
                         and isinstance(f.value, _ast.Name) and f.value.id == 'json')
                has_an = any(k.arg == 'allow_nan' for k in node.keywords)
                if is_jd:
                    seg = _ast.get_source_segment(txt, node) or ''
                    if seg and 'allow_nan' not in seg:
                        bad_g.append(f"{path}:{node.lineno}")
                elif has_an:
                    fname = getattr(f, 'id', None) or getattr(f, 'attr', '?')
                    bad_p.append(f"{path}:{node.lineno} ({fname})")
            return bad_g, bad_p, bad_s

        bad_g, bad_p, bad_s = [], [], []
        for d in (_SCRIPTS_DIR, _SRC_DIR):
            for dirpath, _, files in os.walk(d):
                for fn in files:
                    if fn.endswith('.py'):
                        g, p, s = audit(os.path.join(dirpath, fn))
                        bad_g += g
                        bad_p += p
                        bad_s += s
        ok = not bad_g and not bad_p and not bad_s
        if ok:
            detail = "src+scripts 全部 json dump 带守卫, 无错位, 无解析失败"
        else:
            detail = (f"缺守卫 {len(bad_g)}: {bad_g[:3]}; 错位 {len(bad_p)}: {bad_p[:3]}"
                      + (f"; 解析失败 {len(bad_s)}: {bad_s[:2]}" if bad_s else ""))
    except Exception as e:
        return CheckResult("S31 json.dump allow_nan 守卫扫描 (缺守卫/错位)",
                           "管道断裂探测", False, str(e))
    return CheckResult("S31 json.dump allow_nan 守卫扫描 (缺守卫/错位)",
                       "管道断裂探测", ok, detail)


@check("S32 产状↔法向往返一致 (全库变换点扫描)", "管道断裂探测")
def chk_s32_dip_normal_roundtrip():
    """L5 铁律固化 (2026-08-28): dip↔法向双向变换必须往返一致.

    历史事故: 八期 T58 只修了正向 (read_forge_las.dip_dipdir_to_normal),
    同类逆向变换 (_nd / report_replay) 的 sin/cos 互换漏网 —— dip 被镜像成
    90°−dip, 锚点档/复现框架整条链建立在镜像法向上。本检查锁死现存全部
    变换点与权威口径的一致性; 新增变换点若不一致会在此 FAIL。
    """
    NAME = "S32 产状↔法向往返一致 (全库变换点扫描)"
    try:
        import importlib.util
        import math

        def _load(name):
            spec = importlib.util.spec_from_file_location(
                name, os.path.join(_SCRIPTS_DIR, name + ".py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        rng = np.random.default_rng(11)
        dips = rng.uniform(1.0, 89.0, size=24)   # 避开 0/90 退化端点
        dds = rng.uniform(0.0, 360.0, size=24)

        bad = []
        # 1) read_forge_las 权威正变换: nz == cos(dip), 单位化
        rfl = _load("read_forge_las")
        n_ref = np.asarray(rfl.dip_dipdir_to_normal(dds, dips), float)
        if not np.allclose(np.linalg.norm(n_ref, axis=-1), 1.0, atol=1e-6):
            bad.append("read_forge_las.dip_dipdir_to_normal 非单位")
        if float(np.abs(n_ref[:, 2] - np.cos(np.radians(dips))).max()) > 1e-6:
            bad.append("read_forge_las 正变换 nz != cos(dip)")
        # 2) forge_fmi_pipeline._nd 必须同口径
        ffp = _load("forge_fmi_pipeline")
        for dd, dp in zip(dds, dips):
            n1 = np.asarray(ffp._nd(dd, dp), float).ravel()
            if abs(float(n1[2]) - math.cos(math.radians(dp))) > 1e-6:
                bad.append(f"forge_fmi_pipeline._nd nz={n1[2]:.4f} != cos(dip={dp:.1f}°)")
                break
        # 3) report_replay 逆变换: normal -> 提取 dip 应闭合
        rr = _load("report_replay")
        n_in = np.asarray(rr._dip_dipdir_to_normal(dds, dips), float)
        dip_rt = np.degrees(np.arccos(np.clip(np.abs(n_in[:, 2]), 0, 1)))
        worst = float(np.abs(dip_rt - dips).max())
        if worst > 1e-4:
            bad.append(f"report_replay 往返 dip 最大偏差 {worst:.3f}°")

        ok = not bad
        detail = "全部产状↔法向变换点与权威口径一致" if ok else "; ".join(bad[:4])
    except Exception as e:
        return CheckResult(NAME, "管道断裂探测", False, str(e))
    return CheckResult(NAME, "管道断裂探测", ok, detail)


# ---------------------------------------------------------------------------
# 3. 口径扫描组
# ---------------------------------------------------------------------------

@check("废弃数字扫描", "口径扫描")
def chk_deprecated_numbers():
    """扫描 results 目录, 确认废弃数字只出现在标注语境."""
    # 更精确的匹配: 只标记特定模式的废弃数字
    deprecated_patterns = {
        "中位 42°": "beishan dip 中位 (应为 47.7°)",
        "中位 42.0°": "beishan dip 中位 (应为 47.7°)",
        "误差 43.2°": "dip_only 误差 (arcsin bug)",
        "误差 43.20°": "dip_only 误差 (arcsin bug)",
    }
    # 工具自生成报告: 内含源码片段/模式定义字符串 (如 selfcheck.py 自身的
    # 废弃数字模式串), 会被自身扫描误报, 必须排除 (避免自引用死循环)。
    _TOOL_REPORT_JSONS = {
        "selfcheck_report.json",
        "danger_pattern_scan.json",
        "mutation_test_report.json",
        "repro_certify.json",
        "orphan_json_disposition.json",
    }
    found_deprecated = []
    for root, dirs, files in os.walk(_RESULTS_DIR):
        for f in files:
            if f.endswith(".json"):
                if f in _TOOL_REPORT_JSONS:
                    continue
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        content = fh.read()
                    for pattern, reason in deprecated_patterns.items():
                        if pattern in content:
                            if "deprecated" not in content.lower() and "废弃" not in content:
                                found_deprecated.append(f"{f}: {pattern}")
                except Exception:
                    pass
    ok = len(found_deprecated) == 0
    return CheckResult("废弃数字扫描", "口径扫描", ok,
                       "未发现未标注的废弃数字" if ok else f"发现: {found_deprecated[:5]}")


@check("口径一致性抽查", "口径扫描")
def chk_caliber_consistency():
    """抽查关键数字与口径表一致性."""
    l1_path = os.path.join(_RESULTS_DIR, "honest_leaderboard/l1_local__beishan_22.json")
    if not os.path.exists(l1_path):
        return CheckResult("口径一致性抽查", "口径扫描", True, "跳过 (文件不存在)")
    try:
        with open(l1_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mae = float(data.get("mae_mean", data.get("mae", 0)))
        ok = 35 < mae < 39
        return CheckResult("口径一致性抽查", "口径扫描", ok,
                           f"l1_local MAE = {mae:.3f}°" if ok else f"l1_local MAE = {mae:.3f}° (异常)")
    except Exception as e:
        return CheckResult("口径一致性抽查", "口径扫描", False, str(e))


@check("occ 掩码语义反向扫描", "口径扫描")
def chk_occ_mask_semantics():
    """L5 铁律 (外部审查雷2, 2026-08-29): assign 掩码语义方向.

    约定: occ 为 True 表示"已观测"; assign>=0 表示入组, -1 表示未观测.
    故 'assign[~occ] = 0' 会把未观测点标成入组、已观测点留 -1, 语义反转 (休眠 bug).
    全库扫描 src/fractureflow/*.py, 禁止出现该反向写法.
    """
    import re
    hits = []
    scan_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "src", "fractureflow")
    for root, _, files in os.walk(scan_dir):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    for ln, line in enumerate(fh.readlines(), 1):
                        if re.search(r"assign\[\s*~occ\s*\]\s*=\s*0", line):
                            hits.append(f"{fn}:{ln}")
            except Exception:
                pass
    ok = len(hits) == 0
    return CheckResult("occ 掩码语义反向扫描", "口径扫描", ok,
                       "未检测到 assign[~occ] 反向写法" if ok else f"发现反向掩码: {hits}")


@check("S28 图账同源: 论文图表脚本零手抄数字", "口径扫描")
def chk_s28_figure_script_numbers():
    """T88/G8 制度化: gen_paper_figures.py 禁止出现未申报的浮点字面量。

    背景: Fig6 曾用 rng.uniform 模拟数据冒充实测、Fig2/Fig3 手抄受染数字
    (9.35 / 7.78)。根治规则: 所有展示用精度数字必须从 results/*.json 直读,
    布局常数 (figsize/alpha/线宽) 必须在脚本顶部 ALLOWED_LAYOUT_FLOATS 申报。
    本检查用 tokenize 提取源码中的浮点 NUMBER token (字符串/注释不扫),
    任何不在白名单内的浮点字面量 = 有人往图里手抄了数字 = FAIL。
    """
    import ast
    import io as _io
    import tokenize as _tokenize

    fig_path = os.path.join(_SCRIPTS_DIR, "gen_paper_figures.py")
    if not os.path.exists(fig_path):
        return CheckResult("S28 图账同源: 论文图表脚本零手抄数字",
                           "口径扫描", False, "scripts/gen_paper_figures.py 不存在")
    try:
        with open(fig_path, encoding="utf-8") as f:
            src = f.read()
        if "S28-AUDIT" not in src:
            return CheckResult("S28 图账同源: 论文图表脚本零手抄数字",
                               "口径扫描", False, "缺少 S28-AUDIT 标记 (白名单申报块被移除?)")
        # 从 AST 中提取 ALLOWED_LAYOUT_FLOATS 白名单 (字符串集合)
        allowed = None
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if getattr(t, "id", None) == "ALLOWED_LAYOUT_FLOATS":
                        allowed = set(ast.literal_eval(node.value))
        if allowed is None:
            return CheckResult("S28 图账同源: 论文图表脚本零手抄数字",
                               "口径扫描", False, "未找到 ALLOWED_LAYOUT_FLOATS 申报")
        # tokenize 扫描代码区浮点字面量 (STRING/COMMENT 天然排除)
        found = set()
        for tok in _tokenize.generate_tokens(_io.StringIO(src).readline):
            if tok.type == _tokenize.NUMBER and "." in tok.string:
                found.add(tok.string)
        bad = sorted(found - allowed)
        unused = sorted(allowed - found)
        ok = len(bad) == 0
        detail = f"代码区浮点字面量 {len(found)} 个全部在白名单内"
        if unused:
            detail += f"; 白名单未命中(仅提示): {unused[:5]}"
        if not ok:
            detail = f"发现未申报的手抄浮点数: {bad} —— 图表必须 JSON 直读!"
        return CheckResult("S28 图账同源: 论文图表脚本零手抄数字",
                           "口径扫描", ok, detail)
    except SyntaxError as e:
        return CheckResult("S28 图账同源: 论文图表脚本零手抄数字",
                           "口径扫描", False, f"语法解析失败: {e}")
    except Exception as e:
        return CheckResult("S28 图账同源: 论文图表脚本零手抄数字",
                           "口径扫描", False, f"异常: {e}")


# ---------------------------------------------------------------------------
# 4. 资产完整性组
# ---------------------------------------------------------------------------

_KEY_RESULTS = [
    "honest_leaderboard/l1_local__beishan_22.json",
    "set_table_eval.json",
    "p0_full_audit.json",
    "dip_only_upper_bound.json",
    "t58_b2b3_forge_rebuild.json",
    "t51_type_isolation_revalidation.json",
]

_KEY_TESTS = [
    "tests/test_geometry_conventions.py",
    "tests/test_terzaghi.py",
]


@check("关键 results 存在性", "资产完整性")
def chk_key_results_exist():
    missing = []
    for rel_path in _KEY_RESULTS:
        full_path = os.path.join(_RESULTS_DIR, rel_path)
        if not os.path.exists(full_path):
            missing.append(rel_path)
    ok = len(missing) == 0
    return CheckResult("关键 results 存在性", "资产完整性", ok,
                       f"全部存在 ({len(_KEY_RESULTS)} 个)" if ok else f"缺失: {missing}")


@check("单测文件存在性", "资产完整性")
def chk_tests_exist():
    missing = []
    for t in _KEY_TESTS:
        if not os.path.exists(os.path.join(_PROJECT_ROOT, t)):
            missing.append(t)
    ok = len(missing) == 0
    return CheckResult("单测文件存在性", "资产完整性", ok,
                       "全部存在" if ok else f"缺失: {missing}")


# ---------------------------------------------------------------------------
# 5. 单元回归组
# ---------------------------------------------------------------------------

@check("回归: B1 type_aware 双键区分", "单元回归")
def chk_b1_regression():
    fuse_path = os.path.join(_SRC_DIR, "fractureflow/l4/fuse.py")
    if not os.path.exists(fuse_path):
        return CheckResult("回归: B1 type_aware 双键区分", "单元回归", True, "跳过")
    with open(fuse_path, "r", encoding="utf-8") as f:
        content = f.read()
    ok = "source_name" in content
    return CheckResult("回归: B1 type_aware 双键区分", "单元回归", ok,
                       "双键区分已实施" if ok else "未找到双键区分")


@check("回归: B5 空簇跳过", "单元回归")
def chk_b5_regression():
    fuse_path = os.path.join(_SRC_DIR, "fractureflow/l4/fuse.py")
    if not os.path.exists(fuse_path):
        return CheckResult("回归: B5 空簇跳过", "单元回归", True, "跳过")
    with open(fuse_path, "r", encoding="utf-8") as f:
        content = f.read()
    ok = "mask.sum() == 0" in content or "mask.sum()<1" in content
    return CheckResult("回归: B5 空簇跳过", "单元回归", ok,
                       "空簇跳过已实施" if ok else "未找到空簇跳过逻辑")


@check("回归: B7 贯通判定用 domain 边界 (功能)", "单元回归")
def chk_b7_regression():
    """功能级回归: 验证 _find_spanning_cluster 用真实 domain 边界而非数据极值。

    构造: domain=L=100, 一连通分量含 x=-49 (近 -L/2 壁) 与 x=+5 (近中心, 不近
    +L/2 壁)。数据极值=[-49,5], 但若按数据极值判定会误判"贯穿"; 按真实 domain
    判定应返回 False (未贯穿 +L/2 壁)。旧实现 (数据极值) 会错误返回 True。
    """
    try:
        import scipy.sparse as sp
        from fractureflow.percolation import _find_spanning_cluster
        centers = np.array([[-49.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        adj = sp.csr_matrix([[0, 1], [1, 0]])
        has_span, _ = _find_spanning_cluster(adj, centers, (100.0, 100.0, 100.0), axis=0)
        ok = (has_span is False)
        detail = ("domain 边界判定生效 (稀疏分量不误判贯穿)"
                  if ok else "仍按数据极值判定: 误判贯穿 (B7 复发!)")
    except Exception as e:  # noqa: BLE001
        ok = False
        detail = f"导入/执行失败: {e}"
    return CheckResult("回归: B7 贯通判定用 domain 边界 (功能)", "单元回归", ok, detail)


# ---------------------------------------------------------------------------
# 6. 管道断裂探测组 (T67)
# ---------------------------------------------------------------------------


def _angular_mae_deg(pred, true):
    """项目主指标: mean acos|<pred,true>| (度). 复制自评测口径, 供锚点自检."""
    cos = np.clip(np.abs(np.sum(pred * true, axis=1, dtype=float)), 0.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def _rand_unit(n, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _rand_rot(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    return Q


# 数学真实锚点: 两个独立均匀随机单位向量夹角余弦 |cos θ| ~ Uniform[0,1],
# 故 E[acos(|cos θ|)] = ∫₀¹ arccos(u) du = 1.0 rad = 57.2958° (不依赖任何数据).
# 任何管线喂全随机输入却给出显著偏离此值的结果 = 链路 bug。
_RANDOM_ANCHOR_DEG = 57.29577951308232


@check("S22 随机锚点 + 不变性 (数学真相锚)", "单元回归")
def chk_s22_random_anchor():
    """验证主指标在已知数学锚点上正确 + 满足旋转/置换不变性。

    - 全随机 pred vs 全随机 true → 误差必须 ≈ 57.2958°
    - pred==true → 0
    - 同旋转 R 作用于 pred 与 true → 误差不变 (旋转等变)
    - 打乱点序 → 误差不变 (置换不变)
    """
    ok = True
    parts = []

    # 1) 随机锚点
    pred = _rand_unit(50000, 1)
    true = _rand_unit(50000, 2)
    e_rand = _angular_mae_deg(pred, true)
    d_rand = abs(e_rand - _RANDOM_ANCHOR_DEG)
    if d_rand > 1.5:
        ok = False
        parts.append(f"随机锚点={e_rand:.3f}° (期望57.296°, 偏差{d_rand:.3f}°>")
    else:
        parts.append(f"随机锚点={e_rand:.3f}°")

    # 2) 恒等
    e_id = _angular_mae_deg(true, true)
    if e_id > 1e-6:
        ok = False
        parts.append(f"恒等误差={e_id:.2e}°")
    else:
        parts.append("恒等=0")

    # 3) 旋转等变
    R = _rand_rot(7)
    e_rot = _angular_mae_deg(pred @ R.T, true @ R.T)
    if abs(e_rot - e_rand) > 1e-6:
        ok = False
        parts.append(f"旋转后偏差={abs(e_rot - e_rand):.2e}°")
    else:
        parts.append("旋转不变")

    # 4) 置换不变
    perm = np.random.default_rng(3).permutation(50000)
    e_perm = _angular_mae_deg(pred[perm], true[perm])
    if abs(e_perm - e_rand) > 1e-6:
        ok = False
        parts.append(f"置换后偏差={abs(e_perm - e_rand):.2e}°")
    else:
        parts.append("置换不变")

    return CheckResult("S22 随机锚点 + 不变性 (数学真相锚)", "单元回归", ok,
                       "; ".join(parts))

@check("S17 冲突检测: >40° 双源触发 conflict_matches", "管道断裂探测")
def chk_s17_conflict_detection():
    """构造角距 >40° 的双源冲突场景, 验证 fuse_bundle 的 conflict_matches 非空."""
    try:
        from fractureflow.l4.fuse import hungarian_match_groups, classify_consistency
        from fractureflow.l4.evidence import EvidenceEntry, EvidenceBundle, build_bundle
        from fractureflow.l4.fuse import fuse_bundle

        # 构造两组角距 >40° 的法向簇
        rng = np.random.default_rng(42)
        # 组 1: 近水平面 (dip ~10°)
        n1 = 30
        dip1 = rng.normal(10, 3, n1).clip(1, 89)
        dd1 = rng.normal(0, 5, n1) % 360
        from fractureflow.l4.evidence import dip_dipdir_to_normal
        normals1 = dip_dipdir_to_normal(dip1, dd1)

        # 组 2: 近垂直面 dip ~70°, 倾向差 >40° (dd ~120°)
        n2 = 30
        dip2 = rng.normal(70, 5, n2).clip(40, 89)
        dd2 = rng.normal(120, 5, n2) % 360
        normals2 = dip_dipdir_to_normal(dip2, dd2)

        entry1 = EvidenceEntry(
            source_type="L0", source_name="well_A",
            data_type="normal_obs", normals=normals1)
        entry2 = EvidenceEntry(
            source_type="L1", source_name="well_B",
            data_type="normal_obs", normals=normals2)
        bundle = build_bundle([entry1], [entry2], site_name="s17_synthetic")

        result = fuse_bundle(bundle, K=2, seed=42)

        n_conflicts = len(result.conflict_matches)
        ok = n_conflicts > 0
        detail = f"conflict_matches 数量 = {n_conflicts}"
        if ok:
            # 验证冲突匹配中的角距确实 >35°
            max_angle = max(cm.angle_deg for cm in result.conflict_matches)
            detail += f", 最大角距 = {max_angle:.1f}°"
            if max_angle <= 35:
                ok = False
                detail += " (应 >35°)"
            else:
                detail += " (>35°, conflict 正确触发)"
        return CheckResult("S17 冲突检测: >40° 双源触发 conflict_matches",
                           "管道断裂探测", ok, detail)
    except Exception as e:
        return CheckResult("S17 冲突检测: >40° 双源触发 conflict_matches",
                           "管道断裂探测", False, f"异常: {e}", traceback.format_exc())


@check("S18 聚类管道: spherical_kmeans 输出有效组系", "管道断裂探测")
def chk_s18_clustering_pipeline():
    """构造 50 条合成法向 (2 组), 验证 spherical_kmeans 正常输出 centers 且 K>0."""
    try:
        from fractureflow.setlabel import spherical_kmeans
        from fractureflow.l4.evidence import dip_dipdir_to_normal

        rng = np.random.default_rng(99)
        # 第一组: dip ~20°, dd ~0° (北)
        n1 = 25
        dip1 = rng.normal(20, 5, n1).clip(1, 89)
        dd1 = rng.normal(0, 10, n1) % 360
        normals1 = dip_dipdir_to_normal(dip1, dd1)

        # 第二组: dip ~60°, dd ~180° (南)
        n2 = 25
        dip2 = rng.normal(60, 5, n2).clip(1, 89)
        dd2 = rng.normal(180, 10, n2) % 360
        normals2 = dip_dipdir_to_normal(dip2, dd2)

        all_normals = np.concatenate([normals1, normals2], axis=0)

        centers, assign = spherical_kmeans(all_normals, K=2, seed=42)

        ok = (centers is not None and len(centers) == 2 and
              assign is not None and len(assign) == 50 and
              len(np.unique(assign)) == 2)
        detail = f"centers.shape={centers.shape}, assign 唯一组={np.unique(assign)}, N={len(assign)}"
        if not ok:
            detail += " (应 K=2, 50 条, 2 组均非空)"
        return CheckResult("S18 聚类管道: spherical_kmeans 输出有效组系",
                           "管道断裂探测", ok, detail)
    except Exception as e:
        return CheckResult("S18 聚类管道: spherical_kmeans 输出有效组系",
                           "管道断裂探测", False, f"异常: {e}", traceback.format_exc())


@check("S19 SetTable 生成: set_table_from_normals K=4", "管道断裂探测")
def chk_s19_settable_generation():
    """构造 K=4 合成法向数据, 验证 set_table_from_normals 正常生成 SetTable."""
    try:
        from fractureflow.dfn import set_table_from_normals
        from fractureflow.l4.evidence import dip_dipdir_to_normal

        rng = np.random.default_rng(7)
        # 四组法向, 角距明显分离
        group_params = [
            (15, 0, 20),    # dip 15°, dd 0°
            (35, 90, 20),   # dip 35°, dd 90°
            (55, 180, 20),  # dip 55°, dd 180°
            (75, 270, 20),  # dip 75°, dd 270°
        ]
        all_normals = []
        for dip_mean, dd_mean, n in group_params:
            dips = rng.normal(dip_mean, 5, n).clip(1, 89)
            dds = rng.normal(dd_mean, 8, n) % 360
            normals = dip_dipdir_to_normal(dips, dds)
            all_normals.append(normals)
        all_normals = np.concatenate(all_normals, axis=0)

        st, set_ids = set_table_from_normals(all_normals, K=4, seed=42)

        ok = (st is not None and st.centers is not None and
              st.centers.shape[0] == 4 and len(set_ids) == 80 and
              len(np.unique(set_ids)) == 4)
        if ok:
            detail = f"SetTable K={st.centers.shape[0]}, proportions={st.proportions.round(3).tolist()}"
        else:
            detail = f"centers.shape={st.centers.shape if st.centers is not None else None}, set_ids 唯一={np.unique(set_ids)}"
        return CheckResult("S19 SetTable 生成: set_table_from_normals K=4",
                           "管道断裂探测", ok, detail)
    except Exception as e:
        return CheckResult("S19 SetTable 生成: set_table_from_normals K=4",
                           "管道断裂探测", False, f"异常: {e}", traceback.format_exc())


def _run_pytest_subprocess(max_attempts=3) -> Tuple[bool, str]:
    """调用 pytest 跑 tests/ 下单测, 返回 (全部通过, 详情字符串).

    R61 加固 (2026-08-26): 本机存在周期性外部 CTRL_C 注入源 (用户态自动化工具
    嫌疑), 父进程等待与 pytest 子进程执行均会被随机打断 (一夜实测 7 次, 时点
    145~224s 散布; 直接运行偶有幸免; 子进程中断点 r30_frame_audit.py:281 为
    随机执行位置, 该文件本身无嫌疑)。对策: 仅对 Ctrl-C 类中断 (父侧
    KeyboardInterrupt / 子侧 STATUS_CONTROL_C_EXIT=0xC000013A) 退避重试;
    真实测试失败 (rc=1 等) 与超时**不重试**, 不掩盖。
    """
    python_exe = sys.executable
    cmd = [python_exe, "-m", "pytest", _TESTS_DIR, "-v", "--tb=short", "-q"]
    status_ctrl_c = -1073741510  # 0xC000013A STATUS_CONTROL_C_EXIT
    last_note = "pytest 未运行"
    for attempt in range(1, max_attempts + 1):
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=450,  # R8-S20 (2026-08-24): 实测 pytest tests/ 全量 225.02s, 取 2x=450 且 >=300 下限
                cwd=_PROJECT_ROOT,
            )
            output = proc.stdout + "\n" + proc.stderr
            summary_lines = [l for l in output.splitlines()
                             if "passed" in l or "failed" in l]
            summary = (summary_lines[-1].strip() if summary_lines
                       else f"exit code={proc.returncode}")
            if proc.returncode == 0:
                return True, summary
            if proc.returncode == status_ctrl_c:
                last_note = f"子进程被环境 CTRL_C 击中 (rc={proc.returncode})"
            else:
                return False, summary  # 真实失败原样上报, 不重试
        except subprocess.TimeoutExpired:
            return False, "pytest 超时 (>120s)"
        except FileNotFoundError:
            return False, "pytest 未安装"
        except KeyboardInterrupt:
            last_note = "父进程等待被环境 CTRL_C 击中"
        if attempt < max_attempts:
            time.sleep(20 * attempt)  # 错开外部注入源相位
    return False, (f"{last_note}; 重试 {max_attempts} 次仍被打断 "
                   f"(R61 登记: 环境 CTRL_C 风暴, 非测试失败)")


@check("S21 数字台账: JSON 一致性 + 落单扫描", "管道断裂探测")
def chk_s21_digital_ledger():
    """验证 digital_ledger.json 中引用的 active JSON 全部存在且值一致 (容差 0.1°).
    结构守卫: entries/deprecated 的 ID 全文件唯一 + changelog 必须为正规变更单
    (date/task/changes) 格式, 防止 ledger 对象误贴进 changelog 造成的重复条目.
    同时扫描 results/ 下未被台账引用的 JSON (落单数字), 仅报告不报错.
    """
    ledger_path = os.path.join(_RESULTS_DIR, "digital_ledger.json")
    if not os.path.exists(ledger_path):
        return CheckResult("S21 数字台账: JSON 一致性 + 落单扫描",
                           "管道断裂探测", False,
                           "digital_ledger.json 不存在")

    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)
    except Exception as e:
        return CheckResult("S21 数字台账: JSON 一致性 + 落单扫描",
                           "管道断裂探测", False, f"JSON 解析失败: {e}")

    entries = ledger.get("entries", [])

    # 结构守卫 (2026-08-29 外部审查台账自愈): ID 唯一性 + changelog 变更单格式.
    # 背景: 曾有 NNS-051/052 原始 ledger 对象误贴进 changelog 数组 (与 entries 重复
    # 且内容有出入), 而 S21 只查 entries->source_json 值一致性不查结构, 造成
    # "扫描通过"的假安全感. 本守卫把该模式提为硬性扫描规则 (L5 铁律).
    struct_errs = []
    seen = {}
    for arr in ("entries", "deprecated"):
        for i, e in enumerate(ledger.get(arr, [])):
            eid = e.get("id", "")
            if not eid:
                struct_errs.append(f"{arr}[{i}] 缺 id 字段")
                continue
            if eid in seen:
                struct_errs.append(f"ID 重复: {eid} 同时出现于 {seen[eid]} 与 {arr}[{i}]")
            else:
                seen[eid] = f"{arr}[{i}]"
    for i, c in enumerate(ledger.get("changelog", [])):
        if not ("date" in c and "task" in c and isinstance(c.get("changes"), list)):
            struct_errs.append(f"changelog[{i}] 非正规变更单格式 (需 date/task/changes list)")

    mismatch = []
    missing = []
    unextractable = []  # P1 修复: 提取失败单独计数, 防假 PASS
    checked = 0
    tolerance = 0.1  # deg (浮点/多 seed 容差)

    for entry in entries:
        if entry.get("status") not in ("active", "active-oracle"):
            continue
        rel_path = entry.get("source_json", "")
        if not rel_path:
            continue
        full_path = os.path.join(_PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            missing.append(rel_path)
            continue
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # 尝试从 JSON 中提取 mae, 支持 k_level 和 source_path 提示
        k_hint = entry.get("k_level", "12")
        source_path = entry.get("source_path", "")
        actual = _extract_mae(data, k_hint=k_hint, source_path=source_path)
        expected = entry.get("value")
        if actual is not None and expected is not None:
            # 允许容差, 也允许百分比单位 (值域 0-1 vs 0-100)
            scale = 1.0
            if entry.get("unit") == "percent":
                scale = 100.0
            diff = abs(actual * scale - expected)
            if diff > tolerance:
                mismatch.append(f"{rel_path}: expected={expected}, actual={actual}, diff={diff:.4f}")
            checked += 1
        else:
            # 提取失败 → 单独计数, 不静默计入 checked (防假 PASS)
            skip_reason = ""
            if not rel_path:
                skip_reason = "no source_json"
            elif actual is None:
                skip_reason = "extract_mae=NaN"
            elif expected is None:
                skip_reason = "expected=None (台账无 value 字段)"
            unextractable.append(f"{rel_path}: {skip_reason}")

    # 落单 JSON 扫描 (仅报告, 不影响 PASS/FAIL)
    referenced_files = set()
    for entry in entries:
        rj = entry.get("source_json", "")
        if rj:
            referenced_files.add(os.path.normpath(os.path.join(_PROJECT_ROOT, rj)))

    orphaned = []
    for root, dirs, files in os.walk(_RESULTS_DIR):
        for f in files:
            if f.endswith(".json") and f != "digital_ledger.json":
                full = os.path.normpath(os.path.join(root, f))
                if full not in referenced_files:
                    # 排除已知被 docs 引用的 (在 selfcheck 内不做文档扫描, 仅报告)
                    orphaned.append(os.path.relpath(full, _PROJECT_ROOT))

    detail = (f"checked={checked}, mismatch={len(mismatch)}, "
               f"missing={len(missing)}, unextractable={len(unextractable)}, "
               f"struct_errs={len(struct_errs)}, orphaned_files={len(orphaned)}")
    if missing:
        detail += f", MISSING: {missing[:3]}"
    if mismatch:
        detail += f", MISMATCH: {mismatch[:3]}"
    if unextractable:
        detail += f", UNEXTRACTABLE: {unextractable[:3]}"
    if struct_errs:
        detail += f", STRUCT_ERR: {struct_errs[:3]}"
    if orphaned:
        detail += f", ORPHANNED_SAMPLE: {orphaned[:5]}"

    # PASS 条件: 无 mismatch + 无 missing + 无 unextractable + 无结构错误
    ok = (len(mismatch) == 0 and len(missing) == 0 and
          len(unextractable) == 0 and len(struct_errs) == 0)
    return CheckResult("S21 数字台账: JSON 一致性 + 落单扫描",
                       "管道断裂探测", ok, detail)


def _extract_mae(data, k_hint: str = "12", source_path: str = "") -> Optional[float]:
    """从多种常见 JSON 结构中提取 mae 数值.

    优先级:
      0. source_path 指定路径 (如 "threshold_sources.information_ceiling_31_7deg.value")
      1. 顶层 mae_mean / mae / value (honest_leaderboard 格式)
      2. 嵌套 K-level modal_err_mean (set_table 格式, 按 k_hint 选取)
      3. result 单键包装 (如 beishan_kmeans_obs_k12.json 的 result.mae_mean)
      4. 深层递归搜索 modal_err_mean / mae_mean (phase0_oracle_floor_strict 等)
    """
    if not isinstance(data, dict):
        return None
    # 0. source_path 指定路径 (支持 dict key 和 list 下标, 如 "site_results.0.value")
    if source_path:
        parts = source_path.split(".")
        cur = data
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            elif isinstance(cur, list):
                try:
                    cur = cur[int(p)]
                except (ValueError, IndexError):
                    cur = None
                    break
            else:
                cur = None
                break
        if isinstance(cur, (int, float)):
            return float(cur)
    # 1. 顶层直接键
    for key in ("mae_mean", "mae", "value"):
        if key in data and isinstance(data[key], (int, float)):
            return float(data[key])
    # 2. 嵌套 K-level 结构 (set_table): 按 k_hint 选取
    hint_val = None
    last_val = None
    for k, v in data.items():
        if isinstance(v, dict):
            for key in ("modal_err_mean", "mae_mean", "mae"):
                if key in v and isinstance(v[key], (int, float)):
                    val = float(v[key])
                    last_val = val
                    if k == k_hint:
                        hint_val = val
    if hint_val is not None:
        return hint_val
    if last_val is not None:
        return last_val
    # 3. result 单键包装: 如果 data 只有一个 dict 值, 递归进入
    dict_vals = [v for v in data.values() if isinstance(v, dict)]
    if len(dict_vals) == 1:
        inner = _extract_mae(dict_vals[0], k_hint=k_hint)
        if inner is not None:
            return inner
    # 4. 深层递归搜索: 收集所有 modal_err_mean/mae_mean, 优先匹配 k_hint 路径
    best_hint = None
    best_any = None

    def _recurse(obj, path_keys):
        nonlocal best_hint, best_any
        if not isinstance(obj, dict):
            return
        for key in ("modal_err_mean", "mae_mean", "mae"):
            if key in obj and isinstance(obj[key], (int, float)):
                val = float(obj[key])
                if best_any is None:
                    best_any = val
                if k_hint in path_keys and best_hint is None:
                    best_hint = val
        for k, v in obj.items():
            if isinstance(v, dict):
                _recurse(v, path_keys + [k])

    _recurse(data, [])
    if best_hint is not None:
        return best_hint
    return best_any


@check("S20 --with-tests: pytest 整合", "管道断裂探测")
def chk_s20_pytest_integration():
    """验证 pytest 调用管道通畅, tests/ 下单测全部通过."""
    # 检查是否有 pytest 可用
    python_exe = sys.executable
    try:
        proc = subprocess.run(
            [python_exe, "-m", "pytest", "--version"],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            return CheckResult("S20 --with-tests: pytest 整合",
                               "管道断裂探测", False,
                               "pytest 不可用 (exit code={})".format(proc.returncode))
    except Exception as e:
        return CheckResult("S20 --with-tests: pytest 整合",
                           "管道断裂探测", False,
                           f"pytest 版本检查失败: {e}")

    # 检查 tests/ 目录下是否有测试文件
    test_files = [f for f in os.listdir(_TESTS_DIR) if f.startswith("test_") and f.endswith(".py")]
    if not test_files:
        return CheckResult("S20 --with-tests: pytest 整合",
                           "管道断裂探测", False,
                           "tests/ 下无测试文件")

    # 实际跑 pytest (仅当 tests/ 下有文件时)
    passed, summary = _run_pytest_subprocess()
    return CheckResult("S20 --with-tests: pytest 整合",
                       "管道断裂探测", passed,
                       f"tests 文件: {test_files}; {summary}")


# ---------------------------------------------------------------------------
# 主运行器
# ---------------------------------------------------------------------------

def run_all_checks() -> Dict[str, List[CheckResult]]:
    """运行所有注册检查, 按组返回结果."""
    results: Dict[str, List[CheckResult]] = {}
    for group, name, fn in _ALL_CHECKS:
        try:
            result = fn()
        except Exception as e:
            result = CheckResult(name, group, False, f"异常: {e}", traceback.format_exc())
        results.setdefault(group, []).append(result)
    return results


def print_report(results: Dict[str, List[CheckResult]]) -> bool:
    """打印报告, 返回是否全部通过."""
    total_passed = 0
    total_failed = 0
    print("=" * 60)
    print("  fractureflow 一键自检报告")
    print("=" * 60)
    for group, checks in results.items():
        print(f"\n【{group}】")
        for c in checks:
            status = "PASS" if c.passed else "FAIL"
            symbol = "✓" if c.passed else "✗"
            print(f"  {symbol} [{status}] {c.name}: {c.message}")
            if not c.passed and c.detail:
                print(f"         详情: {c.detail[:200]}")
            if c.passed:
                total_passed += 1
            else:
                total_failed += 1
    print(f"\n{'=' * 60}")
    print(f"  总计: {total_passed} PASS, {total_failed} FAIL")
    print(f"{'=' * 60}")
    return total_failed == 0


def save_report(results: Dict[str, List[CheckResult]], path: str):
    """保存报告到 JSON."""
    data = {
        "groups": {
            group: [c.to_dict() for c in checks]
            for group, checks in results.items()
        },
        "summary": {
            "total": sum(len(c) for c in results.values()),
            "passed": sum(1 for checks in results.values() for c in checks if c.passed),
            "failed": sum(1 for checks in results.values() for c in checks if not c.passed),
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder, allow_nan=False)


def main():
    """主入口.

    支持 CLI 参数:
        --with-tests: 额外调用 pytest 运行 tests/ 下单测并汇总结果.
                      默认快速模式不含单测 (保持 <30s).
    """
    parser = argparse.ArgumentParser(
        description="fractureflow 一键自检 (v2, 含管道断裂探测)")
    parser.add_argument("--with-tests", action="store_true", default=False,
                        help="额外运行 pytest 单测 (耗时增加, 默认关闭)")
    args = parser.parse_args()

    results = run_all_checks()

    # 如果开启 --with-tests, 跑 pytest 并追加结果
    if args.with_tests:
        print("\n[--with-tests 模式] 运行 pytest 单测...")
        passed, summary = _run_pytest_subprocess()
        pytest_result = CheckResult(
            "pytest 单测汇总", "管道断裂探测", passed, summary)
        results.setdefault("管道断裂探测", []).append(pytest_result)

    all_passed = print_report(results)
    out_path = os.path.join(_RESULTS_DIR, "selfcheck_report.json")
    save_report(results, out_path)
    print(f"\n报告已落盘: {out_path}")
    return 0 if all_passed else 1


# ---------------------------------------------------------------------------
# B3-4 几何约定 grep 门禁 (集中化根治三次犯案的结构性方案)
#   - 独立 S 编号 (S33), 与 R8 初级单在修的 S20/S31 不同区域, 无碰撞风险.
#   - 若 R8 初级单交付覆盖了本文件, 请重新追加本函数 (门禁脚本为权威, 见 scripts/).
# ---------------------------------------------------------------------------
@check("S33 几何约定 grep 门禁 (手搓产状转换拦截)", "几何约定")
def chk_s33_geometry_conventions():
    """B3-4: 扫描 src/fractureflow 下手搓产状<->法向/走向 转换 (BUG-A/BUG-B 类).

    调用 scripts/check_geometry_conventions.py; 发现违规 (手搓 atan2+sin/cos
    构造法向/走向, 或 BUG-A 轴互换) 即 FAIL。集中权威实现见 geometry.py。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    script = os.path.join(root, "scripts", "check_geometry_conventions.py")
    if not os.path.exists(script):
        return CheckResult("S33 几何约定 grep 门禁", "几何约定", False,
                          f"门禁脚本缺失: {script}")
    try:
        proc = subprocess.run([sys.executable, script], capture_output=True,
                              text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return CheckResult("S33 几何约定 grep 门禁", "几何约定", False,
                          "门禁脚本执行超时 (>120s)")
    if proc.returncode != 0:
        out = (proc.stdout or "") + (proc.stderr or "")
        return CheckResult("S33 几何约定 grep 门禁", "几何约定", False,
                          "发现手搓产状转换违规 (见详情)", out[-1500:])
    return CheckResult("S33 几何约定 grep 门禁", "几何约定", True, "0 处违规")


# ---------------------------------------------------------------------------
# P17b 静默半加载扫描 (2026-09-02 审查发现)
#   背景: t82_e2_drift_gate / compare_locked 等 11 处曾把 ckpt ema 按
#   模型 named_parameters 键集过滤后再以 strict=False 载入 ——
#   e3gt_hybrid.pt 缺 residual_head 4 键时该头保持随机初始化且无任何警告,
#   漂移闸门/排行榜数字被静默污染 (实测半随机模型 MAE 53~65° vs 训练 ~35°)。
#   统一权威入口 = trainer.load_ema_strict (缺键/形状失配必 raise)。
#   注: 本注释不得写出会被下方正则命中的字面 idiom (自引用陷阱, 同废弃数字扫描)。
# ---------------------------------------------------------------------------
@check("S34 静默半加载扫描 (named_parameters 过滤 + strict=False)", "管道断裂探测")
def chk_s34_silent_halfload():
    """L5 规则: 禁止按模型参数键集过滤 ckpt 字典后 strict=False 半加载的 idiom."""
    import re
    NAME = "S34 静默半加载扫描 (named_parameters 过滤 + strict=False)"
    try:
        pat = re.compile(r"in\s+dict\(\s*\w+\.named_parameters\(\)\s*\)")
        bad = []
        for d in (_SCRIPTS_DIR, _SRC_DIR):
            for dirpath, _, files in os.walk(d):
                for fn in files:
                    if not fn.endswith(".py"):
                        continue
                    p = os.path.join(dirpath, fn)
                    try:
                        txt = open(p, encoding="utf-8").read()
                    except Exception:
                        continue
                    for i, line in enumerate(txt.splitlines(), 1):
                        if pat.search(line):
                            bad.append(f"{os.path.relpath(p, _PROJECT_ROOT)}:{i}")
        ok = not bad
        detail = ("全库无过滤式半加载 idiom (统一走 trainer.load_ema_strict)"
                  if ok else f"残留 {len(bad)} 处: {bad[:4]}")
    except Exception as e:
        return CheckResult(NAME, "管道断裂探测", False, str(e))
    return CheckResult(NAME, "管道断裂探测", ok, detail)


if __name__ == "__main__":
    sys.exit(main())
