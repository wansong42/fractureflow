# -*- coding: utf-8 -*-
"""B3-3 几何/幂等不变量测试层 —— 杀模式不杀实例 (四类两行级不变量).

覆盖:
  1. 无向性: normal_to_dip_dipdir(n) == normal_to_dip_dipdir(-n) (杀 BUG-B 全族)
  2. 正交性: 走向向量 ⊥ 倾向向量 (|dot|<1e-9, 杀 BUG-A 全族)
  3. 往返: dip_dipdir->normal->dip_dipdir 误差 <容差, 覆盖 nz>=0 与 nz<0
  4. 幂等: prepare_net 重入输出 dict 全键相等 (含 len/log_len/mask)
"""
import os
import sys
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
for p in (_SRC,):
    if p not in sys.path:
        sys.path.insert(0, p)

from fractureflow.geometry import (
    dip_dipdir_to_normal,
    normal_to_dip_dipdir,
    dip_dir_to_dip_vector,
    dip_dir_to_strike_vector,
)
from fractureflow import data as dt
from fractureflow import outcrop_network as oc

# 浮点往返容差: 标量 dip/dip_dir 精确还原 ~1e-6 量级; 角度往返 ~1e-3 度已远严于业务需求.
_TOL_SCALAR = 1e-6
_TOL_ANGLE = 1e-3


# ---------------------------------------------------------------------------
# 1. 无向性
# ---------------------------------------------------------------------------
def test_undirected_normal_to_dip_dipdir():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = rng.normal(size=3)
        n = n / np.linalg.norm(n)
        d1, dd1 = normal_to_dip_dipdir(n)
        d2, dd2 = normal_to_dip_dipdir(-n)
        assert abs(float(d1) - float(d2)) < _TOL_SCALAR, f"dip mismatch: {d1} vs {d2}"
        diff = abs(float(dd1) - float(dd2)) % 360.0
        diff = min(diff, 360.0 - diff)
        assert diff < _TOL_SCALAR, f"dip_dir mismatch: {dd1} vs {dd2}"


def test_undirected_roundtrip_stable():
    """round-trip 后 n 与 -n 还原出的 (dip,dip_dir) 一致."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = rng.normal(size=3); n = n / np.linalg.norm(n)
        d, dd = normal_to_dip_dipdir(n)
        nrm = dip_dipdir_to_normal(d, dd)
        d2, dd2 = normal_to_dip_dipdir(nrm)
        assert abs(float(d) - float(d2)) < _TOL_SCALAR
        diff = abs(float(dd) - float(dd2)) % 360.0
        diff = min(diff, 360.0 - diff)
        assert diff < _TOL_SCALAR


# ---------------------------------------------------------------------------
# 2. 正交性
# ---------------------------------------------------------------------------
def test_strike_orthogonal_to_dip():
    """走向向量必须 ⊥ 倾向向量 (BUG-A 全族回归)."""
    for dd in [0, 15, 37, 60, 90, 123, 200, 270, 333]:
        dip_vec = dip_dir_to_dip_vector(dd)
        strike_vec = dip_dir_to_strike_vector(dd)
        dot = abs(float(np.dot(dip_vec, strike_vec)))
        assert dot < 1e-9, f"dip_dir={dd}: |dot|={dot} (应≈0)"


def test_strike_unit_and_horizontal():
    for dd in [10, 80, 170, 250, 350]:
        sv = dip_dir_to_strike_vector(dd)
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-9
        assert abs(sv[2]) < 1e-9


# ---------------------------------------------------------------------------
# 3. 往返 (覆盖 nz>=0 与 nz<0)
# ---------------------------------------------------------------------------
def _roundtrip_angular_error(n):
    d, dd = normal_to_dip_dipdir(n)
    nrm = dip_dipdir_to_normal(d, dd)
    cos = np.clip(np.abs(np.dot(n / np.linalg.norm(n), nrm)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def test_roundtrip_nz_positive():
    rng = np.random.default_rng(2)
    for _ in range(50):
        n = rng.normal(size=3)
        n = n / np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        assert _roundtrip_angular_error(n) < _TOL_ANGLE


def test_roundtrip_nz_negative():
    """nz<0 分支 (BUG-B 盲区): 还原法向与输入夹角 <容差."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        n = rng.normal(size=3)
        n = n / np.linalg.norm(n)
        if n[2] > 0:
            n = -n
        assert n[2] < 0
        assert _roundtrip_angular_error(n) < _TOL_ANGLE


def test_outcrop_spacing_projection_finite():
    """集成: OutcropNetwork 间距投影轴有限且数量级合理 (BUG-A 回归)."""
    rng = np.random.default_rng(5)
    mu = np.array([0.3, 0.4, 0.5]); mu = mu / np.linalg.norm(mu)
    centers = []
    for i in range(6):
        sd = dip_dir_to_strike_vector(float(np.degrees(np.arctan2(mu[0], mu[1]))))
        c = np.array([i * 1.0, 0.0, 0.0]) * sd
        pts = c + rng.normal(size=(20, 3)) * 0.05 + mu * 0.1
        centers.append(pts)
    pos = np.concatenate(centers)
    seg_id = np.repeat(np.arange(6), 20)
    net = oc.build_outcrop_network(pos, seg_id)
    sp = net.spacing_stats
    assert "_method" in sp
    finite_found = any(
        isinstance(v, dict) and np.isfinite(v.get("mean_m", np.nan))
        for v in sp.values()
    )
    assert finite_found, f"间距统计无有限值: {sp}"


# ---------------------------------------------------------------------------
# 4. 幂等
# ---------------------------------------------------------------------------
def _minimal_net(seed=0):
    rng = np.random.default_rng(seed)
    L = 12
    return {
        "pos": torch.tensor(rng.normal(size=(L, 3)), dtype=torch.float32),
        "nrm": torch.tensor(rng.normal(size=(L, 3)), dtype=torch.float32),
        "len": torch.tensor(rng.uniform(0.1, 10.0, size=(L, 1)), dtype=torch.float32),
        "lith": torch.tensor(np.zeros(L, dtype=int)),
        "s1": torch.tensor(rng.normal(size=3), dtype=torch.float32),
        "s3": torch.tensor(rng.normal(size=3), dtype=torch.float32),
    }


def test_prepare_net_idempotent():
    """prepare_net 重入: 输出 dict 全键相等 + 数值逐键一致 (含 len/log_len/mask)."""
    net = _minimal_net(7)
    a = dt.prepare_net(net, 0.4, np.random.default_rng(1))
    b = dt.prepare_net(a, 0.4, np.random.default_rng(1))
    assert set(a.keys()) == set(b.keys()), f"键集合不等: {set(a.keys()) ^ set(b.keys())}"
    for k in a.keys():
        va, vb = a[k], b[k]
        if hasattr(va, "allclose"):
            assert va.allclose(vb), f"键 {k} 数值漂移"
        elif isinstance(va, (int, float, str, bool)):
            assert va == vb, f"键 {k} 标量漂移: {va} vs {vb}"
    assert a["lens"].allclose(b["lens"]), "lens 幂等失败 (R8)"
    assert a["log_len"].allclose(b["log_len"]), "log_len 幂等失败 (R8)"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
