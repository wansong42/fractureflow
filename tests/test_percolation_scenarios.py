# -*- coding: utf-8 -*-
"""场景指标语义单测 — 防法向/方向混用 bug 复发 (十四期 P0).

核心几何: connectivity_anisotropy 的 dominant_direction 是优势面**法向**,
判定必须用面-轴夹角 = 90° − 法向-轴夹角。历史 bug: 三个场景函数直接拿
法向套阈值, 判级全部反向 (水平组系+竖直井对误判"对齐良好")。
"""
import json
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fractureflow.dfn import DFNRealization
from fractureflow.percolation import (
    _plane_axis_angle_deg, egs_connectivity_metric, mine_risk_sections,
    disposal_escape_priority)


def _single_set_dfn(normal, seed=7, n=12):
    """单组系 DFN: 法向=normal 的裂隙彼此贴近 => 单一连通分量."""
    rng = np.random.default_rng(seed)
    nrm = np.asarray(normal, float)
    nrm = nrm / np.linalg.norm(nrm)
    u = np.array([1.0, 0.0, 0.0])
    v = np.cross(nrm, u)
    if np.linalg.norm(v) < 1e-6:
        u = np.array([0.0, 1.0, 0.0])
        v = np.cross(nrm, u)
    v = v / np.linalg.norm(v)
    centers, normals = [], []
    for _ in range(n):
        a, b = rng.uniform(-2, 2), rng.uniform(-2, 2)
        centers.append(a * u + b * v + rng.uniform(-0.5, 0.5) * nrm)
        jitter = nrm + rng.normal(0, 0.02, 3)
        normals.append(jitter / np.linalg.norm(jitter))
    return DFNRealization(
        centers=np.array(centers), normals=np.array(normals),
        radii=np.full(n, 5.0), sets=np.zeros(n, dtype=int))


def test_plane_axis_angle_math():
    """面-轴夹角换算: 轴在面内=0°, 轴⊥面=90°."""
    n_z = np.array([0.0, 0.0, 1.0])
    assert abs(_plane_axis_angle_deg(n_z, [1, 0, 0]) - 0.0) < 1e-9
    assert abs(_plane_axis_angle_deg(n_z, [0, 0, 1]) - 90.0) < 1e-9


def test_egs_horizontal_set():
    """水平组系: 竖直井对=较差, 水平井对=良好 (旧实现两者都判'良好')."""
    hz = _single_set_dfn([0, 0, 1])
    m_v = egs_connectivity_metric(hz, [0, 0, 1])
    m_h = egs_connectivity_metric(hz, [1, 0, 0])
    assert '较差' in m_v['assessment'], f"竖直井对⊥水平面应较差, got {m_v['assessment']}"
    assert m_h['assessment'] == '对齐良好', f"水平井对在面内应良好, got {m_h['assessment']}"


def test_egs_steep_set():
    """陡立组系(法向x): x 向井对=较差, z 向井对在面内=良好."""
    st = _single_set_dfn([1, 0, 0])
    m_bad = egs_connectivity_metric(st, [1, 0, 0])
    m_good = egs_connectivity_metric(st, [0, 0, 1])
    assert '较差' in m_bad['assessment'], f"got {m_bad['assessment']}"
    assert m_good['assessment'] == '对齐良好', f"got {m_good['assessment']}"


def test_mine_risk_semantics():
    """巷道轴躺在连通面内 → 高风险; 垂直于面 → 低 (旧实现反向)."""
    hz = _single_set_dfn([0, 0, 1])
    m_high = mine_risk_sections(hz, [1, 0, 0])
    m_low = mine_risk_sections(hz, [0, 0, 1])
    assert m_high['risk_level'] == '高', f"沿面贯通应高风险, got {m_high['risk_level']}"
    assert m_low['risk_level'] == '低', f"巷道⊥连通面应低风险, got {m_low['risk_level']}"


def test_disposal_escape_semantics():
    """垂直方向躺在陡立连通面内 → 高逃逸; 水平面 → 低 (旧实现反向)."""
    steep = _single_set_dfn([1, 0, 0])
    hz = _single_set_dfn([0, 0, 1])
    m_high = disposal_escape_priority(steep, [0, 0, 1])
    m_low = disposal_escape_priority(hz, [0, 0, 1])
    assert m_high['escape_priority'] == '高', f"陡立面含垂直方向应高逃逸, got {m_high['escape_priority']}"
    assert m_low['escape_priority'] == '低', f"水平面逃逸应低, got {m_low['escape_priority']}"


def test_isotropic_guard(monkeypatch=None):
    """dominant≈0 (各向同性) 时判级='不适用', 角度=None (非 NaN)."""
    import fractureflow.percolation as pc

    def fake_aniso(dfn, pbc=False):
        return {'dominant_direction': np.zeros(3), 'largest_fraction': 0.9,
                'n_components': 1, 'component_sizes': [len(dfn.radii)]}

    orig = pc.connectivity_anisotropy
    pc.connectivity_anisotropy = fake_aniso
    try:
        dfn = _single_set_dfn([0, 0, 1])
        me = egs_connectivity_metric(dfn, [1, 0, 0])
        mm = mine_risk_sections(dfn, [1, 0, 0])
        md = disposal_escape_priority(dfn)
        assert me['angle_to_well_pair_deg'] is None and '各向同性' in me['assessment']
        assert mm['risk_level'].startswith('不适用')
        assert md['escape_priority'].startswith('不适用')
    finally:
        pc.connectivity_anisotropy = orig


def test_output_json_strict():
    """指标 dict 必须过 allow_nan=False 序列化 (防 NaN JSON 再犯).

    对齐交付管线真实 dump 参数 (default=str, 见 dfn_from_borehole.py)。
    """
    hz = _single_set_dfn([0, 0, 1])
    for m in (egs_connectivity_metric(hz, [1, 0, 0]),
              mine_risk_sections(hz, [1, 0, 0]),
              disposal_escape_priority(hz, [0, 0, 1])):
        json.dumps(m, allow_nan=False, default=str)


if __name__ == "__main__":
    tests = [
        test_plane_axis_angle_math,
        test_egs_horizontal_set,
        test_egs_steep_set,
        test_mine_risk_semantics,
        test_disposal_escape_semantics,
        test_isotropic_guard,
        test_output_json_strict,
    ]
    passed = 0
    for t in tests:
        try:
            if t.__name__ == 'test_isotropic_guard':
                t()
            else:
                t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
