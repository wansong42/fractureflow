# -*- coding: utf-8 -*-
"""雷2 + set_table_score dispersion/max_dev 回收测试 (外部审查 2026-08-29).

覆盖:
  - 雷2 退化分支: 小井 (观测点<=1) assign 掩码语义 + n_points 报点数
  - _group_dispersion: 对跖簇(0°)/跨边界簇(350°~10°)/N=2 不可统计 三种边界
  - set_table_score 纯加法: 无 nrm 时 dispersion=None, 旧字段零漂移; 有 nrm 时算对
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fractureflow.set_table_eval import (
    set_table_score,
    estimate_set_table,
    _group_dispersion,
)
from fractureflow.borehole_report import build_group_table


def _normal(dipdir_deg, dip_deg):
    a = np.radians(dipdir_deg)
    d = np.radians(dip_deg)
    return np.array([np.sin(d) * np.cos(a), np.sin(d) * np.sin(a), np.cos(d)])


def test_lei2_degenerate_branch_semantics():
    """雷2: 退化分支 (观测点<=1) assign 掩码方向 + n_points 报点数."""
    pos = np.zeros((3, 3))
    nrm = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    occ = np.array([True, False, False])  # 仅 1 个观测点 -> 触发退化分支

    table = estimate_set_table(pos, nrm, occ, K=4, seed=42, strict=True)
    # 观测点应入组 (assign>=0); 未观测点留 -1
    assert table["assign"][0] == 0, "观测点必须入组"
    assert table["assign"][1] == -1 and table["assign"][2] == -1, "未观测点留 -1"
    # n_points 报已观测点数, 而非 K 值
    assert table["n_points"] == [1], f"n_points 应报点数 [1], 得 {table['n_points']}"
    assert table["centers"].shape == (1, 3)


def test_group_dispersion_antipodal():
    """对跖簇: 同一平面 ±n, 无向口径 dispersion/max_dev 必为 0."""
    center = np.array([[0.0, 0.0, 1.0]])
    nrm = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])  # 对跖
    assign = np.array([0, 0])
    disp = _group_dispersion(center, assign, nrm)
    assert disp[0]["dispersion"] == 0.0, "对跖簇 dispersion 应为 0"
    assert disp[0]["max_dev"] == 0.0, "对跖簇 max_dev 应为 0"
    assert disp[0]["statistical"] is False, "N=2 < 3, 不可统计"


def test_group_dispersion_boundary_350_10():
    """跨边界簇: 走向 350° 与 10° 在 3D 上仅差 ~20°, arccos|cos| 无环绕 bug."""
    nrm = np.vstack([_normal(350, 80), _normal(10, 80)])
    center = nrm.mean(0)
    center = center / (np.linalg.norm(center) + 1e-12)
    assign = np.array([0, 0])
    disp = _group_dispersion(center[None], assign, nrm)
    d = disp[0]["dispersion"]
    # 两点的 3D 角距 ~19.6°, 各点到组心角距 ~9.8°, dispersion 取均值
    assert 5.0 < d < 13.0, f"跨边界簇 dispersion 异常: {d}"
    assert disp[0]["max_dev"] > d - 1e-6, "max_dev 应 >= dispersion"


def test_group_dispersion_n3_statistical():
    """N=3 组系可统计 (statistical=True)."""
    base = _normal(45, 60)
    nrm = np.vstack([base, base * 0.999 + 1e-3, base * 0.999 - 1e-3])
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    center = nrm.mean(0)
    center = center / (np.linalg.norm(center) + 1e-12)
    assign = np.array([0, 0, 0])
    disp = _group_dispersion(center[None], assign, nrm)
    assert disp[0]["n"] == 3
    assert disp[0]["statistical"] is True


def test_set_table_score_additive_zero_drift():
    """纯加法: 无 nrm 时 dispersion=None, 旧字段零漂移."""
    c = np.array([[0.0, 0.0, 1.0]])
    s = set_table_score(c, c)  # 无 pred/truth_assign, 无 nrm
    assert "dispersion_pred" in s and "dispersion_truth" in s
    assert s["dispersion_pred"] is None and s["dispersion_truth"] is None
    # 旧字段: 自评应 0 / 1 / 1
    assert abs(s["modal_err_deg"]) < 1e-6
    assert s["coverage"] == 1.0
    assert s["K_diff"] == 0


def test_set_table_score_with_nrm():
    """有 nrm 时 dispersion 正确 (对跖簇 -> 0)."""
    c = np.array([[0.0, 0.0, 1.0]])
    nrm = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    assign = np.array([0, 0])
    s = set_table_score(c, c, assign, assign, pred_nrm=nrm, truth_nrm=nrm)
    assert s["dispersion_pred"][0]["dispersion"] == 0.0
    assert s["dispersion_truth"][0]["max_dev"] == 0.0


def test_build_group_table_new_columns():
    """交付组系表新增列存在 (bootstrap CI / max_dev / 备注)."""
    nrm = np.vstack([_normal(350, 80), _normal(10, 80)])
    center = nrm.mean(0)
    center = center / (np.linalg.norm(center) + 1e-12)
    assign = np.array([0, 0])
    df = build_group_table(nrm, assign, center[None], K=1)
    for col in ["组内最大偏差(°)", "CI半宽(°)", "备注"]:
        assert col in df.columns, f"缺少新增列 {col}"
    # N=2 -> 不可统计标记
    assert df["组内离散(°)"].iloc[0] == "不可统计(N<3)"


if __name__ == "__main__":
    test_lei2_degenerate_branch_semantics()
    test_group_dispersion_antipodal()
    test_group_dispersion_boundary_350_10()
    test_group_dispersion_n3_statistical()
    test_set_table_score_additive_zero_drift()
    test_set_table_score_with_nrm()
    test_build_group_table_new_columns()
    print("ALL set_table dispersion / 雷2 测试 PASS")
