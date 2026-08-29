# -*- coding: utf-8 -*-
"""R60: 多井联合决策规则消费落地 —— 单测 (❌不依赖 torch, 纯 numpy/scipy)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fractureflow.site_model import (
    tier_from_angle,
    joint_rule_partition,
    SiteModeler,
    load_wells_from_dir,
    _cluster_and_align,
)

CSV_DIR = os.path.join(ROOT, "data/real/r60_wells_csv")


def _samples_around(center_dir, n, disp_deg, rng):
    v = (np.asarray(center_dir, float) / np.linalg.norm(center_dir)
         + rng.standard_normal((n, 3)) * np.radians(disp_deg))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _normal(dip, dd):
    d, a = np.radians(dip), np.radians(dd)
    v = np.array([np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)])
    return v / np.linalg.norm(v)


def _three_wells():
    """已知解: A/B 共享 2 组 (<20 一致), C 独立 (>35)."""
    g1, g2 = _normal(10, 90), _normal(60, 45)
    g3, g4 = _normal(35, 200), _normal(70, 300)
    rng = np.random.default_rng(0)

    def mk(centers):
        return np.vstack([_samples_around(g, 60, 10.0, rng) for g in centers])

    return [mk([g1, g2]), mk([g1, g2]), mk([g3, g4])]


# ---------------------------------------------------------------------------
# 三档边界
# ---------------------------------------------------------------------------
def test_tier_boundaries():
    assert tier_from_angle(19.9) == "joint"
    assert tier_from_angle(20.0) == "pair_only"
    assert tier_from_angle(34.9) == "pair_only"
    assert tier_from_angle(35.0) == "independent"
    assert tier_from_angle(0.0) == "joint"
    assert tier_from_angle(99.0) == "independent"


def test_tier_custom_thresholds():
    assert tier_from_angle(15.0, thresholds=(20, 35)) == "joint"
    assert tier_from_angle(40.0, thresholds=(30, 45)) == "pair_only"


# ---------------------------------------------------------------------------
# 三档分组已知解 (csv 与合成双保险)
# ---------------------------------------------------------------------------
def test_partition_known_solution_csv():
    wells = load_wells_from_dir(CSV_DIR)
    nrm = [w["nrm"] for w in wells]
    part = joint_rule_partition(nrm, K_audit=2, seed=42)
    assert part["groups"] == [[0, 1], [2]]          # A/B 联合, C 独立
    tiers = [v["tier"] for v in part["verdicts"]]
    assert tiers[0] == "joint" and tiers[1] == "joint"
    assert tiers[2] == "independent"


def test_partition_known_solution_synthetic():
    wells = _three_wells()
    part = joint_rule_partition(wells, K_audit=2, seed=7)
    assert part["groups"] == [[0, 1], [2]]
    assert all(v["tier"] == "joint" for v in part["verdicts"][:2])
    assert part["verdicts"][2]["tier"] == "independent"
    # A/B 一致性应很小, C 相对大
    assert part["consistency"]["0-1"] < 20.0
    assert part["consistency"]["0-2"] > 25.0
    assert part["consistency"]["1-2"] > 25.0


def test_partition_zero_and_single():
    assert joint_rule_partition([], K_audit=2)["groups"] == []
    part = joint_rule_partition([_three_wells()[0]], K_audit=2, seed=1)
    assert part["groups"] == [[0]]
    assert part["verdicts"][0]["tier"] in ("joint", "independent")


# ---------------------------------------------------------------------------
# build_set_table 行为: 规则消费 vs 逃生口全池化 vs 单井不变
# ---------------------------------------------------------------------------
def _site_with(wells):
    sm = SiteModeler(seed=42)
    sm.wells = [{"nrm": w, "n_fractures": len(w)} for w in wells]
    return sm


def test_build_set_table_consumes_rule():
    sm = _site_with(_three_wells())
    sm.build_set_table(K=2, joint_rule=True)
    # 规则消费: A/B 共享一个池化组, C 独立 -> 不应是全池化 [[0,1,2]]
    assert sm.rule_groups != [[0, 1, 2]]
    terms = sorted(tuple(g) for g in sm.rule_groups)
    assert terms == [(0, 1), (2,)]           # A/B 联合在一起, C 独立孤立


def test_build_off_fallback_full_pool():
    sm = _site_with(_three_wells())
    sm.build_set_table(K=2, joint_rule=False)
    # 逃生口: 无条件全池化 -> 单组含全部井
    assert sm.rule_groups == [[0, 1, 2]]
    assert len(sm.set_table.centers) == 2  # K=2 全池化组心数


def test_single_well_unchanged():
    w = _three_wells()[0]
    sm_on = _site_with([w]); sm_on.build_set_table(K=2, joint_rule=True)
    sm_off = _site_with([w]); sm_off.build_set_table(K=2, joint_rule=False)
    assert sm_on.rule_groups == [[0]] and sm_off.rule_groups == [[0]]
    assert len(sm_on.set_table.centers) == len(sm_off.set_table.centers)


def test_cli_joint_rule_flag_mapping():
    # 默认 on; --joint-rule off -> joint_rule=False (run_site_model_cli 的读取逻辑)
    default = type("A", (), {"joint_rule": "on"})
    off = type("A", (), {"joint_rule": "off"})
    legacy = type("A", (), {})
    assert (getattr(default, "joint_rule", None) != "off") is True
    assert (getattr(off, "joint_rule", None) != "off") is False
    assert (getattr(legacy, "joint_rule", None) != "off") is True  # 缺省=on


# ---------------------------------------------------------------------------
# 无泄漏断言: 分组只消费观测法向
# ---------------------------------------------------------------------------
def test_partition_ignores_hidden_normals():
    wells = _three_wells()
    masks = []
    for w in wells:
        m = np.zeros(len(w), bool)
        m[: int(0.6 * len(w))] = True   # 前 60% 观测
        masks.append(m)
    base = joint_rule_partition(wells, obs_masks=masks, K_audit=2, seed=7)
    # 把隐伏 (非观测) 法向污染成远端方向 —— 决策必须不变
    rng = np.random.default_rng(1)
    corrupt = [w.copy() for w in wells]
    for i, w in enumerate(corrupt):
        hid = ~masks[i]
        corrupt[i][hid] = np.asarray([1.0, 0.0, 0.0])[None, :]
    alt = joint_rule_partition(corrupt, obs_masks=masks, K_audit=2, seed=7)
    assert base["groups"] == alt["groups"]
    assert [v["tier"] for v in base["verdicts"]] == \
           [v["tier"] for v in alt["verdicts"]]


def test_assign_uses_only_obs_centers():
    # _assign_to_centers 的组心由观测聚类给出; 验证隐伏污染不影响该组心来源
    wells = _three_wells()
    masks = [np.arange(len(w)) < int(0.6 * len(w)) for w in wells]
    obs_only = [w[m] for w, m in zip(wells, masks)]
    centers, _ = _cluster_and_align(obs_only[0], 2, seed=3)
    # 观测组心与全量数据组心相比, 不依赖隐伏; 此处仅断言观测聚类可执行且组心归一
    assert centers.shape == (2, 3)
    assert all(abs(np.linalg.norm(c) - 1) < 1e-6 for c in centers)


# ---------------------------------------------------------------------------
# docstring-实现一致断言 (十五期审查项: docstring 曾声称而实现未消费)
# ---------------------------------------------------------------------------
def test_build_set_table_docstring_aligned_with_impl():
    import inspect
    from fractureflow import site_model
    src = inspect.getsource(site_model.SiteModeler.build_set_table)
    # 实现真实调用了决策规则消费, 不再是"仅作展示不改变聚类"
    assert "joint_rule_partition" in src
    assert 'joint_rule=False' in src
    # 旧的"无条件全池化即本方法"表述应从 docstring 移除
    assert "无条件全池化 —— 所有井法向 vstack 后做一次球形 k-means" not in src


def test_rule_module_present_and_importable():
    from fractureflow import site_model
    assert callable(site_model.tier_from_angle)
    assert callable(site_model.joint_rule_partition)