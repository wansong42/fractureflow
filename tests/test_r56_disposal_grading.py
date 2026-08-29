# -*- coding: utf-8 -*-
"""R56: 核废处置判级引擎 + site_report 组件单测.

覆盖 (DoD: 三档判级 + 翻转表 + CI + bootstrap + load_well_csv 容错 ≥8 测试):
  1. 三档判级: P32<阈值→适宜; P32>阈值→不适宜; 区间重叠→需补充勘查
  2. 垂直逃逸保守修正: 高逃逸→判级降一档
  3. 判级 dict 可 JSON 序列化 (allow_nan=False)
  4. 翻转敏感性矩阵: baseline 判级 + flip 计算
  5. bootstrap p32_crit 逻辑: 多 seed 集 → 中值/分位 CI
  6. load_well_csv 缺列 → ValueError (含缺失列清单)
  7. load_well_csv 坏行 → 跳过 + n_skipped 计数
  8. load_well_csv 全坏 → ValueError
  9. 边界情形: 翻转矩阵在 P32 跨阈值附近确实会翻转
 10. 词表模板结构完整 (三档词表 + watermark + assumptions)
"""
import json
import os
import sys
import tempfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fractureflow.disposal_grading import (
    grade_disposal, sensitivity_matrix, _grade_from_overlap,
    GRADE_SUITABLE, GRADE_SUPPLEMENT, GRADE_UNSUITABLE, load_nuclear_template)
from fractureflow.site_model import load_well_csv


# ---------------------------------------------------------------------------
# 三档判级
# ---------------------------------------------------------------------------

def test_grade_suitable_when_p32_below_crit():
    """P32 估计整体低于渗流阈值 → 适宜."""
    g = grade_disposal(
        p32_crit_lo=0.30, p32_crit_hi=0.40, p32_crit_median=0.35,
        p32_est_p10=0.01, p32_est_p90=0.10,
        escape_priority="低")
    assert g["grade"] == GRADE_SUITABLE, g["grade"]


def test_grade_unsuitable_when_p32_above_crit():
    """P32 估计整体高于渗流阈值 → 不适宜."""
    g = grade_disposal(
        p32_crit_lo=0.15, p32_crit_hi=0.20, p32_crit_median=0.18,
        p32_est_p10=0.8, p32_est_p90=3.0,
        escape_priority="高")
    assert g["grade"] == GRADE_UNSUITABLE, g["grade"]


def test_grade_supplement_when_overlap():
    """P32 估计区间与渗流阈值区间重叠 → 需补充勘查."""
    g = grade_disposal(
        p32_crit_lo=0.20, p32_crit_hi=0.40, p32_crit_median=0.30,
        p32_est_p10=0.05, p32_est_p90=0.90,
        escape_priority="中")
    assert g["grade"] == GRADE_SUPPLEMENT, g["grade"]


# ---------------------------------------------------------------------------
# 垂直逃逸保守修正
# ---------------------------------------------------------------------------

def test_escape_penalty_lowers_grade():
    """基础'适宜' + 高垂直逃逸 → 降为'需补充勘查'."""
    g_safe = grade_disposal(
        p32_crit_lo=0.30, p32_crit_hi=0.40, p32_crit_median=0.35,
        p32_est_p10=0.01, p32_est_p90=0.10,
        escape_priority="低")
    g_risky = grade_disposal(
        p32_crit_lo=0.30, p32_crit_hi=0.40, p32_crit_median=0.35,
        p32_est_p10=0.01, p32_est_p90=0.10,
        escape_priority="高")
    assert g_safe["grade"] == GRADE_SUITABLE
    # 高逃逸: 适宜 → 需补充勘查 (保守降档)
    assert g_risky["grade"] == GRADE_SUPPLEMENT, g_risky["grade"]


def test_escape_penalty_does_not_raise_unsuitable():
    """已'不适宜'不受逃逸惩罚影响 (不设惩罚, 不升档)."""
    g = grade_disposal(
        p32_crit_lo=0.15, p32_crit_hi=0.20, p32_crit_median=0.18,
        p32_est_p10=0.8, p32_est_p90=3.0,
        escape_priority="低")
    assert g["grade"] == GRADE_UNSUITABLE


# ---------------------------------------------------------------------------
# 可序列化 + 纪律红线
# ---------------------------------------------------------------------------

def test_grade_json_serializable():
    """判级 dict 必须过 allow_nan=False 序列化 + 强制 assumptions/水印."""
    g = grade_disposal(
        p32_crit_lo=0.20, p32_crit_hi=0.40, p32_crit_median=0.30,
        p32_est_p10=0.05, p32_est_p90=0.90, escape_priority="中")
    json.dumps(g, ensure_ascii=False, allow_nan=False, default=str)
    assert len(g["assumptions"]) > 0, "assumptions 必须非空 (H1 规则)"
    assert "筛查级" in g["watermark"], "必须含'筛查级, 非场址最终判定'水印"
    assert "保证不泄漏" not in g["watermark"], "禁止'保证不泄漏'类承诺"


# ---------------------------------------------------------------------------
# 翻转敏感性矩阵
# ---------------------------------------------------------------------------

def test_sensitivity_matrix_baseline_and_flip():
    """翻转矩阵: baseline 判级一致 + 翻转率在 (0,1) 之间."""
    sens = sensitivity_matrix(
        p32_crit_lo=0.20, p32_crit_hi=0.40,
        p32_est_p10=0.05, p32_est_p90=0.90,
        escape_priority="中")
    assert sens["n_total"] == len(sens["rows"])
    assert sens["n_total"] > 0
    assert 0 <= sens["n_flipped"] <= sens["n_total"]
    assert sens["flip_rate"] == round(sens["n_flipped"] / sens["n_total"], 4)


def test_sensitivity_flips_at_boundary():
    """边界情形: P32 估计紧贴阈值, 乘性扰动导致判级翻转 (矩阵有翻转)."""
    # 构造: P32 整体略低于阈值 → 基础"适宜"; P32×2.0 后跨过阈值 → 翻转
    sens = sensitivity_matrix(
        p32_crit_lo=0.30, p32_crit_hi=0.40,
        p32_est_p10=0.10, p32_est_p90=0.20,   # 整体 < 0.30 → 适宜
        escape_priority="低")
    # p32_factor=2.0: est_hi=0.40 -> 与阈值重叠 → 需补充勘查 (翻转)
    assert sens["n_flipped"] > 0, "边界情形应产生翻转"


def test_sensitivity_stable_when_well_separated():
    """强分离情形: 翻转矩阵应为 0 翻转 (判级稳健)."""
    sens = sensitivity_matrix(
        p32_crit_lo=0.30, p32_crit_hi=0.40,
        p32_est_p10=0.001, p32_est_p90=0.002,   # 极远低于阈值 → 始终适宜
        escape_priority="低")
    assert sens["n_flipped"] == 0, \
        f"强分离情形不应翻转, got {sens['n_flipped']} flips"


# ---------------------------------------------------------------------------
# bootstrap CI
# ---------------------------------------------------------------------------

def test_bootstrap_p32crit_structure():
    """bootstrap 输出含中值/上下界/样本/有效集数."""
    from fractureflow.dfn import SetTable
    from fractureflow.site_report import _bootstrap_p32crit
    st = SetTable.from_centers([
        [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p32_est = {"p32_p10": 0.05, "p32_p50": 0.3, "p32_p90": 0.9}
    # 用最少 seed 集跑通结构 (单元内验证结构, 不重算全量)
    res = _bootstrap_p32crit(st, (20, 20, 20), p32_est,
                             n_sets=2, n_real=4, beta=3.5)
    assert res["p32_crit_lo"] <= res["p32_crit_median"] <= res["p32_crit_hi"]
    assert res["n_sets_ok"] >= 1
    assert len(res["p32_crit_samples"]) == res["n_sets_ok"]


# ---------------------------------------------------------------------------
# load_well_csv 容错
# ---------------------------------------------------------------------------

def _write_csv(tmpdir, content, name="w1.csv"):
    p = os.path.join(tmpdir, name)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    return p


def test_load_well_csv_missing_columns():
    """缺列 → ValueError 且报出缺失列清单."""
    with tempfile.TemporaryDirectory() as td:
        p = _write_csv(td, "depth,dip\n1,10\n2,20\n")
        try:
            load_well_csv(p)
            assert False, "应抛 ValueError"
        except ValueError as e:
            assert "dip_direction" in str(e), f"缺列清单应含 dip_direction, got {e}"


def test_load_well_csv_bad_rows_skipped():
    """坏行跳过 + n_skipped 计数."""
    with tempfile.TemporaryDirectory() as td:
        p = _write_csv(td, "depth,dip,dip_direction\n1,10,30\nbad,20,40\n3,abc,50\n")
        w = load_well_csv(p)
        assert w["n_fractures"] == 1, f"应保留 1 条有效, got {w['n_fractures']}"
        assert w["n_skipped"] == 2, f"应跳过 2 行坏数据, got {w['n_skipped']}"


def test_load_well_csv_all_bad():
    """全部行坏 → ValueError."""
    with tempfile.TemporaryDirectory() as td:
        p = _write_csv(td, "depth,dip,dip_direction\nbad,bad,bad\nx,y,z\n")
        try:
            load_well_csv(p)
            assert False, "应抛 ValueError"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# 词表模板
# ---------------------------------------------------------------------------

def test_nuclear_template_structure():
    """模板词表三档齐全 + watermark + assumptions."""
    tmpl = load_nuclear_template()
    for g in (GRADE_SUITABLE, GRADE_SUPPLEMENT, GRADE_UNSUITABLE):
        assert g in tmpl["grades"], f"模板缺 {g} 档"
    assert "watermark" in tmpl and "筛查级" in tmpl["watermark"]
    assert len(tmpl["assumptions"]) > 0


if __name__ == "__main__":
    tests = [
        test_grade_suitable_when_p32_below_crit,
        test_grade_unsuitable_when_p32_above_crit,
        test_grade_supplement_when_overlap,
        test_escape_penalty_lowers_grade,
        test_escape_penalty_does_not_raise_unsuitable,
        test_grade_json_serializable,
        test_sensitivity_matrix_baseline_and_flip,
        test_sensitivity_flips_at_boundary,
        test_sensitivity_stable_when_well_separated,
        test_bootstrap_p32crit_structure,
        test_load_well_csv_missing_columns,
        test_load_well_csv_bad_rows_skipped,
        test_load_well_csv_all_bad,
        test_nuclear_template_structure,
    ]
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
