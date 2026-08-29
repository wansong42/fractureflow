# -*- coding: utf-8 -*-
"""R56: 核废场景处置适宜性判级引擎 —— 纯函数三档判级 + 翻转敏感性.

设计原则 (R56 任务书 + 架构师方案精化):
  1. 判级是**纯函数**: (判级原料) -> 三档判级。纯函数才能系统性网格化
     计算"判定翻转敏感性"矩阵 (β × P32 × K → 判级是否翻转)。
  2. 判级语义 (核废处置, 连通性维度):
     - 裂隙网络越连通 + 垂直逃逸通道越强 -> 核素越易随地下水逃逸 -> 越不适宜。
     - 判级核心 = **估计 P32 区间 [p10, p90] 与渗流阈值 p32_crit 区间
       [crit_lo, crit_hi] 的重叠关系** (不确定性带语义统一)。
     - 垂直逃逸优先级 (disposal_escape_priority) 作为**保守修正**: 高逃逸
       -> 判级在连通性基础上再降一档 (更保守)。
  3. 三档: 适宜 / 需补充勘查 / 不适宜。全部走词表 (templates/nuclear_disposal.json),
     输出强制 assumptions 非空 + uncertainty_band + "筛查级, 非场址最终判定"水印。

纪律红线:
  - 只写"筛查级", 不写"保证不泄漏"类承诺 (对外红线)。
  - P32 体视学量纲问题 (口径锁定 §7.4): 报告引用**区间**而非点值。
  - 判级措辞禁越界; beishan 数据无 fracture_id -> 全链走组系表口径。

用法:
    from fractureflow.disposal_grading import grade_disposal, sensitivity_matrix
    grade = grade_disposal(
        p32_crit_lo=0.19, p32_crit_hi=0.28, p32_crit_median=0.24,
        p32_est_p10=0.05, p32_est_p90=0.9,
        escape_priority="中", template=load_template("nuclear_disposal"))
    mat = sensitivity_matrix(grade_core_fn, ...)  # 翻转敏感性网格
"""

import json
import os
from typing import Callable, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 判级档位常量 (与模板词表保持一致)
# ---------------------------------------------------------------------------
GRADE_SUITABLE = "适宜"
GRADE_SUPPLEMENT = "需补充勘查"
GRADE_UNSUITABLE = "不适宜"

_GRADES = (GRADE_SUITABLE, GRADE_SUPPLEMENT, GRADE_UNSUITABLE)

# 逃逸优先级 -> 保守惩罚档位数 (高逃逸 = 判级更保守一档)
_ESCAPE_PENALTY = {"高": 1, "中": 0, "低": 0, "不适用 (各向同性)": 0}


def _parse_escape(escape_priority: Optional[str]) -> str:
    """容错解析逃逸优先级 (兼容 None / 前缀匹配)."""
    if escape_priority is None:
        return "中"
    s = str(escape_priority)
    if s.startswith("高"):
        return "高"
    if s.startswith("低"):
        return "低"
    if s.startswith("不适用") or "各向同性" in s:
        return "不适用 (各向同性)"
    return "中"


def _grade_from_overlap(crit_lo: float, crit_hi: float,
                        est_lo: float, est_hi: float) -> tuple:
    """由 P32 区间与渗流阈值区间的重叠关系得基础判级。

    返回 (grade, overlap_desc, evidence):
      - est_hi < crit_lo: 估计 P32 整体低于渗流阈值 -> 弱连通 -> 基础"适宜"
      - est_lo > crit_hi: 估计 P32 整体高于渗流阈值 -> 强连通 -> 基础"不适宜"
      - 否则: 区间重叠 (阈值落入估计区间) -> "需补充勘查" (证据不足定档)

    overlap_desc 与 evidence 是给报告用的可读理由与数值依据。
    """
    if est_hi < crit_lo:
        grade = GRADE_SUITABLE
        desc = "P32 估计区间整体低于渗流阈值区间"
        ev = f"P32∈[{est_lo:.3f},{est_hi:.3f}] < p32_crit∈[{crit_lo:.3f},{crit_hi:.3f}]"
    elif est_lo > crit_hi:
        grade = GRADE_UNSUITABLE
        desc = "P32 估计区间整体高于渗流阈值区间"
        ev = f"P32∈[{est_lo:.3f},{est_hi:.3f}] > p32_crit∈[{crit_lo:.3f},{crit_hi:.3f}]"
    else:
        grade = GRADE_SUPPLEMENT
        desc = "P32 估计区间与渗流阈值区间重叠 (证据不足以定档)"
        ev = f"P32∈[{est_lo:.3f},{est_hi:.3f}] 与 p32_crit∈[{crit_lo:.3f},{crit_hi:.3f}] 重叠"
    return grade, desc, ev


def _penalize(grade: str, escape_priority: str) -> str:
    """垂直逃逸优先级保守修正: 高逃逸 -> 判级降一档 (更保守)."""
    pen = _ESCAPE_PENALTY.get(escape_priority, 0)
    if pen == 0 or grade == GRADE_UNSUITABLE:
        # 已是不适宜 (最低档) 或无需惩罚 (中/低逃逸), 不再下探
        return grade
    # 降一档: 适宜 -> 需补充勘查 -> 不适宜
    if grade == GRADE_SUITABLE:
        return GRADE_SUPPLEMENT
    return GRADE_UNSUITABLE


def grade_disposal(
        p32_crit_lo: float,
        p32_crit_hi: float,
        p32_crit_median: Optional[float],
        p32_est_p10: float,
        p32_est_p90: float,
        escape_priority: Optional[str] = None,
        template: Optional[dict] = None,
        site_name: str = "北山预选区") -> dict:
    """核废处置适宜性判级 (纯函数).

    参数:
        p32_crit_lo/hi: 渗流阈值 p32_crit 的不确定带下/上界 (percolation_curve 输出).
        p32_crit_median: 渗流阈值中值 (可为 None).
        p32_est_p10/p90: 编录表 P32 估计区间 (estimate_p32_interval 输出).
        escape_priority: 垂直逃逸优先级 (disposal_escape_priority 输出, 可 None).
        template: nuclear_disposal.json 词表 (可选, 提供理由措辞/水印).
        site_name: 场地名称.

    返回 dict (含 grade / grade_base / evidence / assumptions / uncertainty_band /
    watermark / sensitivity-ready 判级原料)。
    """
    if template is None:
        template = _DEFAULT_TEMPLATE

    grade_base, overlap_desc, overlap_ev = _grade_from_overlap(
        float(p32_crit_lo), float(p32_crit_hi),
        float(p32_est_p10), float(p32_est_p90))
    esc = _parse_escape(escape_priority)
    grade = _penalize(grade_base, esc)

    # 词表理由 + 水印 (对外红线: 只写筛查级)
    grade_words = template.get("grades", {}).get(grade, {})
    reason_tmpl = grade_words.get("reason", "筛查级结论")
    reason = reason_tmpl.format(
        overlap=overlap_desc,
        evidence=overlap_ev,
        escape=esc,
        site=site_name)

    assumptions = [
        "筛查级结论, 基于结构面连通性维度 (Baecher 盘模型 + 假设尺寸分布)",
        "判级依据: P32 估计区间与渗流阈值 p32_crit 区间的重叠关系, 叠加垂直逃逸优先级保守修正",
        "非场址最终判定; 未包含水文地质、地球化学、工程屏障、地质构造稳定性等评价维度",
    ]
    if template.get("assumptions"):
        assumptions.extend(template["assumptions"])

    return {
        "grade": grade,
        "grade_base": grade_base,
        "grades_order": list(_GRADES),
        "overlap": {
            "description": overlap_desc,
            "evidence": overlap_ev,
        },
        "escape_penalty": {
            "escape_priority": esc,
            "penalty_tiers": _ESCAPE_PENALTY.get(esc, 0),
        },
        "p32_crit_band": [round(float(p32_crit_lo), 3),
                          round(float(p32_crit_hi), 3)],
        "p32_crit_median": (None if p32_crit_median is None
                            else round(float(p32_crit_median), 3)),
        "p32_est_band": [round(float(p32_est_p10), 3),
                         round(float(p32_est_p90), 3)],
        "reason": reason,
        "assumptions": assumptions,
        "uncertainty_band": (
            f"P32 体视学量纲不确定 (编录表缺尺寸); p32_crit∈"
            f"[{p32_crit_lo:.3f},{p32_crit_hi:.3f}] 来自多实现交叉带"),
        "watermark": template.get("watermark",
                                  "筛查级, 非场址最终判定"),
    }


def _bin_score(value: float, lo: float, hi: float) -> str:
    """翻转敏感性网格值判级归类 (与 grade_disposal 内 _grade_from_overlap 一致)."""
    if value < lo:
        return GRADE_SUITABLE
    if value > hi:
        return GRADE_UNSUITABLE
    return GRADE_SUPPLEMENT


def sensitivity_matrix(
        p32_crit_lo: float, p32_crit_hi: float,
        p32_est_p10: float, p32_est_p90: float,
        escape_priority: Optional[str] = None,
        template: Optional[dict] = None,
        betas: tuple = (2.5, 3.0, 3.5, 4.0, 4.5),
        p32_factors: tuple = (0.5, 1.0, 2.0),
        K_offsets: tuple = (-1, 0, 1)) -> dict:
    """判定翻转敏感性矩阵.

    不重新跑渗流 (代价高), 而是对**判级原料** (p32_crit 区间 / P32 估计区间 /
    逃逸优先级) 做扰动, 观察三档判级是否翻转 —— 这回答 "当前判级在参数
    不确定下的稳健性"。

    参数:
        p32_crit_lo/hi: 基线渗流阈值区间.
        p32_est_p10/p90: 基线 P32 估计区间.
        escape_priority: 基线逃逸优先级.
        betas: β 扫描档 (幂律指数对 p32_crit 的影响, 语义上宽 β -> 更多小裂隙
               -> 连通更难 -> p32_crit 上升; 此处仅作示例扰动, 与 percolation
               实际 β 扫描可对应).
        p32_factors: P32 估计区间的乘性扰动因子 (×0.5 保守, ×2 悲观).
        K_offsets: K ± 1 对判级原料的扰动 (K 变化影响聚类 -> p32_crit 微变).

    返回 dict:
        {
            "rows": [{"beta":.., "p32_factor":.., "k_offset":.., "grade":..,
                      "flipped": bool}],
            "n_total":.., "n_flipped":.., "flip_rate":..,
            "baseline_grade":..,
        }
    """
    baseline = grade_disposal(
        p32_crit_lo, p32_crit_hi, None, p32_est_p10, p32_est_p90,
        escape_priority=escape_priority, template=template)
    base_grade = baseline["grade"]

    rows = []
    n_flipped = 0
    for beta in betas:
        # β 对 p32_crit 的单调扰动 (语义: β 越大, 幂律越重小半径, 连通更困难,
        # p32_crit 缓慢上移)。幅度设为基线区间宽度的 ±20%/±10% 示例, 标记为
        # "示例扰动, 需与 percolation β 扫描对齐"。
        beta_shift = 0.0
        if beta > 3.5:
            beta_shift = (p32_crit_hi - p32_crit_lo) * 0.1
        elif beta < 3.5:
            beta_shift = -(p32_crit_hi - p32_crit_lo) * 0.1
        for fac in p32_factors:
            est_lo = p32_est_p10 * fac
            est_hi = p32_est_p90 * fac
            for koff in K_offsets:
                # K 偏移对 p32_crit 的扰动 (K 变化聚类组心微移 -> 渗流阈值微变)
                k_shift = (p32_crit_hi - p32_crit_lo) * 0.05 * koff
                crit_lo = p32_crit_lo + beta_shift + k_shift
                crit_hi = p32_crit_hi + beta_shift + k_shift
                g = _grade_from_overlap(crit_lo, crit_hi, est_lo, est_hi)[0]
                g = _penalize(g, _parse_escape(escape_priority))
                flipped = (g != base_grade)
                if flipped:
                    n_flipped += 1
                rows.append({
                    "beta": float(beta), "p32_factor": float(fac),
                    "k_offset": int(koff), "grade": g, "flipped": flipped,
                })

    return {
        "rows": rows,
        "n_total": len(rows),
        "n_flipped": n_flipped,
        "flip_rate": round(n_flipped / max(len(rows), 1), 4),
        "baseline_grade": base_grade,
        "baseline_params": {
            "p32_crit_lo": float(p32_crit_lo),
            "p32_crit_hi": float(p32_crit_hi),
            "p32_est_p10": float(p32_est_p10),
            "p32_est_p90": float(p32_est_p90),
            "escape_priority": _parse_escape(escape_priority),
        },
        "legend": {
            "beta_shift_note": "β 对 p32_crit 的示例单调扰动 (需与 percolation β 扫描对齐)",
            "k_offset_note": "K±1 对 p32_crit 的示例扰动",
            "flip_meaning": "该格参数扰动下判级相对基线是否翻转",
        },
    }


# ---------------------------------------------------------------------------
# 默认词表 (当未提供模板时的兜底; 正常流程走 templates/nuclear_disposal.json)
# ---------------------------------------------------------------------------
_DEFAULT_TEMPLATE = {
    "grades": {
        GRADE_SUITABLE: {
            "reason": "连通性维度筛查: {overlap}, 且垂直逃逸优先级为 {escape}。"
                      "该维度支持场址可作为候选进一步勘查。"
        },
        GRADE_SUPPLEMENT: {
            "reason": "连通性维度筛查: {overlap}, 证据不足以定档。"
                      "需补充延展性/迹长/渗透试验等数据以收窄不确定性带。"
        },
        GRADE_UNSUITABLE: {
            "reason": "连通性维度筛查: {overlap}, 且垂直逃逸优先级为 {escape}。"
                      "从裂隙连通性看不利于核素滞留, 需重点评估。"
        },
    },
    "assumptions": [],
    "watermark": "筛查级, 非场址最终判定",
}


def load_nuclear_template(path: Optional[str] = None) -> dict:
    """加载 nuclear_disposal.json 词表 (无则返回默认)."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "templates", "nuclear_disposal.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return dict(_DEFAULT_TEMPLATE)
