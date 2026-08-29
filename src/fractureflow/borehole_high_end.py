# -*- coding: utf-8 -*-
"""高端市场准备: 可辩护性 + 多井联合决策 + 数据升档.

原则: 高端市场不缺算法, 缺的是可辩护性与对口案例.

模块:
    - multiwell_joint_decision: 多井联合决策产品化 (T12)
    - generate_data_upgrade_table: 数据升档对齐表 (T13)
    - defensibility_checklist: 可辩护性四件套检查 (T14)
"""
import numpy as np
from typing import List, Optional


# ============================================================
# T12: 多井联合决策产品化
# ============================================================

def multiwell_joint_decision(set_tables: List[dict], K: int = 12,
                               threshold_deg: float = 20.0) -> dict:
    """多井联合决策 (T12): 输入多井组系表 → 联合/独立建议.

    决策规则 (来自 multiwell_joint_audit):
        - 一致性 < 20° → 联合
        - 20-35° → 只联合最一致对
        - > 35° → 各井独立

    参数:
        set_tables: list of dict, 每个 dict = {"centers": (K, 3), "site": str}
        K: 组数
        threshold_deg: 一致性阈值 (°), 缺省 20

    返回:
        decision: {
            "recommendation": "joint"/"partial"/"independent",
            "details": [{"site_i", "site_j", "consistency_deg", "action"}],
            "honest_note": str,
        }
    """
    n = len(set_tables)
    if n < 2:
        return {
            "recommendation": "single_well",
            "details": [],
            "honest_note": "只有一口井, 无需联合决策。",
        }

    # 两两一致性 (平均匹配角距)
    pairwise = []
    for i in range(n):
        for j in range(i + 1, n):
            ci = set_tables[i]["centers"]
            cj = set_tables[j]["centers"]
            ci = ci / (np.linalg.norm(ci, axis=1, keepdims=True) + 1e-12)
            cj = cj / (np.linalg.norm(cj, axis=1, keepdims=True) + 1e-12)
            # 匈牙利匹配
            from scipy.optimize import linear_sum_assignment
            cost = np.zeros((K, K))
            for a in range(K):
                for b in range(K):
                    cos = np.clip(np.abs(ci[a] @ cj[b]), 0, 1)
                    cost[a, b] = np.degrees(np.arccos(cos))
            row_ind, col_ind = linear_sum_assignment(cost)
            mean_angle = float(np.mean(cost[row_ind, col_ind]))
            pairwise.append({
                "site_i": set_tables[i].get("site", f"well_{i}"),
                "site_j": set_tables[j].get("site", f"well_{j}"),
                "consistency_deg": round(mean_angle, 2),
            })

    # 决策
    details = []
    n_joint = 0
    n_independent = 0
    for p in pairwise:
        if p["consistency_deg"] < threshold_deg:
            action = "联合"
            n_joint += 1
        elif p["consistency_deg"] < 35:
            action = "仅联合最一致对"
            n_joint += 0.5
        else:
            action = "独立"
            n_independent += 1
        details.append({**p, "action": action})

    if n_joint >= len(pairwise) * 0.5:
        recommendation = "joint"
    elif n_joint > 0:
        recommendation = "partial"
    else:
        recommendation = "independent"

    return {
        "recommendation": recommendation,
        "threshold_deg": threshold_deg,
        "details": details,
        "honest_note": (
            "决策基于组系方向几何一致性, 非地质成因判定。"
            "联合后组心更稳定, 但可能掩盖场地异质性。"
            "本建议为筛查级, 最终决策需地质师确认。"
        ),
        "assumptions": [
            f"一致性阈值 = {threshold_deg}° (来自 multiwell_joint_audit)",
            "基于法向几何聚类组心, 非地质成因分组",
            "诚实口径: 多井联合在诚实口径下无增益 (见 AGENTS.md)",
        ],
    }


# ============================================================
# T13: 数据升档对齐表
# ============================================================

def generate_data_upgrade_table(current_level: str = "L0") -> dict:
    """数据升档对齐表 (T13).

    L0 → L4 升档清单, 对应报告哪节变准 + 带收窄多少.

    参数:
        current_level: 当前数据等级 (L0-L4)

    返回:
        table: {current_level, upgrades: [{level, data, effect, band_narrowing}]}
    """
    levels = ["L0", "L1", "L2", "L3", "L4"]
    upgrades = [
        {
            "level": "L0",
            "data": "基础编录表 (深度+倾角+倾向)",
            "effect": "组系方向统计 (modal_err ~10°)",
            "band_narrowing": "基线",
            "report_section": "组系表",
        },
        {
            "level": "L1",
            "data": " + 裂隙类型 (natural/induced)",
            "effect": "类型隔离, 组系更纯",
            "band_narrowing": "~2-3° (FORGE 验证)",
            "report_section": "组系表 + 类型分布",
        },
        {
            "level": "L2",
            "data": " + 迹长/延伸长",
            "effect": "β 拟合解锁, P32_crit 带收窄",
            "band_narrowing": "P32_crit 带收窄 ~30%",
            "report_section": "渗流阈值",
        },
        {
            "level": "L3",
            "data": " + 间距/面密度",
            "effect": "P32 直接估计, 不再需要计数法",
            "band_narrowing": "P32 不再需要估计",
            "report_section": "渗流阈值 + 强度",
        },
        {
            "level": "L4",
            "data": " + 开度/粗糙度/充填物",
            "effect": "Oda 绝对渗透解锁 + 连通性校正",
            "band_narrowing": "可输出绝对渗透量级",
            "report_section": "渗透张量 + 连通性",
        },
    ]

    # 只返回当前等级之后的升档建议
    idx = levels.index(current_level) if current_level in levels else 0
    future = upgrades[idx + 1:] if idx < len(upgrades) - 1 else []

    return {
        "current_level": current_level,
        "future_upgrades": future,
        "all_levels": upgrades,
        "assumptions": [
            "L0 = 基础编录表 (深度+倾角+倾向), 最低可工作输入",
            "带收窄幅度基于 FORGE 验证, 实际效果因场地而异",
            "升档建议为筛查级, 需结合具体工程需求",
        ],
    }


# ============================================================
# T14: 可辩护性四件套检查
# ============================================================

def defensibility_checklist(output_data: dict) -> dict:
    """可辩护性四件套检查 (T14).

    所有对外 JSON/报告必须包含:
        1. assumptions (非空)
        2. uncertainty_band
        3. reproducibility (复现命令)
        4. data_source + date

    参数:
        output_data: 待检查的输出 dict

    返回:
        checklist: {items: [{name, present, content}], all_pass, missing: [str]}
    """
    items = []

    # 1. assumptions
    assumptions = output_data.get("assumptions", [])
    items.append({
        "name": "assumptions (假设)",
        "present": len(assumptions) > 0 if isinstance(assumptions, list) else bool(assumptions),
        "content": assumptions if isinstance(assumptions, list) else [str(assumptions)],
    })

    # 2. uncertainty_band
    has_uncertainty = any(k in output_data for k in [
        "uncertainty_band", "modal_err_std", "ci_half_width",
        "p32_crit_lower", "p32_crit_upper", "confidence",
    ])
    items.append({
        "name": "uncertainty_band (不确定带)",
        "present": has_uncertainty,
        "content": {k: output_data[k] for k in [
            "uncertainty_band", "modal_err_std", "ci_half_width",
            "p32_crit_lower", "p32_crit_upper", "confidence",
        ] if k in output_data},
    })

    # 3. reproducibility
    repro = output_data.get("reproducibility", {})
    has_repro = bool(repro.get("command") or repro.get("environment"))
    items.append({
        "name": "reproducibility (复现命令)",
        "present": has_repro,
        "content": repro,
    })

    # 4. data_source + date
    has_source = any(k in output_data for k in [
        "data_source", "source", "date", "data_date",
    ])
    items.append({
        "name": "data_source + date (数据来源与日期)",
        "present": has_source,
        "content": {k: output_data[k] for k in [
            "data_source", "source", "date", "data_date",
        ] if k in output_data},
    })

    missing = [it["name"] for it in items if not it["present"]]
    all_pass = len(missing) == 0

    return {
        "items": items,
        "all_pass": all_pass,
        "missing": missing,
        "honest_note": (
            "可辩护性是高放处置类客户的第一筛选条件。"
            "任何对外数字必须带假设 + 不确定带 + 复现命令 + 来源日期。"
        ),
    }
