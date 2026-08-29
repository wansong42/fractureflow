# -*- coding: utf-8 -*-
"""连通性筛查报告生成器: pipeline JSON → Markdown 报告.

W4 交付物: 将 dfn_from_borehole.py 的 JSON 输出转化为客户可看的 Markdown 报告.
遵循防坑清单 H1: 报告模板强制 assumptions 字段非空.

============================================================================
T27 扩展: 行业规范条文级对齐 + 多行业 HTML 报告模板系统
============================================================================

新增函数:
  - load_template(code: str) -> dict
      加载行业模板 JSON 配置
  - render_report(data: dict, template_code: str) -> str
      按模板生成完整 HTML 报告
  - render_report_to_file(data: dict, template_code: str, output_path: str) -> str
      渲染并保存 HTML 到文件

用法 (Python):
    from fractureflow.report import render_report_to_file
    data = {
        "n_wells": 22,
        "n_fractures": 880,
        "K": 4,
        "nrm": normals_array,      # (N, 3)
        "assign": assign_array,    # (N,)
        "centers": centers_array,  # (K, 3)
        "site_name": "北山预选区",
        "project_name": "BS34 井",
    }
    render_report_to_file(data, "railway", "output/report.html")

用法 (CLI):
    python -m fractureflow.eval --report-with-template railway --project-name "BS34"
    python -m fractureflow.eval --report-with-template hydropower \\
        --set-table data/my_set_table.pt --output reports/HS34.html

模板文件位置:
    src/fractureflow/templates/universal.json  (GB 50021 通用岩土)
    src/fractureflow/templates/railway.json     (TB 10012 铁路)
    src/fractureflow/templates/highway.json     (JTG C20 公路)
    src/fractureflow/templates/hydropower.json  (GB 50487 水电)
"""

import json
import os
import numpy as np


def _dir_to_dip_azimuth(dir_vec):
    """将法向 (nx, ny, nz) 转为倾向倾角 (dip, dip_direction)."""
    dir_vec = np.asarray(dir_vec, float)
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)
    nx, ny, nz = dir_vec
    # dip = angle from horizontal = acos(|nz|)
    dip = float(np.degrees(np.arccos(np.clip(abs(nz), 0, 1))))
    # dip_direction = azimuth of steepest descent
    # For a plane with normal n, dip direction is the azimuth of the projection
    dip_dir = float(np.degrees(np.arctan2(nx, ny)) % 360)
    return dip, dip_dir


def _fmt_num(v, spec=".1f", na="N/A"):
    """安全数值格式化: None / 非数值 → na (防坑: 各向同性场景角度为 None,
    旧写法 f"{None:.1f}" 直接 TypeError 炸掉客户报告, T91 同类规则的延伸)."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        try:
            import math
            if math.isnan(v) or math.isinf(v):
                return na
        except TypeError:
            return na
        return format(v, spec)
    return na


def generate_report(data: dict, site_name: str = "场地") -> str:
    """从管线 JSON 数据生成 Markdown 报告.

    参数:
        data: dfn_from_borehole.py 的输出 dict
        site_name: 场地名称 (用于报告标题)

    返回:
        Markdown 字符串
    """
    meta = data.get('meta', {})
    inp = data.get('input', {})
    st = data.get('set_table', {})
    p32_est = data.get('p32_estimate', {})
    perc = data.get('percolation', {})
    scenario = data.get('scenario_metrics', {})
    honest = data.get('honest_boundary', {})

    lines = []
    lines.append(f"# 裂隙网络连通性筛查报告")
    lines.append(f"")
    lines.append(f"**场地**: {site_name}  ")
    lines.append(f"**生成时间**: {meta.get('timestamp', 'N/A')}  ")
    lines.append(f"**管线版本**: {meta.get('pipeline', 'N/A')}  ")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## ⚠ 诚实边界 (重要)")
    lines.append(f"")
    lines.append(f"本报告是**筛查级结论 + 不确定性区间**，**不是连通性判定**。")
    lines.append(f"")

    # 强制 assumptions 非空 (防坑 H1)
    assumptions_list = []
    if honest.get('screening_level'):
        assumptions_list.append("- 筛查级结论，非真实连通性判定")
    if honest.get('no_aperture'):
        assumptions_list.append(f"- {honest['no_aperture']}")
    if honest.get('size_distribution_unknown'):
        assumptions_list.append(f"- {honest['size_distribution_unknown']}")
    if perc.get('assumptions'):
        assumptions_list.append(f"- {perc['assumptions']}")

    if not assumptions_list:
        # H1 防线: 不允许空 assumptions
        assumptions_list.append("- [错误: assumptions 字段为空，请检查管线输出]")

    for a in assumptions_list:
        lines.append(a)
    lines.append(f"")

    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 1. 输入数据")
    lines.append(f"")
    lines.append(f"| 参数 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 裂隙总数 | {inp.get('N_fractures', 'N/A')} |")
    lines.append(f"| 组数 K | {st.get('K', 'N/A')} |")
    lines.append(f"| 域尺寸 | {inp.get('domain_m', 'N/A')} m |")
    lines.append(f"")

    # 组系表
    lines.append(f"## 2. 裂隙组系表 (SetTable)")
    lines.append(f"")
    lines.append(f"| 组 | 法向 (nx, ny, nz) | 倾向° | 倾角° | 浓度 κ | 占比 |")
    lines.append(f"|-----|---------------------|-------|--------|--------|------|")

    centers = st.get('centers', [])
    concentrations = st.get('concentrations', [])
    proportions = st.get('proportions', [])
    total_prop = sum(proportions) if proportions else 1.0

    for k in range(len(centers)):
        c = centers[k]
        kappa = concentrations[k] if k < len(concentrations) else 0
        prop = proportions[k] if k < len(proportions) else 0
        dip, dip_dir = _dir_to_dip_azimuth(c)
        lines.append(f"| {k} | ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}) | {dip_dir:.1f} | {dip:.1f} | {kappa:.1f} | {prop:.1%} |")

    lines.append(f"")
    lines.append(f"**说明**: 倾向/倾角由法向换算 (右手法则)。浓度 κ 越大组内越集中。")
    lines.append(f"")

    # P32 估计
    lines.append(f"## 3. 强度估计 (P32)")
    lines.append(f"")
    lines.append(f"| 分位 | P32 (m²/m³) |")
    lines.append(f"|------|--------------|")
    lines.append(f"| P10 | {p32_est.get('p32_p10', 0):.4f} |")
    lines.append(f"| P50 | {p32_est.get('p32_p50', 0):.4f} |")
    lines.append(f"| P90 | {p32_est.get('p32_p90', 0):.4f} |")
    lines.append(f"")
    lines.append(f"**假设**: {p32_est.get('assumptions', 'N/A')}")
    lines.append(f"")

    # Phase D: 数据升档信息 (向后兼容: 旧版 JSON 可能无此字段)
    phase_d = data.get('phase_d_upgrade') or {}
    if phase_d.get('beta_fitted') is not None:
        beta_info = phase_d.get('beta_fit_info', {})
        lines.append(f"### 数据升档: 迹长 → β 拟合")
        lines.append(f"")
        lines.append(f"| 参数 | 值 |")
        lines.append(f"|------|-----|")
        beta_val = round(phase_d['beta_fitted'], 2)
        beta_std = round(beta_info.get('beta_std', 0), 2)
        lines.append(f"| β (拟合) | {beta_val} ± {beta_std} |")
        lines.append(f"| 样本数 | {beta_info.get('n_samples', 'N/A')} |")
        lines.append(f"| 方法 | {beta_info.get('method', 'N/A')} |")
        lines.append(f"")

    if phase_d.get('spacing_p32') is not None:
        sp_info = phase_d.get('spacing_info', {})
        lines.append(f"### 数据升档: 间距 → P32 直接估计")
        lines.append(f"")
        lines.append(f"| 参数 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| P32 (间距) | {phase_d['spacing_p32']:.4f} |")
        lines.append(f"| 间距均值 | {sp_info.get('spacing_mean', 0):.2f} m |")
        lines.append(f"| 方法 | {sp_info.get('method', 'N/A')} |")
        lines.append(f"")

    # 渗流曲线
    lines.append(f"## 4. 渗流阈值")
    lines.append(f"")
    has_perc = bool(perc) and perc.get('p32_crit') is not None
    if not has_perc:
        # H1 防线延伸: 缺渗流字段时不得渲染成 "P32_crit=0.000 大概率渗流"
        lines.append(f"| 参数 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| **P32_crit** | **N/A (输入 JSON 未含渗流结果, 无法给出结论)** |")
        lines.append(f"")
    else:
        p32_crit = perc['p32_crit']
        p32_crit_lo = perc.get('p32_crit_lower')
        p32_crit_hi = perc.get('p32_crit_upper')
        beta = perc.get('beta', 'N/A')
        n_real = perc.get('n_realizations_per_point', 'N/A')

        lines.append(f"| 参数 | 值 |")
        lines.append(f"|------|-----|")
        if isinstance(beta, float):
            beta_str = f'{beta:.2f}'
        else:
            beta_str = str(beta)
        lines.append(f"| 幂律指数 β | {beta_str} |")
        lines.append(f"| 每 P32 点实现数 | {n_real} |")
        lines.append(f"| **P32_crit** | **{p32_crit:.3f}** |")
        lo_str = _fmt_num(p32_crit_lo, ".3f") if p32_crit_lo is not None else "N/A"
        hi_str = _fmt_num(p32_crit_hi, ".3f") if p32_crit_hi is not None else "N/A"
        lines.append(f"| 不确定带 | [{lo_str}, {hi_str}] |")
        lines.append(f"")
        lines.append(f"**解读**: P32_crit 是网络从'不连通'到'连通'的临界强度。")
        hi_note = (f"(不确定带上界 {hi_str} 对应 p_conn≈0.9 的交叉点)"
                   if p32_crit_hi is not None else "")
        lines.append(f"当实测 P32 > {p32_crit:.3f} 时，网络大概率渗流 {hi_note}。")
        lines.append(f"注: 不确定带来自多实现 [p_conn=0.1, 0.9] 交叉带, **非统计学 95% 置信区间**。")
        lines.append(f"")

    # β sweep
    beta_sweep = perc.get('beta_sweep', {}) if perc else {}
    if len(beta_sweep) > 1:
        lines.append(f"| β | P32_crit | 不确定带 |")
        lines.append(f"|---|----------|----------|")
        for key, val in beta_sweep.items():
            lo = val.get('p32_crit_lower', 0)
            hi = val.get('p32_crit_upper', 0)
            lines.append(f"| {key} | {val.get('p32_crit', 0):.3f} | [{lo:.3f}, {hi:.3f}] |")
        lines.append(f"")

    # 场景指标
    lines.append(f"## 5. 场景化指标")
    lines.append(f"")

    egs = scenario.get('egs', {})
    mine = scenario.get('mine', {})
    disp = scenario.get('disposal', {})

    if egs:
        lines.append(f"### 5.1 地热 (EGS)")
        lines.append(f"")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 连通分数 | {egs.get('connectivity_fraction', 0):.3f} |")
        lines.append(f"| 井对方向 | {egs.get('well_pair_axis', 'N/A')} |")
        lines.append(f"| 与井对夹角 | {_fmt_num(egs.get('angle_to_well_pair_deg'))}° (面-轴; N/A=各向同性) |")
        lines.append(f"| 评价 | {egs.get('assessment', 'N/A')} |")
        lines.append(f"| 不确定带 | {egs.get('uncertainty_band', 'N/A')} |")
        lines.append(f"")

    if mine:
        lines.append(f"### 5.2 矿山/隧洞 (突水)")
        lines.append(f"")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 连通分数 | {mine.get('connectivity_fraction', 0):.3f} |")
        lines.append(f"| 巷道轴向 | {mine.get('tunnel_axis', 'N/A')} |")
        lines.append(f"| 与巷道夹角 | {_fmt_num(mine.get('angle_to_tunnel_deg'))}° (面-轴; N/A=各向同性) |")
        lines.append(f"| 风险等级 | {mine.get('risk_level', 'N/A')} |")
        lines.append(f"| 不确定带 | {mine.get('uncertainty_band', 'N/A')} |")
        lines.append(f"")

    if disp:
        lines.append(f"### 5.3 核废料处置 (逃逸)")
        lines.append(f"")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 连通分数 | {disp.get('connectivity_fraction', 0):.3f} |")
        lines.append(f"| 垂直方向 | {disp.get('vertical_direction', 'N/A')} |")
        lines.append(f"| 与垂直夹角 | {_fmt_num(disp.get('angle_to_vertical_deg'))}° (面-轴; N/A=各向同性) |")
        lines.append(f"| 逃逸优先级 | {disp.get('escape_priority', 'N/A')} |")
        lines.append(f"| 不确定带 | {disp.get('uncertainty_band', 'N/A')} |")
        lines.append(f"")

    # 数据升档建议 (Phase D tie-in)
    lines.append(f"## 6. 数据升档建议")
    lines.append(f"")
    lines.append(f"当前结论的不确定性主要来自尺寸分布未知。以下数据可收窄不确定带:")
    lines.append(f"")
    lines.append(f"| 升级项 | 效果 | 对应带收窄 |")
    lines.append(f"|--------|------|------------|")
    lines.append(f"| 露头/岩心迹长 | β 可拟合 | P32_crit 带收窄 ~30% |")
    lines.append(f"| 延伸长 | 最大半径约束 | P32_crit 带收窄 ~20% |")
    lines.append(f"| 开度 | Oda 绝对渗透解锁 | 可输出绝对渗透量级 |")
    lines.append(f"| 间距 | 面密度直接算 | P32 估计不再需要 |")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    elapsed = meta.get('elapsed_sec', '')
    lines.append(f"*本报告由 fractureflow v3 连通性模块自动生成。耗时 {elapsed}s*")

    return "\n".join(lines)


def generate_report_from_file(json_path: str, output_path: str = None,
                               site_name: str = "场地") -> str:
    """从 JSON 文件生成报告并保存.

    参数:
        json_path: 管线输出 JSON 路径
        output_path: 报告输出路径 (默认同名 .md)
        site_name: 场地名称

    返回:
        Markdown 字符串
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    md = generate_report(data, site_name)

    if output_path is None:
        output_path = os.path.splitext(json_path)[0] + "_report.md"

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)

    return md


# ===========================================================================
# T27: 行业规范模板系统  —— 条文级对齐 + 多行业 HTML 报告生成
# ===========================================================================

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# 行业代码 → JSON 文件名 映射
TEMPLATE_MAP = {
    "universal": "universal.json",
    "railway": "railway.json",
    "highway": "highway.json",
    "hydropower": "hydropower.json",
}


def load_template(code: str) -> dict:
    """加载行业模板配置 JSON.

    参数:
        code: 行业代码 (universal / railway / highway / hydropower)

    返回:
        模板配置 dict

    用法:
        tmpl = load_template("railway")
        print(tmpl["code"])  # "TB 10012-2019 J124-2019"
    """
    if code not in TEMPLATE_MAP:
        raise ValueError(
            f"未知模板代码: {code}. 可选: {list(TEMPLATE_MAP.keys())}"
        )
    path = os.path.join(TEMPLATES_DIR, TEMPLATE_MAP[code])
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _dir_to_dip_azimuth_local(dir_vec):
    """将法向 (nx, ny, nz) 转为 (dip, dip_direction).
    与 _dir_to_dip_azimuth 同口径 (无向法向).
    """
    dir_vec = np.asarray(dir_vec, float)
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)
    nx, ny, nz = dir_vec
    dip = float(np.degrees(np.arccos(np.clip(abs(nz), 0, 1))))
    dip_dir = float(np.degrees(np.arctan2(nx, ny)) % 360)
    return dip, dip_dir


def _build_group_rows(nrm, assign, centers, K):
    """根据法向/指派/组心构建组系表行列表 (供 HTML 渲染用).

    返回 list[dict], 每个 dict = 一行, 含 group_id / n_fractures /
    mean_dip_dir / mean_dip / mean_strike / development_grade.
    """
    nrm = np.asarray(nrm, dtype=np.float64)
    assign = np.asarray(assign, dtype=int)
    centers = np.asarray(centers, dtype=np.float64)
    n_total = max(len(nrm), 1)
    rows = []
    for k in range(K):
        mask = assign == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        if k < len(centers) and np.linalg.norm(centers[k]) > 0.1:
            c = centers[k] / (np.linalg.norm(centers[k]) + 1e-12)
            dip, dip_dir = _dir_to_dip_azimuth_local(c)
        else:
            dip = dip_dir = 0.0
        strike = (dip_dir - 90.0) % 360.0
        # 定性发育程度 (按占比)
        pct = n_k / n_total * 100.0
        if pct >= 20.0:
            grade = "很发育"
        elif pct >= 10.0:
            grade = "发育"
        elif pct >= 5.0:
            grade = "较发育"
        else:
            grade = "不发育"
        rows.append({
            "group_id": k + 1,
            "n_fractures": n_k,
            "mean_dip_dir": round(dip_dir, 1),
            "mean_dip": round(dip, 1),
            "mean_strike": round(strike, 1),
            "proportion": round(pct, 1),
            "development_grade": grade,
        })
    return rows


_DEV_TERMS_STD = ["不发育", "较发育", "发育", "很发育"]


def _development_summary(rows, qualitative_terms=None):
    """根据组数生成发育程度定性评价文字.

    术语来源: 固定标准四级词表 (与 HTML 图例表一致).
    历史 bug (2026-08-28 修复): 曾直接消费模板 qualitative_terms ——
    hydropower 模板该字段是**间距描述** [">2m (稀疏)", ...], 会生成
    "共识别 N 组结构面, 属<0.5m (很密集)" 这类病句.
    """
    n_groups = len(rows)
    terms = _DEV_TERMS_STD
    if n_groups >= 4:
        level = terms[-1]
        desc = f"共识别 {n_groups} 组结构面，属{level}。"
    elif n_groups == 3:
        level = terms[-2]
        desc = f"共识别 {n_groups} 组结构面，属{level}。"
    elif n_groups == 2:
        level = terms[-3]
        desc = f"共识别 {n_groups} 组结构面，属{level}。"
    else:
        level = terms[0]
        desc = f"仅识别 {n_groups} 组结构面，属{level}。"
    return desc


def render_report(data: dict, template_code: str) -> str:
    """按行业模板生成完整 HTML 报告.

    参数:
        data: 报告数据 dict, 必须包含:
            - n_wells (int): 井数
            - n_fractures (int): 总裂隙数
            - K (int): 组数
            - nrm: (N, 3) 法向数组
            - assign: (N,) 组指派
            - centers: (K, 3) 组心
            - site_name (str, optional): 场地名称
        可选:
            - chart_rose (str): 玫瑰图路径
            - chart_stereonet (str): 极点图路径
            - project_name (str): 项目名称
            - honesty_note (str): 覆盖模板的诚实边界文字
        template_code: 行业代码 (universal / railway / highway / hydropower)

    返回:
        HTML 字符串

    用法:
        html = render_report(data, "railway")
        with open("report.html", "w", encoding="utf-8") as f:
            f.write(html)
    """
    tmpl = load_template(template_code)
    site_name = data.get("site_name", "场地")
    project_name = data.get("project_name", site_name)
    n_wells = data.get("n_wells", 1)
    n_fractures = data.get("n_fractures", 0)
    K = data.get("K", 0)
    nrm = np.asarray(data.get("nrm", []), dtype=np.float64)
    assign = np.asarray(data.get("assign", []), dtype=int)
    centers = np.asarray(data.get("centers", []), dtype=np.float64)

    # 构建组系表行
    group_rows = []
    if len(nrm) > 0 and len(assign) > 0 and K > 0:
        group_rows = _build_group_rows(nrm, assign, centers, K)

    # 定性评价
    dev_summary = _development_summary(group_rows, tmpl.get("qualitative_terms", {}))

    # 日期
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d")

    # 诚实边界 (优先用 data 覆盖, 否则用模板)
    honesty_note = data.get("honesty_note", tmpl.get("honesty_note", ""))

    # 图表 HTML (如果有路径则内嵌 base64)
    chart_html_blocks = []
    chart_rose = data.get("chart_rose", "")
    chart_stereo = data.get("chart_stereonet", "")

    def _img_to_base64(path):
        import base64
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
        return ""

    rose_b64 = _img_to_base64(chart_rose)
    stereo_b64 = _img_to_base64(chart_stereo)

    charts_section = ""
    if rose_b64 or stereo_b64:
        charts_section = '<div class="charts-section">'
        if rose_b64:
            charts_section += (
                f'<div class="chart-img">'
                f'<img src="data:image/png;base64,{rose_b64}" alt="玫瑰图"/>'
                f'<p class="chart-caption">走向玫瑰图</p></div>'
            )
        if stereo_b64:
            charts_section += (
                f'<div class="chart-img">'
                f'<img src="data:image/png;base64,{stereo_b64}" alt="极点图"/>'
                f'<p class="chart-caption">极点图 (立体投影)</p></div>'
            )
        charts_section += '</div>'

    # 组系表 HTML
    headers_cn = tmpl.get("table_headers_cn", ["组号", "条数", "倾向(°)", "倾角(°)"])
    header_to_field = {
        "组号": "group_id",
        "条数": "n_fractures",
        "倾向(°)": "mean_dip_dir",
        "优势倾向(°)": "mean_dip_dir",
        "倾角(°)": "mean_dip",
        "优势倾角(°)": "mean_dip",
        "走向(°)": "mean_strike",
        "发育程度": "development_grade",
        "充填特征": "filling",
        "占比(%)": "proportion",
        "隙宽(mm)": "aperture",
        "隙长": "aperture",
        "粗糙度": "roughness",
        "迹长(m)": "trace_length",
        "间距(m)": "spacing",
        "张开度(mm)": "aperture",
        "充填物": "filling",
        "组数": "n_fractures",
        "充填物特征": "filling",
        "Dip Dir(°)": "mean_dip_dir",
        "Dip(°)": "mean_dip",
        "Trend(°)": "mean_strike",
        "Strike(°)": "mean_strike",
        "Set": "group_id",
        "N": "n_fractures",
        "Development": "development_grade",
        "Filling": "filling",
        "Aperture(mm)": "aperture",
        "Aperture": "aperture",
        "Spacing(m)": "spacing",
        "Roughness": "roughness",
        "Trace(m)": "trace_length",
        "Count": "n_fractures",
        "Plunge(°)": "mean_dip",
    }
    # 构建表格列 (仅显示数据中存在的列)
    active_headers = []
    for h in headers_cn:
        field = header_to_field.get(h, "")
        if field == "group_id" or field == "n_fractures":
            active_headers.append((h, field))
        elif group_rows and field in group_rows[0]:
            active_headers.append((h, field))

    table_header_html = "".join(f"<th>{h}</th>" for h, _ in active_headers)

    # 通用列构建
    table_html = f"""
    <table class="group-table">
      <thead><tr>{table_header_html}</tr></thead>
      <tbody>
    """
    for row in group_rows:
        table_html += "<tr>"
        for _, field in active_headers:
            val = row.get(field, "")
            table_html += f"<td>{val}</td>"
        table_html += "</tr>"
    table_html += "</tbody></table>"

    # sections 列表
    sections = tmpl.get("sections", ["概述", "结构面分组统计", "结论建议"])

    # 按 sections 顺序组织 HTML
    sections_html = []
    for sec in sections:
        if sec in ("概述",):
            sections_html.append(f"""
            <section class="report-section">
              <h2>{sec}</h2>
              <table class="info-table">
                <tr><th>项目</th><td>{project_name}</td></tr>
                <tr><th>钻孔数</th><td>{n_wells}</td></tr>
                <tr><th>裂隙总数</th><td>{n_fractures}</td></tr>
                <tr><th>识别组数 K</th><td>{K}</td></tr>
                <tr><th>报告日期</th><td>{ts}</td></tr>
              </table>
            </section>""")
        elif sec in ("地质条件", "工程地质条件", "岩体结构"):
            sections_html.append(f"""
            <section class="report-section">
              <h2>{sec}</h2>
              <p>场地共发育 {K} 组结构面，总计 {n_fractures} 条裂隙，分布在 {n_wells} 个钻孔中。
              各组优势产状详见"结构面分组统计"章节。</p>
            </section>""")
        elif sec in ("结构面分组统计", "节理统计表", "裂隙统计表"):
            sections_html.append(f"""
            <section class="report-section">
              <h2>{sec}</h2>
              {table_html}
            </section>""")
        elif sec in ("结构面发育程度评价",):
            sections_html.append(f"""
            <section class="report-section">
              <h2>{sec}</h2>
              <p class="dev-summary">{dev_summary}</p>
              <table class="info-table">
                <tr><th>发育等级</th><th>特征</th></tr>
                <tr><td>不发育</td><td>1~2 组规则节理，一般延伸长度 &lt;3m，多闭合</td></tr>
                <tr><td>较发育</td><td>2~3 组规则节理，延伸长度 &lt;10m，多闭合或细脉充填</td></tr>
                <tr><td>发育</td><td>规则节理多于 3 组，延伸长度多超过 10m，风化者多张开、夹泥</td></tr>
                <tr><td>很发育</td><td>规则节理多于 3 组，裂隙多张开、夹泥，有延伸较长的大裂隙</td></tr>
              </table>
            </section>""")
        elif sec in ("隧道围岩分级相关评价", "岩体结构类型判定"):
            sections_html.append(f"""
            <section class="report-section">
              <h2>{sec}</h2>
              <p>结构面状态对围岩分级影响显著。共识别 {K} 组结构面:
              {dev_summary}
              </p>
              <p class="honest-warning">⚠ 具体围岩分级应结合岩石强度、完整程度、地下水等指标综合判定。</p>
            </section>""")
        elif sec in ("软弱结构面分类",):
            sections_html.append(f"""
            <section class="report-section">
              <h2>{sec}</h2>
              <p>依据 GB 50487 附录 V, 硬性结构面与软弱结构面应区分并分别给出抗剪断强度参数。</p>
              <p class="honest-warning">⚠ 本报告暂无结构面强度数据；如需抗滑稳定评价，请补充充填物类型、厚度及粗糙度信息。</p>
            </section>""")
        elif sec in ("结论建议", "结论与建议"):
            sections_html.append(f"""
            <section class="report-section">
              <h2>{sec}</h2>
              <ul>
                <li>场地共发育 {K} 组结构面，总计 {n_fractures} 条。</li>
                <li>发育程度评价: {dev_summary}</li>
                <li>优势产状请参见组系表与玫瑰图。</li>
                <li>详细工程评价应结合地质师现场判断与规范要求。</li>
              </ul>
            </section>""")
    
    all_sections_html = "\n".join(sections_html)

    # 完整 HTML 组装
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>{tmpl.get('header_text', '结构面统计报告')} - {project_name}</title>
<style>
  body {{ font-family: "Microsoft YaHei", "SimHei", sans-serif; margin: 40px; color: #222; line-height: 1.7; }}
  h1 {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 12px; }}
  h2 {{ color: #1a5fa8; border-left: 4px solid #1a5fa8; padding-left: 10px; margin-top: 30px; }}
  .subtitle {{ text-align: center; color: #555; margin-bottom: 30px; }}
  .report-section {{ margin-bottom: 25px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: center; font-size: 14px; }}
  th {{ background: #1a5fa8; color: white; }}
  tbody tr:nth-child(even) {{ background: #f5f8fc; }}
  .info-table th {{ width: 160px; text-align: left; background: #4a7ab5; }}
  .info-table td {{ text-align: left; }}
  .charts-section {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin: 20px 0; }}
  .chart-img {{ text-align: center; }}
  .chart-img img {{ max-width: 480px; max-height: 480px; border: 1px solid #ddd; }}
  .chart-caption {{ font-size: 13px; color: #666; margin-top: 4px; }}
  .dev-summary {{ background: #eef5fc; border-left: 4px solid #1a5fa8; padding: 12px 16px; font-size: 15px; margin: 10px 0; }}
  .honest-warning, .honesty-box {{ background: #fff8e1; border: 1px solid #f9a825; border-radius: 6px; padding: 14px 18px; font-size: 14px; color: #5d4037; margin: 14px 0; }}
  .honesty-box h3 {{ margin: 0 0 8px 0; color: #e65100; font-size: 15px; }}
  .footer {{ margin-top: 40px; padding-top: 12px; border-top: 1px solid #ccc; text-align: center; color: #888; font-size: 12px; }}
  .cover-info {{ text-align: center; margin: 30px 0; }}
  .cover-info .label {{ color: #666; font-size: 14px; }}
  .cover-info .value {{ font-size: 18px; font-weight: bold; margin: 4px 0 20px 0; }}
</style>
</head>
<body>

<!-- ========== 封面 ========== -->
<h1>{tmpl.get('header_text', '结构面统计报告')}</h1>
<div class="cover-info">
  <div class="label">项目名称</div>
  <div class="value">{project_name}</div>
  <div class="label">报告日期</div>
  <div class="value">{ts}</div>
  <div class="label">依据规范</div>
  <div class="value">{tmpl.get('code', '')} — {tmpl.get('name', '')}</div>
</div>

{all_sections_html}

<!-- ========== 可视化 ========== -->
<section class="report-section">
  <h2>可视化图件</h2>
  {charts_section if charts_section else '<p class="honest-warning">暂无图表数据。请使用 borehole_report.py 生成玫瑰图与极点图后内嵌。</p>'}
</section>

<!-- ========== 诚实边界 (强制, 不可关闭) ========== -->
<section class="report-section honesty-box">
  <h3>⚠ 重要说明 (必读)</h3>
  <p>{honesty_note}</p>
  <ul>
    <li>组系划分基于法向几何聚类 (球面 k-means)，非地质成因分组。</li>
    <li>法向基于钻孔产状测量，可能存在测量误差。</li>
    <li>本报告供初步参考，详细评价应结合地质师现场判断。</li>
  </ul>
</section>

<!-- ========== 页脚 ========== -->
<div class="footer">
  {tmpl.get('footer_text', '自动生成 - FractureFlow')}
</div>

</body>
</html>"""
    return html


def render_report_to_file(data: dict, template_code: str, output_path: str) -> str:
    """渲染报告并保存到文件.

    参数:
        data: render_report 所需数据
        template_code: 行业代码
        output_path: HTML 输出路径

    返回:
        HTML 字符串
    """
    html = render_report(data, template_code)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return html
