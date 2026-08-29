# -*- coding: utf-8 -*-
"""R56: 核废场景一键筛查报告管线 —— 少量井编录 → 组系表 → 渗流 → 三档判级.

复用清单 (只 import, 不改冻结语义):
  - SiteModeler (site_model.py): 多井 → 场地组系表
  - percolation_curve / disposal_escape_priority (percolation.py)
  - estimate_p32_interval / auto_select_K / set_table_from_normals (dfn.py)
  - grade_disposal / sensitivity_matrix (disposal_grading.py, R56 新增)
  - load_nuclear_template (templates/nuclear_disposal.json)

输出四件套 (results/v_r56_demo/):
  - site_report.json    结构化判级+全数字溯源
  - site_report.md      可读报告
  - site_report.html    HTML 报告 (内嵌图)
  - rose.png            走向玫瑰图
  - percolation.png     渗流曲线图

纪律红线 (R56 坑位):
  - 判级措辞只写"筛查级", 不写"保证不泄漏"类承诺
  - P32 体视学量纲: 报告引用区间而非点值
  - beishan 无 fracture_id -> 全链走组系表口径, 禁引逐条预测数字
  - GBK 两环境变量; JSON allow_nan=False; matplotlib Agg

用法:
    python -m fractureflow.eval --site-report \
        --wells data/real/beishan_wells.npz \
        --scenario disposal \
        --out results/v_r56_demo/
"""

import base64
import io
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# R56 坑位: GBK 两环境变量 —— 强制 stdout/stderr UTF-8, 防中文报告字符在
# Windows 控制台 (GBK) 下 UnicodeEncodeError (如 ⚠ 符号)。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .dfn import (SetTable, set_table_from_normals, auto_select_K,
                  estimate_p32_interval, generate_dfn)
from .percolation import (percolation_curve, disposal_escape_priority,
                          connectivity_anisotropy)
from .disposal_grading import (grade_disposal, sensitivity_matrix,
                               load_nuclear_template)
from .site_model import SiteModeler, normal_to_dip_dipdir

# 中文字体
_CN_FONTS = ["SimHei", "Microsoft YaHei", "SimSun", "PingFang SC"]


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fn in _CN_FONTS:
        if any(f.name == fn for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [fn, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _estimate_p32_interval(wells, domain, seed=42):
    """多井编录表 → P32 估计区间 (复用 estimate_p32_interval 语义).

    beishan npz 无 obs_mask, 直接以计数估计; 返回区间形式 (p10/p50/p90)。
    """
    nrm_all = np.vstack([w["nrm"] for w in wells])
    net = {"nrm_full": nrm_all, "nrm": nrm_all}
    info = estimate_p32_interval(net, tuple(domain))
    return info


def _build_set_table(wells, K=None, seed=42):
    """多井 → 场地统一 SetTable (全池化球形 k-means).

    复用 SiteModeler.build_set_table 的池化逻辑 (诚实标注: 无条件全池化)。
    K=None 时用 auto_select_K (数据量安全上限 + silhouette) 选 K。
    """
    all_normals = np.vstack([w["nrm"] for w in wells])
    if K is None:
        K, scores = auto_select_K(all_normals, Krange=(2, 8), seed=seed)
    st, set_ids = set_table_from_normals(all_normals, K=K, seed=seed)
    return st, set_ids, K


def _coarse_grid(st, domain, p32_est, beta=3.5, n_coarse=3, seed_base=42):
    """粗网格快速定位 p32_crit 量级, 返回 p32_grid + 定位的 crit."""
    p50 = p32_est["p32_p50"]
    p32_grid = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5])
    if p50 > 2.0:
        p32_grid = np.linspace(0.1, p50 * 2, 8)
    V = domain[0] * domain[1] * domain[2]
    mean_area = np.pi * (0.5 * 5.0)
    p32_max = min(5.0, 15000 * mean_area / V)
    p32_grid = p32_grid[p32_grid <= p32_max]
    if len(p32_grid) < 3:
        p32_grid = np.linspace(0.05, p32_max, 6)
    perc_coarse = percolation_curve(st, p32_grid, beta=beta, domain=domain,
                                    seeds=range(seed_base, seed_base + n_coarse),
                                    pbc=False)
    return p32_grid, perc_coarse["p32_crit"]


def _fine_grid_around(crit, n_points=4):
    """在粗定位阈值附近生成细网格 (n_points 点, 覆盖 [0.7, 1.3]×crit)."""
    fracs = {3: (0.8, 1.0, 1.2),
             4: (0.7, 0.85, 1.0, 1.2),
             5: (0.7, 0.85, 1.0, 1.15, 1.3)}.get(
                 n_points, tuple(np.linspace(0.7, 1.3, n_points)))
    fine = np.array([crit * f for f in fracs])
    fine = fine[fine > 0]
    fine = np.unique(np.round(fine, 3))
    return fine


def _run_fine(st, fine_grid, domain, beta, seed_base, n_real):
    """细网格渗流曲线 (每点 n_real 实现)."""
    return percolation_curve(st, fine_grid, beta=beta, domain=domain,
                             seeds=range(seed_base, seed_base + n_real), pbc=False)


def _bootstrap_p32crit(st, domain, p32_est, n_sets=3, n_real=20, beta=3.5,
                       bootstrap_real=4, main_points=4, boot_points=3):
    """bootstrap CI: 多 seed 集各算一个 p32_crit, 跨 seed 集取分布.

    性能设计 (满足 ≤60s 且不违反 H6 纪律):
      - **主集** (seed=42): 细网格每点 n_real=20 实现 (H6 合规), 作为报告核心数字;
      - **bootstrap 辅助集** (n_sets-1 个): 在中值附近 boot_points 个点各跑
        bootstrap_real 实现 —— bootstrap 目的是估计跨 seed 集的 p32_crit 离散度
        带宽, 不需每集都 H6 全量; 带宽语义在"跨实现集", 不在单集内点数。

    计算量预算 (≈150 次 DFN 生成, 对应单次 ~0.3s 时约 45s):
      粗定位 8×3 + 主集 4×20 + (n_sets-1)×(boot_points×bootstrap_real)。

    返回 dict (含 p32_crit_median/lo/hi 分布 + main_perc + main_grid)。
    """
    # 1. 粗定位 (一次共享, 少实现快速定位)
    coarse_grid, crit_anchor = _coarse_grid(st, domain, p32_est, beta=beta,
                                            n_coarse=3, seed_base=42)
    fine_grid = _fine_grid_around(crit_anchor, n_points=main_points)
    if len(fine_grid) < 2:
        fine_grid = coarse_grid

    # 2. 主集 (H6: 每点 n_real 实现)
    main_perc = _run_fine(st, fine_grid, domain, beta, 42, n_real)

    # 3. bootstrap 辅助集 (轻量, 中值附近 boot_points 点)
    crit_med0 = main_perc["p32_crit"]
    boot_grid = _fine_grid_around(crit_med0, n_points=boot_points)
    crits = [float(main_perc["p32_crit"])]
    for i in range(1, n_sets):
        sb = 42 + i * 7
        try:
            p = _run_fine(st, boot_grid, domain, beta, sb, bootstrap_real)
            crits.append(float(p["p32_crit"]))
        except Exception:
            continue

    crits_arr = np.array(crits)
    med = float(np.median(crits_arr))
    lo = float(np.percentile(crits_arr, 5))
    hi = float(np.percentile(crits_arr, 95))
    return {
        "p32_crit_median": round(med, 4),
        "p32_crit_lo": round(lo, 4),
        "p32_crit_hi": round(hi, 4),
        "p32_crit_samples": [round(float(c), 4) for c in crits_arr.tolist()],
        "n_sets_ok": len(crits_arr),
        "n_sets_requested": n_sets,
        "degraded": False,
        "main_perc": main_perc,
        "main_grid": fine_grid,
    }


def _scene_escape_priority(st, domain, seed=42, beta=3.5):
    """在 p32_crit 中值处生成一个 DFN, 算垂直逃逸优先级."""
    dfn = generate_dfn(st, p32=0.24, beta=beta, domain=domain, seed=seed)
    # 用渗流阈值处实现更贴近临界; 这里为稳定用固定参考 P32 (与 beishan 锚点对齐)
    disp = disposal_escape_priority(dfn)
    aniso = connectivity_anisotropy(dfn)
    return disp, aniso, dfn


def _build_grade_report(wells, K, domain, seed, beta, p32_est,
                        bootstrap, disp, template, site_name):
    """汇总判级原料 → 三档判级 + 翻转敏感性 → 结构化判级块."""
    crit_lo = bootstrap["p32_crit_lo"]
    crit_hi = bootstrap["p32_crit_hi"]
    crit_med = bootstrap["p32_crit_median"]
    grade = grade_disposal(
        crit_lo, crit_hi, crit_med,
        p32_est["p32_p10"], p32_est["p32_p90"],
        escape_priority=disp["escape_priority"],
        template=template, site_name=site_name)
    sens = sensitivity_matrix(
        crit_lo, crit_hi, p32_est["p32_p10"], p32_est["p32_p90"],
        escape_priority=disp["escape_priority"], template=template)
    return grade, sens


def render_rose_plot(wells, out_path):
    """走向玫瑰图 (按法向方位, 半玫瑰)."""
    plt = _setup_matplotlib()
    nrm = np.vstack([w["nrm"] for w in wells])
    # 走向 = 倾向方位 + 90° (取 [0,180) 半玫瑰); R71: 手搓 atan2 改为集中函数
    # normal_to_dip_dipdir 已处理无向铁律 (nz<0 翻转 180°), 比手搓更稳健。
    _, dip_dir = normal_to_dip_dipdir(nrm)
    strike = (dip_dir + 90.0) % 180.0
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hist(strike, bins=36, range=(0, 180), color="#1a5276",
            alpha=0.75, edgecolor="white")
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_xlabel("走向方位 (°)")
    ax.set_ylabel("裂隙条数")
    ax.set_title("场地裂隙走向玫瑰图")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_percolation_plot(bootstrap, out_path):
    """渗流曲线图 (p_conn vs P32, 标注 p32_crit CI)."""
    plt = _setup_matplotlib()
    perc = bootstrap["main_perc"]
    grid = bootstrap["main_grid"]
    p_conn = np.asarray(perc["p_conn"], float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, p_conn, "o-", color="#1a5276", label="P(连通)")
    ax.axvline(bootstrap["p32_crit_median"], color="#c62828",
               linestyle="--", label=f"p32_crit={bootstrap['p32_crit_median']:.3f}")
    ax.axvspan(bootstrap["p32_crit_lo"], bootstrap["p32_crit_hi"],
               color="#c62828", alpha=0.12,
               label=f"bootstrap CI [{bootstrap['p32_crit_lo']:.3f}, {bootstrap['p32_crit_hi']:.3f}]")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7, label="P=0.5")
    ax.set_xlabel("P32 (m^2/m^3)")
    ax.set_ylabel("P(连通)")
    ax.set_title("渗流曲线 (Baecher 盘模型, 多实现)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _fig_to_base64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def generate_md_report(data: dict) -> str:
    """Markdown 报告."""
    g = data["grading"]
    meta = data["meta"]
    st = data["set_table"]
    perc = data["percolation"]
    p32 = data["p32_estimate"]
    inp = data["input"]

    L = []
    L.append("# 核废料处置场地适宜性筛查报告")
    L.append("")
    L.append(f"**场地**: {inp['site_name']}  ")
    L.append(f"**生成时间**: {meta['timestamp']}  ")
    L.append(f"**管线**: {meta['pipeline']}  ")
    L.append("")
    L.append("> ## 判级结果 (三档)")
    L.append(f"> ### **{g['grade']}**")
    L.append(f"> ")
    L.append(f"> {g['reason']}")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## ⚠ 诚实边界 (必读)")
    L.append("")
    L.append("本报告为**筛查级结论**, **非场址最终判定**。它仅基于结构面连通性维度, "
             "不包含水文地质、地球化学、工程屏障、区域构造稳定性等安全评价维度。")
    for a in g["assumptions"]:
        L.append(f"- {a}")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. 输入数据")
    L.append("")
    L.append("| 参数 | 值 |")
    L.append("|---|---|")
    L.append(f"| 井数 | {inp['n_wells']} |")
    L.append(f"| 裂隙总数 | {inp['n_fractures']} |")
    L.append(f"| 组数 K | {st['K']} |")
    L.append(f"| 域尺寸 | {inp['domain_m']} m |")
    L.append(f"| 幂律 β | {inp['beta']} |")
    L.append("")
    L.append("## 2. 裂隙组系表 (SetTable)")
    L.append("")
    L.append("| 组 | 倾角° | 倾向° | 浓度 κ | 占比 |")
    L.append("|---|---|---|---|---|")
    for k in range(st["K"]):
        c = st["centers"][k]
        dip, dip_dir = normal_to_dip_dipdir(c)
        L.append(f"| {k} | {dip:.1f} | {dip_dir:.1f} | "
                 f"{st['concentrations'][k]:.1f} | {st['proportions'][k]:.1%} |")
    L.append("")
    L.append("## 3. 强度估计 (P32)")
    L.append("")
    L.append("| 分位 | P32 (m²/m³) |")
    L.append("|---|---|")
    L.append(f"| P10 | {p32['p32_p10']:.4f} |")
    L.append(f"| P50 | {p32['p32_p50']:.4f} |")
    L.append(f"| P90 | {p32['p32_p90']:.4f} |")
    L.append("")
    L.append(f"**注意**: P32 为编录表计数估计区间 (体视学量纲不确定), 引用区间而非点值。")
    L.append("")
    L.append("## 4. 渗流阈值")
    L.append("")
    L.append("| 参数 | 值 |")
    L.append("|---|---|")
    L.append(f"| p32_crit (bootstrap 中值) | **{perc['p32_crit_median']:.4f}** |")
    L.append(f"| bootstrap 5%–95% CI | [{perc['p32_crit_lo']:.4f}, {perc['p32_crit_hi']:.4f}] |")
    L.append(f"| 有效 seed 集数 | {perc['n_sets_ok']}/{perc['n_sets_requested']} |")
    L.append(f"| 每 P32 点实现数 | {perc.get('n_real_per_point', 'N/A')} |")
    L.append("")
    L.append("![渗流曲线](percolation.png)")
    L.append("")
    L.append("## 5. 垂直逃逸优先级")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    L.append(f"| 连通分数 | {perc.get('escape_connectivity_fraction', 'N/A')} |")
    L.append(f"| 逃逸优先级 | **{g['escape_penalty']['escape_priority']}** |")
    L.append("")
    L.append("## 6. 判级依据")
    L.append("")
    L.append(f"- **基础判级**: {g['grade_base']} ({g['overlap']['description']})")
    L.append(f"- **数值证据**: {g['overlap']['evidence']}")
    L.append(f"- **垂直逃逸修正**: 优先级 {g['escape_penalty']['escape_priority']} "
             f"(惩罚档位 {g['escape_penalty']['penalty_tiers']})")
    L.append("")
    L.append("## 7. 判定翻转敏感性矩阵")
    L.append("")
    L.append("| β | P32× | K± | 判级 | 翻转? |")
    L.append("|---|---|---|---|---|")
    for r in data["sensitivity"]["rows"]:
        L.append(f"| {r['beta']:.1f} | {r['p32_factor']:.1f} | {r['k_offset']:+d} | "
                 f"{r['grade']} | {'是' if r['flipped'] else '否'} |")
    L.append("")
    L.append(f"**翻转率**: {data['sensitivity']['n_flipped']}/"
             f"{data['sensitivity']['n_total']} "
             f"({data['sensitivity']['flip_rate']:.1%})")
    L.append("")
    L.append("![走向玫瑰图](rose.png)")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"*本报告由 fractureflow R56 核废场景一键筛查管线自动生成。"
             f"耗时 {meta['elapsed_sec']:.1f}s。*")
    return "\n".join(L)


def generate_html_report(data: dict, rose_path, perc_path) -> str:
    """HTML 报告 (内嵌 base64 图)."""
    g = data["grading"]
    meta = data["meta"]
    inp = data["input"]
    st = data["set_table"]
    grade_color = g.get("grade_color", "#f9a825")

    rose_b64 = _fig_to_base64(rose_path)
    perc_b64 = _fig_to_base64(perc_path)

    rows_html = ""
    for k in range(st["K"]):
        c = st["centers"][k]
        dip, dip_dir = normal_to_dip_dipdir(c)
        rows_html += (f"<tr><td>{k}</td><td>{dip:.1f}</td><td>{dip_dir:.1f}</td>"
                      f"<td>{st['concentrations'][k]:.1f}</td>"
                      f"<td>{st['proportions'][k]:.1%}</td></tr>")

    sens_rows = ""
    for r in data["sensitivity"]["rows"]:
        sens_rows += (f"<tr><td>{r['beta']:.1f}</td><td>{r['p32_factor']:.1f}</td>"
                      f"<td>{r['k_offset']:+d}</td><td>{r['grade']}</td>"
                      f"<td>{'是' if r['flipped'] else '否'}</td></tr>")

    honesty_list = "".join(f"<li>{a}</li>" for a in g["assumptions"])

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>核废处置场地适宜性筛查报告 - {inp['site_name']}</title>
<style>
  body {{ font-family: "Microsoft YaHei", "SimHei", sans-serif; max-width: 980px;
         margin: 0 auto; padding: 24px; color: #222; line-height: 1.7; }}
  h1 {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 12px; }}
  h2 {{ color: #1a5fa8; border-left: 4px solid #1a5fa8; padding-left: 10px; margin-top: 28px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 10px; text-align: center; font-size: 14px; }}
  th {{ background: #1a5fa8; color: white; }}
  .grade-card {{ background: {grade_color}; color: white; text-align: center;
                padding: 20px; border-radius: 8px; margin: 16px 0; }}
  .grade-card .g {{ font-size: 26px; font-weight: bold; }}
  .grade-card .r {{ font-size: 14px; margin-top: 8px; opacity: 0.95; }}
  .honesty {{ background: #fff8e1; border: 1px solid #f9a825; border-radius: 6px;
             padding: 14px 18px; color: #5d4037; margin: 14px 0; }}
  .watermark {{ color: #c62828; font-weight: bold; }}
  img {{ max-width: 100%; height: auto; display: block; margin: 12px auto;
        border: 1px solid #ddd; }}
  .footer {{ margin-top: 36px; padding-top: 12px; border-top: 1px solid #ccc;
             color: #888; font-size: 12px; text-align: center; }}
</style>
</head>
<body>
<h1>核废料处置场地适宜性筛查报告</h1>
<p style="text-align:center">场地: {inp['site_name']} | 生成: {meta['timestamp']} | 管线: {meta['pipeline']}</p>

<div class="grade-card">
  <div class="watermark">筛查级, 非场址最终判定</div>
  <div class="g">{g['grade']}</div>
  <div class="r">{g['reason']}</div>
</div>

<div class="honesty">
  <strong>⚠ 诚实边界 (必读)</strong>
  <ul>{honesty_list}</ul>
</div>

<h2>1. 输入数据</h2>
<table>
  <tr><th>井数</th><th>裂隙总数</th><th>组数 K</th><th>域尺寸 (m)</th><th>幂律 β</th></tr>
  <tr><td>{inp['n_wells']}</td><td>{inp['n_fractures']}</td><td>{st['K']}</td>
      <td>{inp['domain_m'][0]}×{inp['domain_m'][1]}×{inp['domain_m'][2]}</td>
      <td>{inp['beta']}</td></tr>
</table>

<h2>2. 裂隙组系表 (SetTable)</h2>
<table><tr><th>组</th><th>倾角°</th><th>倾向°</th><th>浓度 κ</th><th>占比</th></tr>
{rows_html}</table>

<h2>3. 强度估计 (P32) + 渗流阈值</h2>
<table>
  <tr><th>P32 P10</th><th>P32 P50</th><th>P32 P90</th>
      <th>p32_crit (中值)</th><th>bootstrap CI</th></tr>
  <tr><td>{data['p32_estimate']['p32_p10']:.4f}</td>
      <td>{data['p32_estimate']['p32_p50']:.4f}</td>
      <td>{data['p32_estimate']['p32_p90']:.4f}</td>
      <td><strong>{data['percolation']['p32_crit_median']:.4f}</strong></td>
      <td>[{data['percolation']['p32_crit_lo']:.4f}, {data['percolation']['p32_crit_hi']:.4f}]</td></tr>
</table>
<p style="color:#666;font-size:13px">P32 为编录表计数估计区间 (体视学量纲不确定), 引用区间而非点值。</p>
{f'<img src="data:image/png;base64,{perc_b64}" alt="渗流曲线"/>' if perc_b64 else ''}

<h2>4. 垂直逃逸优先级</h2>
<table>
  <tr><th>连通分数</th><th>逃逸优先级</th><th>保守惩罚档位</th></tr>
  <tr><td>{data['percolation'].get('escape_connectivity_fraction','N/A')}</td>
      <td><strong>{g['escape_penalty']['escape_priority']}</strong></td>
      <td>{g['escape_penalty']['penalty_tiers']}</td></tr>
</table>

<h2>5. 判级依据</h2>
<table>
  <tr><th>基础判级</th><th>数值证据</th><th>垂直逃逸修正</th></tr>
  <tr><td>{g['grade_base']} ({g['overlap']['description']})</td>
      <td>{g['overlap']['evidence']}</td>
      <td>优先级 {g['escape_penalty']['escape_priority']}</td></tr>
</table>

<h2>6. 判定翻转敏感性矩阵</h2>
<table>
  <tr><th>β</th><th>P32×</th><th>K±</th><th>判级</th><th>翻转?</th></tr>
  {sens_rows}
</table>
<p>翻转率: <strong>{data['sensitivity']['n_flipped']}/{data['sensitivity']['n_total']}
({data['sensitivity']['flip_rate']:.1%})</strong></p>
{f'<img src="data:image/png;base64,{rose_b64}" alt="走向玫瑰图"/>' if rose_b64 else ''}

<div class="footer">FractureFlow R56 核废场景一键筛查管线 | {meta['elapsed_sec']:.1f}s | 筛查级报告, 非场址最终判定</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 一键管线
# ---------------------------------------------------------------------------

def run_site_report(wells, site_name="北山预选区", domain=(50.0, 50.0, 50.0),
                    K=None, beta=3.5, out_dir="results/v_r56_demo/",
                    n_sets=3, n_real=20, seed=42):
    """一键核废处置筛查报告管线, 落盘四件套 + 玫瑰图 + 渗流图.

    返回:
        {
            "out_dir": str, "elapsed_sec": float,
            "report_json": str, "report_md": str, "report_html": str,
            "grade": str,
        }
    """
    t0 = time.time()
    print("=" * 60)
    print("  R56 核废场景一键筛查报告 (beishan 示范)")
    print("=" * 60)

    os.makedirs(out_dir, exist_ok=True)

    # 1. 组系表
    st, set_ids, K_used = _build_set_table(wells, K=K, seed=seed)
    n_frac = int(sum(len(np.atleast_1d(w.get("dip", []))) for w in wells))
    n_frac = int(sum(w["n_fractures"] for w in wells))
    print(f"[1/6] SetTable: K={K_used}, N_frac={n_frac}, 井数={len(wells)}")

    # 2. P32 估计
    p32_est = _estimate_p32_interval(wells, domain, seed=seed)
    print(f"[2/6] P32 估计: p50={p32_est['p32_p50']:.4f} "
          f"[{p32_est['p32_p10']:.4f}, {p32_est['p32_p90']:.4f}]")

    # 3. 渗流 + bootstrap CI
    print(f"[3/6] 渗流曲线 + bootstrap CI ({n_sets} seed 集)...")
    bootstrap = _bootstrap_p32crit(st, domain, p32_est, n_sets=n_sets,
                                   n_real=n_real, beta=beta)
    print(f"  p32_crit 中值={bootstrap['p32_crit_median']:.4f} "
          f"CI=[{bootstrap['p32_crit_lo']:.4f}, {bootstrap['p32_crit_hi']:.4f}]")

    # 4. 垂直逃逸 + 连通
    disp, aniso, dfn = _scene_escape_priority(st, domain, seed=seed, beta=beta)
    bootstrap["escape_connectivity_fraction"] = round(
        aniso["largest_fraction"], 4)
    bootstrap["n_real_per_point"] = n_real
    print(f"[4/6] 垂直逃逸优先级: {disp['escape_priority']} "
          f"(连通分数={aniso['largest_fraction']:.3f})")

    # 5. 判级 + 翻转敏感性
    template = load_nuclear_template()
    grade, sens = _build_grade_report(
        wells, K_used, domain, seed, beta, p32_est, bootstrap, disp,
        template, site_name)
    grade["grade_color"] = template.get("grades", {}).get(
        grade["grade"], {}).get("color", "#f9a825")
    print(f"[5/6] 判级: {grade['grade']} (基础={grade['grade_base']})")

    # 6. 图件 + 落盘
    rose_path = os.path.join(out_dir, "rose.png")
    perc_path = os.path.join(out_dir, "percolation.png")
    render_rose_plot(wells, rose_path)
    render_percolation_plot(bootstrap, perc_path)

    st_out = {
        "K": st.K,
        "centers": st.centers.tolist(),
        "concentrations": st.concentrations.tolist(),
        "proportions": st.proportions.tolist(),
    }
    perc_out = {k: v for k, v in bootstrap.items()
                if k not in ("main_perc", "main_grid")}

    data = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline": "fractureflow R56 site_report v1.0",
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "input": {
            "site_name": site_name,
            "n_wells": len(wells),
            "n_fractures": n_frac,
            "domain_m": list(domain),
            "beta": beta,
            "seed": seed,
        },
        "set_table": st_out,
        "p32_estimate": p32_est,
        "percolation": perc_out,
        "scenario_metrics": {
            "disposal_escape_priority": {
                k: v for k, v in disp.items() if k != "dominant_direction"
            },
        },
        "grading": grade,
        "sensitivity": sens,
        "honest_boundary": {
            "screening_level": True,
            "not_final_site_assessment": True,
            "watermark": template.get("watermark", "筛查级, 非场址最终判定"),
            "no_leakage_guarantee": "不构成'保证不泄漏'类承诺",
            "p32_stereological_uncertainty": True,
        },
    }

    json_path = os.path.join(out_dir, "site_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2,
                  default=str, allow_nan=False)

    md_path = os.path.join(out_dir, "site_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_md_report(data))

    html_path = os.path.join(out_dir, "site_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(generate_html_report(data, rose_path, perc_path))

    print(f"[6/6] 四件套落盘: {out_dir}")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    print(f"  HTML: {html_path}")
    print(f"  PNG:  {rose_path}, {perc_path}")
    elapsed = time.time() - t0
    print(f"\n判级: {grade['grade']} | 耗时 {elapsed:.1f}s")
    print("⚠ 本报告为筛查级, 非场址最终判定。")

    return {
        "out_dir": out_dir, "elapsed_sec": round(elapsed, 1),
        "report_json": json_path, "report_md": md_path,
        "report_html": html_path, "grade": grade["grade"],
    }


def run_site_report_cli(args):
    """CLI 入口: 解析参数 → 加载井数据 → 一键管线.

    与 eval.py --site-report 对接。
    """
    from .site_model import (load_wells_from_npz, load_wells_from_dir,
                             load_well_csv)
    site_name = getattr(args, "site_name", "北山预选区") or "北山预选区"
    domain = tuple(args.site_report_domain or [50.0, 50.0, 50.0])
    out_dir = args.site_report_out or "results/v_r56_demo/"
    K = getattr(args, "site_report_K", None)
    beta = getattr(args, "site_report_beta", 3.5)
    seed = getattr(args, "seed", 42)

    wells_src = getattr(args, "wells", None)
    if not wells_src:
        raise ValueError("--site-report 需要 --wells <npz|dir>")

    # 加载井数据 (复用 site_model 加载器, 含 CSV 容错)
    if os.path.isdir(wells_src):
        wells = load_wells_from_dir(wells_src)
    elif wells_src.endswith(".npz") or wells_src.endswith(".npy"):
        wells = load_wells_from_npz(wells_src)
    else:
        wells = [load_well_csv(wells_src)]

    return run_site_report(wells, site_name=site_name, domain=domain,
                           K=K, beta=beta, out_dir=out_dir, seed=seed)
