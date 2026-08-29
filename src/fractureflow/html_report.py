# -*- coding: utf-8 -*-
"""T18: 单文件 HTML + Word 兼容报告生成器.

将管线 JSON 数据转化为:
  1. 单文件 HTML 报告 (玫瑰图/极点图 base64 内嵌, 中文样式, 断网可用)
  2. Word 兼容 .doc (HTML 别名, Word 可直接打开)

零新依赖: stdlib + matplotlib (禁止装 jinja2/python-docx).
遵循防坑 H1: 报告模板强制 assumptions 字段非空.
"""

import base64
import io
import json
import os
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 配置中文字体
_CN_FONT = None
for _fn in ["SimHei", "Microsoft YaHei", "SimSun", "PingFang SC", "Heiti SC",
            "Arial Unicode MS"]:
    if any(f.name == _fn for f in font_manager.fontManager.ttflist):
        _CN_FONT = _fn
        break
if _CN_FONT:
    plt.rcParams["font.sans-serif"] = [_CN_FONT, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 1. 绘图函数
# ---------------------------------------------------------------------------

def _unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def _dir_to_dip_azimuth(dir_vec):
    """法向 → (dip, dip_direction)."""
    dir_vec = np.asarray(dir_vec, float)
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)
    nx, ny, nz = dir_vec
    dip = float(np.degrees(np.arccos(np.clip(abs(nz), 0, 1))))
    dip_dir = float(np.degrees(np.arctan2(nx, ny)) % 360)
    return dip, dip_dir


def _fig_to_base64(fig):
    """matplotlib figure → base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_rose_diagram(nrm, assign, K, bin_width=10):
    """绘制走向玫瑰图, 返回 base64 PNG."""
    nrm = np.asarray(nrm, float)
    nrm = _unit(nrm)
    # 走向 = 倾向 - 90° = atan2(nx, ny) mod 360
    strikes = np.degrees(np.arctan2(nrm[:, 0], nrm[:, 1])) % 360

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    bins = np.arange(0, 360 + bin_width, bin_width)
    # 对称化到 [0, 180] 因为走向无向
    strikes_sym = strikes % 180
    counts, edges = np.histogram(strikes_sym, bins=np.arange(0, 180 + bin_width, bin_width))
    theta = np.radians(edges[:-1] + bin_width / 2)
    width = np.radians(bin_width)
    bars = ax.bar(theta, counts, width=width, bottom=0.0,
                  color="#2a9d8f", alpha=0.8, edgecolor="white")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("走向玫瑰图", fontsize=14, pad=20)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_pole_diagram(nrm, assign, K):
    """绘制极点图 (施密特等面积投影), 返回 base64 PNG."""
    nrm = np.asarray(nrm, float)
    nrm = _unit(nrm)
    strikes = np.degrees(np.arctan2(nrm[:, 0], nrm[:, 1])) % 360
    dips = np.degrees(np.arccos(np.clip(np.abs(nrm[:, 2]), 0, 1)))

    fig, ax = plt.subplots(figsize=(6, 6))
    # 施密特投影
    dip_rad = np.radians(dips)
    strike_rad = np.radians(strikes)
    r = np.sqrt(2) * np.sin(dip_rad / 2)
    x = r * np.sin(strike_rad)
    y = r * np.cos(strike_rad)

    # 按组上色
    if assign is not None:
        assign = np.asarray(assign, int)
        colors = plt.cm.Set1(np.linspace(0, 1, max(K, 1)))
        for k in range(K):
            sel = assign == k
            if sel.sum():
                ax.scatter(x[sel], y[sel], s=20, alpha=0.7,
                           color=colors[k % len(colors)], label=f"组{k}")
        ax.legend(loc="lower right", fontsize=8)
    else:
        ax.scatter(x, y, s=20, alpha=0.7, color="#2a9d8f")

    # 外圆
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.sin(theta), np.cos(theta), "k-", linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("极点图 (施密特等面积投影)", fontsize=14)
    ax.set_xlabel("东")
    ax.set_ylabel("北")
    fig.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# 2. HTML 报告生成
# ---------------------------------------------------------------------------

HTML_TEMPLATE = textwrap.dedent("""\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{
    font-family: "Microsoft YaHei", "SimHei", "SimSun", sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
    color: #333;
    line-height: 1.6;
  }}
  h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 8px; }}
  h2 {{ color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 4px; margin-top: 30px; }}
  h3 {{ color: #34495e; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 14px;
  }}
  th, td {{
    border: 1px solid #bdc3c7;
    padding: 6px 10px;
    text-align: left;
  }}
  th {{ background-color: #2c3e50; color: white; }}
  tr:nth-child(even) {{ background-color: #f2f3f4; }}
  .warning {{
    background-color: #fdf2e9;
    border-left: 4px solid #e67e22;
    padding: 12px 16px;
    margin: 16px 0;
  }}
  .info {{
    background-color: #eaf2f8;
    border-left: 4px solid #2980b9;
    padding: 12px 16px;
    margin: 16px 0;
  }}
  img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 16px auto;
    border: 1px solid #bdc3c7;
  }}
  .footer {{
    margin-top: 40px;
    padding-top: 12px;
    border-top: 1px solid #bdc3c7;
    color: #7f8c8d;
    font-size: 12px;
  }}
</style>
</head>
<body>
{body}
</body>
</html>
""")


def _data_to_html_body(data, site_name, rose_b64, pole_b64):
    """把管线数据转为 HTML body 字符串."""
    meta = data.get('meta', {})
    inp = data.get('input', {})
    st = data.get('set_table', {})
    p32_est = data.get('p32_estimate', {})
    perc = data.get('percolation', {})
    scenario = data.get('scenario_metrics', {})
    honest = data.get('honest_boundary', {})

    parts = []

    # 标题
    parts.append(f"<h1>裂隙网络连通性筛查报告</h1>")
    parts.append(f"<p><strong>场地</strong>: {site_name} &nbsp;|&nbsp; "
                 f"<strong>生成时间</strong>: {meta.get('timestamp', 'N/A')} &nbsp;|&nbsp; "
                 f"<strong>管线版本</strong>: {meta.get('pipeline', 'N/A')}</p>")

    # 诚实边界 (强制非空)
    parts.append("<h2>⚠ 诚实边界 (重要)</h2>")
    parts.append('<div class="warning">')
    parts.append("<p>本报告是<strong>筛查级结论 + 不确定性区间</strong>，"
                 "<strong>不是连通性判定</strong>。</p>")
    assumptions_list = []
    if honest.get('screening_level'):
        assumptions_list.append("筛查级结论，非真实连通性判定")
    if honest.get('no_aperture'):
        assumptions_list.append(honest['no_aperture'])
    if perc.get('assumptions'):
        assumptions_list.append(perc['assumptions'])
    if not assumptions_list:
        assumptions_list.append("[错误: assumptions 字段为空，请检查管线输出]")
    parts.append("<ul>" + "".join(f"<li>{a}</li>" for a in assumptions_list) + "</ul>")
    parts.append("</div>")

    # 1. 输入数据
    parts.append("<h2>1. 输入数据</h2>")
    parts.append("<table><tr><th>参数</th><th>值</th></tr>")
    parts.append(f"<tr><td>裂隙总数</td><td>{inp.get('N_fractures', 'N/A')}</td></tr>")
    parts.append(f"<tr><td>组数 K</td><td>{st.get('K', 'N/A')}</td></tr>")
    parts.append(f"<tr><td>域尺寸</td><td>{inp.get('domain_m', 'N/A')} m</td></tr>")
    parts.append("</table>")

    # 2. 组系表
    parts.append("<h2>2. 裂隙组系表 (SetTable)</h2>")
    parts.append("<table><tr><th>组</th><th>法向 (nx, ny, nz)</th>"
                 "<th>倾向°</th><th>倾角°</th><th>浓度 κ</th><th>占比</th></tr>")
    centers = st.get('centers', [])
    concentrations = st.get('concentrations', [])
    proportions = st.get('proportions', [])
    for k in range(len(centers)):
        c = centers[k]
        kappa = concentrations[k] if k < len(concentrations) else 0
        prop = proportions[k] if k < len(proportions) else 0
        dip, dip_dir = _dir_to_dip_azimuth(c)
        parts.append(
            f"<tr><td>{k}</td>"
            f"<td>({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})</td>"
            f"<td>{dip_dir:.1f}</td><td>{dip:.1f}</td>"
            f"<td>{kappa:.1f}</td><td>{prop:.1%}</td></tr>"
        )
    parts.append("</table>")
    parts.append("<p><em>说明: 倾向/倾角由法向换算 (右手法则)。"
                 "浓度 κ 越大组内越集中。</em></p>")

    # 图: 玫瑰图 + 极点图
    if rose_b64:
        parts.append("<h3>走向玫瑰图</h3>")
        parts.append(f'<img src="data:image/png;base64,{rose_b64}" alt="走向玫瑰图">')
    if pole_b64:
        parts.append("<h3>极点图</h3>")
        parts.append(f'<img src="data:image/png;base64,{pole_b64}" alt="极点图">')

    # 3. P32 估计
    parts.append("<h2>3. 强度估计 (P32)</h2>")
    parts.append("<table><tr><th>分位</th><th>P32 (m²/m³)</th></tr>")
    parts.append(f"<tr><td>P10</td><td>{p32_est.get('p32_p10', 0):.4f}</td></tr>")
    parts.append(f"<tr><td>P50</td><td>{p32_est.get('p32_p50', 0):.4f}</td></tr>")
    parts.append(f"<tr><td>P90</td><td>{p32_est.get('p32_p90', 0):.4f}</td></tr>")
    parts.append("</table>")

    # 4. 渗流阈值
    parts.append("<h2>4. 渗流阈值</h2>")
    has_perc = bool(perc) and perc.get('p32_crit') is not None
    beta = perc.get('beta', 'N/A') if perc else 'N/A'
    beta_str = f"{beta:.2f}" if isinstance(beta, float) else str(beta)
    if not has_perc:
        # H1 防线延伸: 缺渗流字段不得渲染成 "P32_crit=0.000 大概率渗流"
        parts.append("<table><tr><th>参数</th><th>值</th></tr>")
        parts.append(f"<tr><td>幂律指数 β</td><td>{beta_str}</td></tr>")
        parts.append("<tr><td><strong>P32_crit</strong></td>"
                     "<td><strong>N/A (输入 JSON 未含渗流结果)</strong></td></tr>")
        parts.append("</table>")
    else:
        p32_crit = perc['p32_crit']
        p32_crit_lo = perc.get('p32_crit_lower')
        p32_crit_hi = perc.get('p32_crit_upper')
        lo_str = f"{p32_crit_lo:.3f}" if isinstance(p32_crit_lo, (int, float)) else "N/A"
        hi_str = f"{p32_crit_hi:.3f}" if isinstance(p32_crit_hi, (int, float)) else "N/A"
        parts.append("<table><tr><th>参数</th><th>值</th></tr>")
        parts.append(f"<tr><td>幂律指数 β</td><td>{beta_str}</td></tr>")
        parts.append(f"<tr><td><strong>P32_crit</strong></td>"
                     f"<td><strong>{p32_crit:.3f}</strong></td></tr>")
        parts.append(f"<tr><td>不确定带</td>"
                     f"<td>[{lo_str}, {hi_str}]</td></tr>")
        parts.append("</table>")
        parts.append(f'<div class="info">'
                     f'<p><strong>解读</strong>: P32_crit 是网络从"不连通"到"连通"的临界强度。'
                     f'当实测 P32 &gt; {p32_crit:.3f} 时，网络大概率渗流'
                     f' (不确定带为多实现 [p_conn=0.1, 0.9] 交叉带, 非 95% CI)。</p>'
                     f'</div>')

    # 5. 场景指标
    parts.append("<h2>5. 场景化指标</h2>")
    egs = scenario.get('egs', {})
    mine = scenario.get('mine', {})
    disp = scenario.get('disposal', {})
    for name, sec in [("地热 (EGS)", egs), ("矿山/隧洞 (突水)", mine),
                       ("核废料处置 (逃逸)", disp)]:
        if not sec:
            continue
        parts.append(f"<h3>{name}</h3>")
        parts.append("<table><tr><th>指标</th><th>值</th></tr>")
        for k, v in sec.items():
            if isinstance(v, float):
                v = f"{v:.3f}"
            elif v is None:
                v = "N/A (各向同性)"   # None 角度不再裸渲染 "None"
            parts.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
        parts.append("</table>")

    # 6. 数据升档建议
    parts.append("<h2>6. 数据升档建议</h2>")
    parts.append("<table><tr><th>升级项</th><th>效果</th><th>对应带收窄</th></tr>")
    parts.append("<tr><td>露头/岩心迹长</td><td>β 可拟合</td><td>P32_crit 带收窄 ~30%</td></tr>")
    parts.append("<tr><td>开度</td><td>Oda 绝对渗透解锁</td><td>可输出绝对渗透量级</td></tr>")
    parts.append("<tr><td>间距</td><td>面密度直接算</td><td>P32 估计不再需要</td></tr>")
    parts.append("</table>")

    # 页脚
    elapsed = meta.get('elapsed_sec', '')
    parts.append(f'<div class="footer">'
                 f'<em>本报告由 fractureflow v3 连通性模块自动生成。耗时 {elapsed}s</em>'
                 f'</div>')

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 3. 主入口
# ---------------------------------------------------------------------------

def generate_html_report(data, site_name="场地", out_path=None):
    """生成单文件 HTML 报告.

    参数:
        data: 管线 JSON 数据 (dict) 或 JSON 文件路径
        site_name: 场地名称
        out_path: 输出 HTML 路径 (默认 input_path 同名 .html)

    返回:
        HTML 字符串
    """
    if isinstance(data, str):
        with open(data, "r", encoding="utf-8") as f:
            data = json.load(f)
        if out_path is None:
            out_path = os.path.splitext(data)[0] + ".html"

    # 绘图
    st = data.get('set_table', {})
    centers = st.get('centers', [])
    rose_b64 = None
    pole_b64 = None
    if centers:
        centers_arr = np.asarray(centers, float)
        assign_arr = np.arange(len(centers_arr))
        try:
            pole_b64 = plot_pole_diagram(centers_arr, assign_arr, len(centers_arr))
        except Exception:
            pole_b64 = None
        try:
            rose_b64 = plot_rose_diagram(centers_arr, assign_arr, len(centers_arr))
        except Exception:
            rose_b64 = None

    body = _data_to_html_body(data, site_name, rose_b64, pole_b64)
    html = HTML_TEMPLATE.format(title=f"裂隙连通性报告 - {site_name}", body=body)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

    return html


def generate_doc_report(data, site_name="场地", out_path=None):
    """生成 Word 兼容 .doc (HTML 别名).

    Word 可直接打开 .doc 格式的 HTML 文件.
    """
    html = generate_html_report(data, site_name, out_path=None)

    if isinstance(data, str):
        if out_path is None:
            out_path = os.path.splitext(data)[0] + ".doc"
    elif out_path is None:
        out_path = "report.doc"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return out_path
