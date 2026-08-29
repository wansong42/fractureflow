# -*- coding: utf-8 -*-
"""钻孔组系表中文报告生成器 —— T2 产物.

输入: net dict (T1 borehole_excel_entry 输出) + 组系表结果.
输出: 中文 Markdown 报告 + 玫瑰图 + 极点图.

防坑:
  - H1: assumptions 段强制非空
  - H3: 图表仅作可视化, 不声称测量精度
  - H5: 组系表标注数据条件, 不声称是真值

依赖: matplotlib (绘图) + numpy.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 配置中文字体 (Windows 优先 SimHei, 回退 Microsoft YaHei / SimSun)
_CN_FONT = None
for _fn in ["SimHei", "Microsoft YaHei", "SimSun", "PingFang SC", "Heiti SC",
            "Arial Unicode MS"]:
    if any(f.name == _fn for f in font_manager.fontManager.ttflist):
        _CN_FONT = _fn
        break
if _CN_FONT:
    plt.rcParams["font.sans-serif"] = [_CN_FONT, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection


# ---------------------------------------------------------------------------
# 1. 法向 ↔ 倾向/倾角 转换 (与 T1 / report.py 一致)
# ---------------------------------------------------------------------------

def _unit(v, axis=-1):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def normal_to_dip_dir(nrm):
    """法向 → (dip, dip_direction).

    与 src/fractureflow/report.py::_dir_to_dip_azimuth 同口径:
      dip = arccos(|nz|) ∈ [0, 90]
      dip_direction = atan2(nx, ny) mod 360
    """
    n = _unit(nrm)
    nz = np.clip(np.abs(n[..., 2]), 0, 1)
    dip = np.degrees(np.arccos(nz))
    dip_dir = np.degrees(np.arctan2(n[..., 0], n[..., 1])) % 360
    return dip, dip_dir


# ---------------------------------------------------------------------------
# 2. 玫瑰图 (走向玫瑰)
# ---------------------------------------------------------------------------

def plot_rose_diagram(nrm, assign, K, path, bin_width=10, title="走向玫瑰图"):
    """绘制走向玫瑰图.

    参数:
        nrm: (L, 3) 法向数组
        assign: (L,) 组指派 (0..K-1)
        K: 组数
        path: 输出 png 路径
        bin_width: 分箱宽度 (°), 默认 10
        title: 图标题
    """
    nrm = np.asarray(nrm, dtype=np.float64)
    assign = np.asarray(assign, dtype=int)
    _, az = normal_to_dip_dir(nrm)

    n_bins = int(360 / bin_width)
    bins = np.arange(0, 360 + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection="polar"))

    colors = plt.cm.tab20(np.linspace(0, 1, max(K, 1)))

    # 计算最大值用于归一化
    max_count = 1
    for k in range(K):
        mask = assign == k
        if mask.sum() == 0:
            continue
        counts, _ = np.histogram(az[mask], bins=bins)
        max_count = max(max_count, counts.max())

    # 绘制每组
    for k in range(K):
        mask = assign == k
        if mask.sum() == 0:
            continue
        counts, _ = np.histogram(az[mask], bins=bins)
        # 柱宽 (弧度)
        theta = np.deg2rad(bins[:-1] + bin_width / 2)
        width = np.deg2rad(bin_width * 0.9)
        # 归一化高度
        radii = counts / max_count * 0.8 + 0.05
        bars = ax.bar(theta, radii, width=width, bottom=0.0,
                      color=colors[k % 20], alpha=0.7, edgecolor="white",
                      linewidth=0.3, label=f"组{k+1}" if k < 12 else None)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title, pad=20, fontsize=14, fontweight="bold")

    # 方位标签
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=10)

    ax.set_yticklabels([])
    if K <= 12:
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9,
                  title="组号", title_fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. 极点图 (下半球立体投影 / Stereographic)
# ---------------------------------------------------------------------------

def plot_stereonet(nrm, assign, centers, K, path, title="极点图 (立体投影)"):
    """绘制下半球立体投影 (Stereographic) 极点图。

    投影实现: r = tan((90-dip)/2), 即等角立体投影 (Wulff 网)。
    注意: 不是 Schmidt 等面积投影 (后者 = sqrt(2)*sin((90-dep)/2))。

    参数:
        nrm: (L, 3) 法向数组
        assign: (L,) 组指派
        centers: (K, 3) 组心方向
        K: 组数
        path: 输出 png 路径
        title: 图标题
    """
    nrm = np.asarray(nrm, dtype=np.float64)
    assign = np.asarray(assign, dtype=int)
    centers = np.asarray(centers, dtype=np.float64)
    dip, dip_dir = normal_to_dip_dir(nrm)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)

    # 外圆
    outer = Circle((0, 0), 1, fill=False, edgecolor="black", linewidth=1.5)
    ax.add_patch(outer)

    # 纬线 (dip 从水平=0 到 垂直=90)
    for d in range(10, 90, 10):
        r = np.tan(np.radians(d / 2))  # 立体投影 (等角) 半径
        circle = Circle((0, 0), r, fill=False, edgecolor="lightgray",
                        linewidth=0.5, linestyle="--")
        ax.add_patch(circle)

    # 经线 (走向)
    for a in range(0, 180, 10):
        x_end = np.sin(np.radians(a))
        y_end = np.cos(np.radians(a))
        ax.plot([0, x_end], [0, y_end], color="lightgray", linewidth=0.5,
                linestyle="--")
        ax.plot([0, -x_end], [0, -y_end], color="lightgray", linewidth=0.5,
                linestyle="--")

    # 方位标签
    for a, label in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        x = 1.08 * np.sin(np.radians(a))
        y = 1.08 * np.cos(np.radians(a))
        ax.text(x, y, label, ha="center", va="center", fontsize=12,
                fontweight="bold")

    colors = plt.cm.tab20(np.linspace(0, 1, max(K, 1)))

    # 绘制每个点 (立体投影)
    for k in range(K):
        mask = assign == k
        if mask.sum() == 0:
            continue
        dk = dip[mask]
        adk = dip_dir[mask]
        # dip=0 (水平面) → 外圆 r=1; dip=90 (垂直面) → 圆心 r=0
        r = np.tan(np.radians((90 - dk) / 2))
        x = r * np.sin(np.radians(adk))
        y = r * np.cos(np.radians(adk))
        ax.scatter(x, y, s=12, color=colors[k % 20], alpha=0.6, zorder=3,
                   edgecolors="none")

    # 组心大符号 (星号)
    for k in range(K):
        if k >= len(centers):
            break
        c = centers[k]
        if np.linalg.norm(c) < 0.1:
            continue
        cc = _unit(c)
        dip_c, ad_c = normal_to_dip_dir(cc.reshape(1, 3))
        r_c = np.tan(np.radians((90 - dip_c[0]) / 2))
        x_c = r_c * np.sin(np.radians(ad_c[0]))
        y_c = r_c * np.cos(np.radians(ad_c[0]))
        ax.scatter(x_c, y_c, s=200, color=colors[k % 20], marker="*",
                   edgecolors="black", linewidths=1.5, zorder=5,
                   label=f"组{k+1}")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=9, title="组心", title_fontsize=10)

    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 4. 组系表 (DataFrame)
# ---------------------------------------------------------------------------

def build_group_table(nrm, assign, centers, K, depth=None, bootstrap=True,
                      near_dup_threshold=15.0):
    """构建组系表 DataFrame (中文列).

    纯加法变更 (2026-08-29, 外部审查雷5/雷6):
      新增列: 组内最大偏差(°) / CI半宽(°) / 备注
      既有列 (组号/走向/倾向/倾角/条数/占比/组内离散) 完全保留, 旧调用零漂移.

    口径:
      - 组内离散 = arccos(mean|cos|) (球面, 全项目统一)
      - max_dev  = 到组心最大角距
      - 最小样本规则: N<3 时离散/最大偏差标"不可统计", 不报数值
      - CI半宽   = bootstrap_modal_ci 95% CI 半宽 (B4, 已有实现)
      - 近重复组: 任两组模态方向角距 < near_dup_threshold 标警告
    """
    import pandas as pd
    from .set_table_eval import bootstrap_modal_ci

    nrm = np.asarray(nrm, dtype=np.float64)
    assign = np.asarray(assign, dtype=int)
    centers = np.asarray(centers, dtype=np.float64)

    rows = []
    modal_vecs = []   # (k, 组心单位向量) 供近重复检测
    n_total = len(nrm)

    for k in range(K):
        mask = assign == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue

        # 组心方向 → 走向/倾向/倾角
        if k < len(centers):
            c = _unit(centers[k].reshape(1, 3)).flatten()
            dip_c, dipdir_c = normal_to_dip_dir(c.reshape(1, 3))
            dip_c = dip_c[0]
            dipdir_c = dipdir_c[0]
        else:
            dip_c = dipdir_c = 0.0
            c = np.zeros(3)

        # 走向 = 倾向 - 90° (mod 360)
        strike = (dipdir_c - 90) % 360

        # 组内离散 (统一 arccos(mean|cos|)) + max_dev (到组心最大角距)
        pts = nrm[mask]
        cos_to_center = np.clip(np.abs(pts @ c), 0, 1)
        ang = np.degrees(np.arccos(cos_to_center))
        statistical = n_k >= 3
        if statistical:
            dispersion = round(float(ang.mean()), 1)
            max_dev = round(float(ang.max()), 1)
        else:
            # 最小样本规则: N<3 不报离散, 标"不可统计"
            dispersion = "不可统计(N<3)"
            max_dev = "不可统计(N<3)"

        # 可选 bootstrap 95% CI 半宽 (B4)
        ci_hw = None
        if bootstrap:
            try:
                ci_hw = round(float(bootstrap_modal_ci(pts, n_boot=200)["ci_half_width"]), 2)
            except Exception:
                ci_hw = None

        rows.append({
            "组号": k + 1,
            "走向(°)": round(strike, 1),
            "倾向(°)": round(dipdir_c, 1),
            "倾角(°)": round(dip_c, 1),
            "条数": n_k,
            "占比(%)": round(n_k / n_total * 100, 1),
            "组内离散(°)": dispersion,
            "组内最大偏差(°)": max_dev,
            "CI半宽(°)": ci_hw if ci_hw is not None else "—",
            "备注": "",
        })
        modal_vecs.append((k, c))

    # 近重复组警告 (模态方向 3D 角距 < 阈值)
    for i in range(len(modal_vecs)):
        for j in range(i + 1, len(modal_vecs)):
            ki, ci = modal_vecs[i]
            kj, cj = modal_vecs[j]
            if np.linalg.norm(ci) < 1e-9 or np.linalg.norm(cj) < 1e-9:
                continue
            ang_d = np.degrees(np.arccos(np.clip(abs(float(ci @ cj)), 0, 1)))
            if ang_d < near_dup_threshold:
                warn = f"近重复组: 与组{ki + 1}模态角距{ang_d:.1f}°(<{near_dup_threshold}°)"
                rows[i]["备注"] = (rows[i]["备注"] + "; " + warn).strip("; ")
                rows[j]["备注"] = (rows[j]["备注"] + "; " + warn).strip("; ")

    df_table = pd.DataFrame(rows)
    # 整型列强制转换 (防 pandas 推断为 float)
    for col in ["组号", "条数"]:
        if col in df_table.columns:
            df_table[col] = df_table[col].astype(int)
    return df_table


# ---------------------------------------------------------------------------
# 5. 完整报告生成
# ---------------------------------------------------------------------------

def generate_borehole_report(nets, set_table_result, quality_report,
                             site_name, out_dir, depth=None):
    """生成完整中文报告 (Markdown + 玫瑰图 + 极点图).

    参数:
        nets: list[dict] — T1 borehole_excel_entry 输出
        set_table_result: dict — {centers, assign, n_points}
        quality_report: dict — T1 质量报告
        site_name: str — 场地名称
        out_dir: str — 输出目录 (图 + md 都落这里)
        depth: (L,) 可选深度数组

    返回:
        report_md: str — Markdown 报告
    """
    os.makedirs(out_dir, exist_ok=True)

    # 合并所有井的法向和指派
    all_nrm = []
    all_assign = []
    offset = 0
    for net in nets:
        nrm = np.asarray(net["nrm"], dtype=np.float64)
        n = len(nrm)
        assign_k = set_table_result["assign"][offset:offset + n]
        all_nrm.append(nrm)
        all_assign.append(assign_k)
        offset += n

    all_nrm = np.vstack(all_nrm)
    all_assign = np.concatenate(all_assign)
    centers = set_table_result["centers"]
    K = centers.shape[0]

    # 画图
    rose_path = os.path.join(out_dir, "rose_diagram.png")
    stereo_path = os.path.join(out_dir, "stereonet.png")
    plot_rose_diagram(all_nrm, all_assign, K, rose_path,
                      title=f"{site_name} 走向玫瑰图")
    plot_stereonet(all_nrm, all_assign, centers, K, stereo_path,
                   title=f"{site_name} 极点图 (立体投影)")

    # 组系表
    df_table = build_group_table(all_nrm, all_assign, centers, K, depth)

    # 质量等级文字
    grade_map = {"A": "优 (≥50 条)", "B": "良 (20-50 条)", "F": "不足 (<20 条, 已拒绝)"}
    grade = quality_report.get("quality_grade", "未知")
    grade_text = grade_map.get(grade, grade)

    # 各井信息
    well_info = quality_report.get("per_well_counts", {})

    # 组装 Markdown
    lines = []
    lines.append(f"# 钻孔裂隙组系统计报告")
    lines.append("")
    lines.append(f"**场地**: {site_name}  ")
    lines.append(f"**报告生成时间**: {_timestamp()}  ")
    lines.append(f"**数据质量等级**: {grade_text}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 诚实边界 (强制非空)
    lines.append("## ⚠ 重要说明 (必读)")
    lines.append("")
    lines.append("本报告基于钻孔编录表的**自动组系划分**，供初步参考。")
    lines.append("")
    assumptions = [
        "组系划分基于法向几何聚类 (球面 k-means)，非地质成因分组",
        "井内位置用一维近似 (深度, 0, 0)，无三维轨迹信息",
        "法向基于钻孔产状测量，可能存在测量误差",
        "组内离散度反映几何分散，不等于力学性质差异",
    ]
    if grade == "B":
        assumptions.append("数据量偏少 (20-50 条)，组系划分不确定性较大，建议谨慎使用")
    for a in assumptions:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 输入数据摘要
    lines.append("## 1. 输入数据摘要")
    lines.append("")
    lines.append(f"| 参数 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 钻孔数 | {len(nets)} |")
    lines.append(f"| 裂隙总数 | {len(all_nrm)} |")
    lines.append(f"| 识别组数 K | {K} |")
    if well_info:
        well_str = ", ".join(f"{w}: {c} 条" for w, c in list(well_info.items())[:10])
        if len(well_info) > 10:
            well_str += f" ... (共 {len(well_info)} 口井)"
        lines.append(f"| 各井裂隙数 | {well_str} |")
    if quality_report.get("n_removed_oob", 0) > 0:
        lines.append(f"| 越界剔除 | {quality_report['n_removed_oob']} 条 |")
    if quality_report.get("n_removed_nan", 0) > 0:
        lines.append(f"| 空值剔除 | {quality_report['n_removed_nan']} 条 |")
    if quality_report.get("n_removed_dup", 0) > 0:
        lines.append(f"| 重复剔除 | {quality_report['n_removed_dup']} 条 |")
    if quality_report.get("depth_nonmonotonic", False):
        lines.append(f"| ⚠ 深度非单调 | 存在深度回跳, 已给出警告 |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 图表
    lines.append("## 2. 可视化")
    lines.append("")
    lines.append("### 2.1 走向玫瑰图")
    lines.append("")
    lines.append(f"![走向玫瑰图]({os.path.basename(rose_path)})")
    lines.append("")
    lines.append("### 2.2 极点图")
    lines.append("")
    lines.append(f"![极点图]({os.path.basename(stereo_path)})")
    lines.append("")
    lines.append("> 极点图使用下半球立体投影 (Stereographic / Wulff 网)。星号 = 组心方向。")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 组系表
    lines.append("## 3. 裂隙组系表")
    lines.append("")
    if len(df_table) > 0:
        # 手动生成 markdown 表格 (用 itertuples 防 iterrows 类型强转)
        cols = list(df_table.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in df_table.itertuples(index=False):
            cells = []
            for c, v in zip(cols, row):
                if isinstance(v, (int, np.integer)):
                    cells.append(str(int(v)))
                elif isinstance(v, (float, np.floating)):
                    cells.append(f"{v:.1f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("*无有效组系数据*")
    lines.append("")
    lines.append(f"**注**: 走向 = 倾向 - 90° (mod 360)。组内离散 = 与组心的平均角距。")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 复现信息
    lines.append("## 4. 复现信息")
    lines.append("")
    lines.append(f"- 入口脚本: `scripts/borehole_excel_entry.py`")
    lines.append(f"- 报告脚本: `scripts/borehole_report.py`")
    lines.append(f"- 环境: `{sys.executable}`")
    lines.append("")

    report_md = "\n".join(lines)

    # 落盘 md
    md_path = os.path.join(out_dir, "borehole_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    return report_md, md_path, rose_path, stereo_path


def _timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
