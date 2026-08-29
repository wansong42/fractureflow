# -*- coding: utf-8 -*-
"""T1: Excel/CSV 编录表入口.

输入: .xlsx / .csv, 每裂隙一行.
必须三列: 深度(m) / 倾角 dip(°) / 倾向 dip_direction(°).
可选列: 井号/钻孔编号/well (缺失则整个文件当一口井).

坐标转换 (写死):
    ENU 单位法向 = (sinδ·sinα, sinδ·cosα, cosδ)
    δ = dip (倾角, °), α = dip_direction (倾向方位角, 从北顺时针, °).
井内位置: pos = (深度, 0, 0) (编录表无三维轨迹, 诚实假设).

输出: net dict (pos/nrm/nrm_full/wid, 兼容 set_table / dfn 管线) + 质量报告 JSON.

用法:
    python scripts/borehole_excel_entry.py --input data/编录表.xlsx --out results/borehole_net.pt
    python scripts/borehole_excel_entry.py --input data/编录表.csv --out results/borehole_net.pt --well-col 井号
"""

import argparse
import json
import os
import sys
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


# ---------------------------------------------------------------------------
# 1. 列名匹配
# ---------------------------------------------------------------------------

DEPTH_ALIASES = {"深度", "孔深", "md", "depth", "mdepth", "测深", "井深"}
DIP_ALIASES = {"倾角", "dip", "倾角α", "倾角(°)", "倾角(度)", "dip_angle"}
DIPDIR_ALIASES = {"倾向", "倾向角", "方位", "方位角", "dip direction", "dip_direction",
                  "倾向方位角", "倾向(°)", "倾向(度)", "azimuth", "trend"}
WELL_ALIASES = {"井号", "钻孔编号", "well", "钻孔", "井名", "孔号", "borehole", "hole"}


def _norm_col(name):
    """列名归一化: 去掉括号及内容 + 小写 + 去空格."""
    s = str(name).strip().lower()
    # 去掉括号及其内容: (m), (°), （度）, [m] 等
    s = re.sub(r"[\(（\[].*?[\)）\]]", "", s)
    # 去掉残留的 ° 度 等字符
    s = re.sub(r"[°度]", "", s)
    # 去掉首尾空白
    s = s.strip()
    return s


def _match_col(candidate, alias_set):
    """检查候选列名是否匹配任一别名.

    先精确匹配, 再子串匹配 (列名包含别名 或 别名包含列名).
    """
    nc = _norm_col(candidate)
    for alias in alias_set:
        na = _norm_col(alias)
        if nc == na:
            return True
        # 子串匹配: 列名包含别名 或 别名包含列名 (长度>=2 防误匹配)
        if len(na) >= 2 and (na in nc or nc in na):
            return True
    return False


def find_columns(df_columns):
    """从 DataFrame 列名中找到深度/倾角/倾向/井号列.

    返回 (depth_col, dip_col, dipdir_col, well_col) —— 找不到则为 None.
    """
    depth_col = dip_col = dipdir_col = well_col = None
    for col in df_columns:
        if depth_col is None and _match_col(col, DEPTH_ALIASES):
            depth_col = col
        elif dip_col is None and _match_col(col, DIP_ALIASES):
            dip_col = col
        elif dipdir_col is None and _match_col(col, DIPDIR_ALIASES):
            dipdir_col = col
        elif well_col is None and _match_col(col, WELL_ALIASES):
            well_col = col
    return depth_col, dip_col, dipdir_col, well_col


# ---------------------------------------------------------------------------
# 2. 文件读取
# ---------------------------------------------------------------------------

def _read_csv(path):
    """用 pandas 读 CSV, 返回 DataFrame."""
    import pandas as pd
    # 尝试常见编码
    for enc in ["utf-8", "gbk", "gb2312", "utf-8-sig"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # fallback: 让 pandas 自己猜
    return pd.read_csv(path)


def _read_xlsx_minimal(path):
    """最小 xlsx 解析器 (仅用标准库 zipfile + xml).

    读取第一个 sheet 的数据, 返回 DataFrame.
    注意: 仅支持纯文本/数字单元格, 不支持公式/格式.
    """
    import pandas as pd

    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    shared_strings = []

    with zipfile.ZipFile(path, "r") as z:
        # 读取共享字符串
        if "xl/sharedStrings.xml" in z.namelist():
            with z.open("xl/sharedStrings.xml") as f:
                tree = ET.parse(f)
                root = tree.getroot()
                for si in root.findall("main:si", ns):
                    # 简单情况: <si><t>text</t></si>
                    t = si.find("main:t", ns)
                    if t is not None and t.text is not None:
                        shared_strings.append(t.text)
                    else:
                        # 富文本: <si><r><t>...</t></r>...</si>
                        parts = []
                        for r in si.findall("main:r", ns):
                            t2 = r.find("main:t", ns)
                            if t2 is not None and t2.text is not None:
                                parts.append(t2.text)
                        shared_strings.append("".join(parts))

        # 读取第一个 sheet
        # 从 workbook.xml 找第一个 sheet 名
        sheet_name = "Sheet1"
        if "xl/workbook.xml" in z.namelist():
            with z.open("xl/workbook.xml") as f:
                tree = ET.parse(f)
                root = tree.getroot()
                sheets = root.findall(".//main:sheet", ns)
                if sheets:
                    sname = sheets[0].get("name", "Sheet1")
                    # 找对应的 rId → 文件名
                    rId = sheets[0].get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                    if rId:
                        # 从 workbook.xml.rels 找路径
                        if "xl/_rels/workbook.xml.rels" in z.namelist():
                            with z.open("xl/_rels/workbook.xml.rels") as rf:
                                rel_tree = ET.parse(rf)
                                rel_root = rel_tree.getroot()
                                rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
                                for rel in rel_root:
                                    if rel.get("Id") == rId:
                                        target = rel.get("Target")
                                        if target:
                                            if not target.startswith("/"):
                                                target = "xl/" + target.replace("../", "")
                                            sheet_name = target
                                            break

        # 读取 sheet 数据
        if sheet_name.startswith("xl/"):
            sheet_path = sheet_name
        else:
            sheet_path = f"xl/worksheets/{sheet_name}.xml"
            if sheet_path not in z.namelist():
                # 尝试通用名
                sheet_path = "xl/worksheet/sheet1.xml"
            if sheet_path not in z.namelist():
                # 找第一个 sheet*.xml
                for n in z.namelist():
                    if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"):
                        sheet_path = n
                        break

        with z.open(sheet_path) as f:
            tree = ET.parse(f)
            root = tree.getroot()

        # 解析行
        rows_data = []
        sheet_data = root.find("main:sheetData", ns)
        if sheet_data is None:
            return pd.DataFrame()

        for row in sheet_data.findall("main:row", ns):
            row_cells = {}
            for c in row.findall("main:c", ns):
                ref = c.get("r", "")  # e.g. "A1", "B2"
                # 提取列字母
                col_letter = re.match(r"([A-Z]+)", ref).group(1) if ref else ""
                col_idx = _col_letter_to_idx(col_letter)

                cell_type = c.get("t", "")
                v_elem = c.find("main:v", ns)
                if v_elem is None or v_elem.text is None:
                    # 可能是 inlineStr 或空
                    is_elem = c.find("main:is", ns)
                    if is_elem is not None:
                        t = is_elem.find("main:t", ns)
                        val = t.text if t is not None else ""
                    else:
                        val = None
                else:
                    if cell_type == "s":
                        # 共享字符串索引
                        idx = int(v_elem.text)
                        val = shared_strings[idx] if idx < len(shared_strings) else ""
                    elif cell_type == "b":
                        # 布尔
                        val = v_elem.text == "1"
                    else:
                        # 数值 (空类型 = 数值, 或 "n")
                        val = _try_numeric(v_elem.text)

                row_cells[col_idx] = val

            if row_cells:
                max_idx = max(row_cells.keys())
                row_list = []
                for i in range(max_idx + 1):
                    row_list.append(row_cells.get(i, None))
                rows_data.append(row_list)

    if not rows_data:
        return pd.DataFrame()

    # 第一行作为 header
    header = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(rows_data[0])]
    data_rows = rows_data[1:]

    # 补齐列数
    n_cols = len(header)
    clean_rows = []
    for r in data_rows:
        if len(r) < n_cols:
            r = r + [None] * (n_cols - len(r))
        elif len(r) > n_cols:
            r = r[:n_cols]
        clean_rows.append(r)

    return pd.DataFrame(clean_rows, columns=header)


def _try_numeric(text):
    """尝试把字符串转为 int/float, 失败则返回原字符串."""
    if text is None:
        return None
    try:
        # 先尝试 int
        if "." not in text and "e" not in text.lower():
            return int(text)
        return float(text)
    except (ValueError, TypeError):
        return text


def _col_letter_to_idx(letters):
    """A→0, B→1, ..., Z→25, AA→26, ..."""
    letters = letters.upper()
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def read_input(path):
    """读入文件, 返回 DataFrame.

    T68 诚实报错: .xls (旧版 Excel binary, OLE2 Compound Document) —— 
    本函数仅支持 .xlsx (zip+XML 结构) 和 .csv, 拒绝旧 .xls。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return _read_csv(path)
    elif ext == ".xls":
        raise ValueError(
            f"检测到旧版 .xls 格式 ({os.path.basename(path)}), "
            f"请用 Excel 另存为 .xlsx 后重试。本项目不支持 .xls (需 xlrd 依赖)。"
        )
    elif ext == ".xlsx":
        try:
            return _read_xlsx_minimal(path)
        except Exception as e:
            raise RuntimeError(
                f"读取 {path} 失败: {e}\n"
                f"如果是 .xlsx 且标准库解析失败, 可尝试: pip install openpyxl"
            )
    else:
        raise ValueError(f"不支持的文件格式: {ext} (仅支持 .csv / .xlsx)")


# ---------------------------------------------------------------------------
# 3. 数据校验与质量分级
# ---------------------------------------------------------------------------

def validate_and_clean(df, depth_col, dip_col, dipdir_col, well_col):
    """校验 + 清洗, 返回 (clean_df, quality_report).

    quality_report 包含:
        n_raw: 原始行数
        n_removed_oob: 角度越界剔除数
        n_removed_nan: NaN/空行剔除数
        n_removed_dup: 重复行剔除数
        depth_nonmonotonic: 深度是否非单调 (警告)
        n_valid: 有效行数
        quality_grade: "A"(≥50行) / "B"(20-50行) / "F"(<20行, 拒绝)
        per_well_counts: {well_id: n_rows}
    """
    report = {
        "n_raw": len(df),
        "n_removed_oob": 0,
        "n_removed_nan": 0,
        "n_removed_dup": 0,
        "depth_nonmonotonic": False,
        "n_valid": 0,
        "quality_grade": "F",
        "per_well_counts": {},
        "well_col": well_col,
    }

    # 转数值
    df = df.copy()
    df["_depth"] = pd_numeric(df[depth_col])
    df["_dip"] = pd_numeric(df[dip_col])
    df["_dipdir"] = pd_numeric(df[dipdir_col])
    if well_col:
        df["_well"] = df[well_col].fillna("未知井").astype(str).str.strip()
    else:
        df["_well"] = "单井"

    # NaN 剔除
    mask_nan = df["_depth"].notna() & df["_dip"].notna() & df["_dipdir"].notna()
    report["n_removed_nan"] = int((~mask_nan).sum())
    df = df[mask_nan].copy()

    # 角度越界剔除
    mask_oob = (
        (df["_dip"] >= 0) & (df["_dip"] <= 90) &
        (df["_dipdir"] >= 0) & (df["_dipdir"] < 360)
    )
    report["n_removed_oob"] = int((~mask_oob).sum())
    df = df[mask_oob].copy()

    # 重复行计数 (基于三列)
    before_dup = len(df)
    df = df.drop_duplicates(subset=["_depth", "_dip", "_dipdir"])
    report["n_removed_dup"] = before_dup - len(df)

    # 深度单调性检查 (按井分组)
    for wid, g in df.groupby("_well"):
        depths = g["_depth"].values
        if len(depths) > 1 and not np.all(np.diff(depths) >= 0):
            report["depth_nonmonotonic"] = True
            break

    report["n_valid"] = len(df)

    # 按井统计
    for wid, g in df.groupby("_well"):
        report["per_well_counts"][wid] = len(g)

    # 质量分级
    n = report["n_valid"]
    if n < 20:
        report["quality_grade"] = "F"
    elif n < 50:
        report["quality_grade"] = "B"
    else:
        report["quality_grade"] = "A"

    return df, report


def pd_numeric(series):
    """安全转数值, 失败为 NaN."""
    try:
        import pandas as pd
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return pd.Series([float(x) if _is_float(x) else float("nan") for x in series])


def _is_float(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# 4. 坐标转换
# ---------------------------------------------------------------------------

def dip_dipdir_to_normal(dip_deg, dipdir_deg):
    """倾角 + 倾向 → ENU 单位法向.

    公式 (写死): n = (sinδ·sinα, sinδ·cosα, cosδ)
        δ = dip (倾角, °), α = dip_direction (倾向方位角, 从北顺时针, °).
    """
    d = np.deg2rad(np.asarray(dip_deg, dtype=np.float64))
    a = np.deg2rad(np.asarray(dipdir_deg, dtype=np.float64))
    nx = np.sin(d) * np.sin(a)
    ny = np.sin(d) * np.cos(a)
    nz = np.cos(d)
    nrm = np.stack([nx, ny, nz], axis=-1)
    # 单位化
    norms = np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = nrm / np.clip(norms, 1e-12, None)
    return nrm


# ---------------------------------------------------------------------------
# 5. 构建 net dict
# ---------------------------------------------------------------------------

def build_net_dict(clean_df, quality_report):
    """从清洗后 DataFrame 构建 net dict 列表.

    每个 net dict 包含:
        pos: (L, 3) — 一维近似 (深度, 0, 0)
        nrm: (L, 3) — ENU 单位法向
        nrm_full: (L, 3) — 同 nrm (兼容 auto_label_demo 管线)
        wid: str — 井号
        depth_m: (L,) — 原始深度
        dip: (L,) — 原始倾角
        dip_dir: (L,) — 原始倾向
        n: int — 裂隙数
    """
    nets = []
    for wid, g in clean_df.groupby("_well"):
        depth = g["_depth"].values.astype(np.float64)
        dip = g["_dip"].values.astype(np.float64)
        dipdir = g["_dipdir"].values.astype(np.float64)

        nrm = dip_dipdir_to_normal(dip, dipdir)
        pos = np.stack([depth, np.zeros_like(depth), np.zeros_like(depth)], axis=-1)

        net = {
            "pos": pos,
            "nrm": nrm,
            "nrm_full": nrm,
            "wid": str(wid),
            "depth_m": depth,
            "dip": dip,
            "dip_dir": dipdir,
            "n": len(depth),
        }
        nets.append(net)
    return nets


# ---------------------------------------------------------------------------
# 6. 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="T1: Excel/CSV 编录表入口 → net dict")
    ap.add_argument("--input", required=True, help="输入 .xlsx / .csv 文件路径")
    ap.add_argument("--out", required=True, help="输出 .pt 文件路径")
    ap.add_argument("--well-col", default=None, help="强制指定井号列名 (可选)")
    ap.add_argument("--quality-out", default=None, help="质量报告 JSON 路径 (可选)")
    ap.add_argument("--site-name", default="场地", help="场地名称 (用于报告)")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 读入
    print(f"[T1] 读取 {args.input} ...")
    df = read_input(args.input)
    print(f"[T1] 原始数据: {len(df)} 行, 列: {list(df.columns)}")

    # 列匹配
    depth_col, dip_col, dipdir_col, well_col = find_columns(df.columns)
    if args.well_col:
        well_col = args.well_col

    print(f"[T1] 列匹配: 深度={depth_col}, 倾角={dip_col}, 倾向={dipdir_col}, 井号={well_col}")

    if depth_col is None or dip_col is None or dipdir_col is None:
        missing = []
        if depth_col is None: missing.append("深度")
        if dip_col is None: missing.append("倾角")
        if dipdir_col is None: missing.append("倾向")
        print(f"[ERROR] 找不到必要列: {missing}")
        print(f"  可用列: {list(df.columns)}")
        print(f"  深度别名: {DEPTH_ALIASES}")
        print(f"  倾角别名: {DIP_ALIASES}")
        print(f"  倾向别名: {DIPDIR_ALIASES}")
        sys.exit(1)

    # 校验
    clean_df, quality_report = validate_and_clean(df, depth_col, dip_col, dipdir_col, well_col)
    print(f"[T1] 质量报告:")
    print(f"  原始行数: {quality_report['n_raw']}")
    print(f"  NaN剔除: {quality_report['n_removed_nan']}")
    print(f"  越界剔除: {quality_report['n_removed_oob']}")
    print(f"  重复剔除: {quality_report['n_removed_dup']}")
    print(f"  有效行数: {quality_report['n_valid']}")
    print(f"  深度非单调: {quality_report['depth_nonmonotonic']}")
    print(f"  质量等级: {quality_report['quality_grade']}")
    print(f"  各井裂隙数: {quality_report['per_well_counts']}")

    # 质量 F 档 → 拒绝
    if quality_report["quality_grade"] == "F":
        print(f"[ERROR] 数据不足: 有效行数 {quality_report['n_valid']} < 20, 无法产出组系表")
        sys.exit(1)

    # 构建 net dict
    nets = build_net_dict(clean_df, quality_report)
    print(f"[T1] 生成 {len(nets)} 口井的 net dict")

    # 落盘
    torch.save(nets, args.out)
    print(f"[T1] net dict → {args.out}")

    # 质量报告落盘
    quality_json = dict(quality_report)
    # 把 per_well_counts 的 key 确保是 str
    quality_json["per_well_counts"] = {str(k): v for k, v in quality_report["per_well_counts"].items()}
    quality_json["assumptions"] = [
        "井内位置用一维近似 pos=(深度,0,0), 编录表无三维轨迹",
        "法向转换公式: ENU = (sinδ·sinα, sinδ·cosα, cosδ)",
        "法向无向 (±n 同面), 聚类/组心需用 |cos| + 符号对齐",
    ]
    quality_json["input_file"] = os.path.abspath(args.input)
    quality_json["output_file"] = os.path.abspath(args.out)
    quality_json["n_wells"] = len(nets)

    q_out = args.quality_out or args.out.replace(".pt", "_quality.json")
    with open(q_out, "w", encoding="utf-8") as f:
        json.dump(quality_json, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"[T1] 质量报告 → {q_out}")

    print("[T1] 完成.")


if __name__ == "__main__":
    main()
