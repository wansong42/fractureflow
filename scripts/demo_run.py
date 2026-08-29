# -*- coding: utf-8 -*-
"""T3: 一键商务演示包 —— "这就是将来见客户的全部家当".

用法:
    python scripts/demo_run.py                    # 用内置样例跑
    python scripts/demo_run.py --input data/编录表.xlsx  # 自定义输入
    python scripts/demo_run.py --out-dir results/demo/   # 自定义输出目录
    python scripts/demo_run.py --generate-samples        # 仅生成样例文件

输出 (一条命令):
    1) 单井打标 net dict (.pt)
    2) 组系表 CSV
    3) 玫瑰图 PNG
    4) 极点图 PNG
    5) 中文报告 MD
    可直接打包发人.
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

DEMO_DIR = os.path.join(ROOT, "data", "demo")
OUT_DIR_DEFAULT = os.path.join(ROOT, "results", "demo")


# ---------------------------------------------------------------------------
# 样例数据生成
# ---------------------------------------------------------------------------

def _build_minimal_xlsx(path, headers, rows, sheet_name="Sheet1"):
    """构造最小有效 xlsx 文件 (标准库 only)."""
    import zipfile
    import xml.etree.ElementTree as ET

    shared_strings = []

    def ss_idx(s):
        if s not in shared_strings:
            shared_strings.append(s)
        return shared_strings.index(s)

    # 构建 sheet XML
    sheet_rows_xml = []
    # Header row
    header_cells = []
    for i, h in enumerate(headers):
        col_letter = chr(65 + i) if i < 26 else "A"
        ref = col_letter + "1"
        header_cells.append(f'<c r="{ref}" t="s"><v>{ss_idx(str(h))}</v></c>')
    sheet_rows_xml.append(f'<row r="1">{"".join(header_cells)}</row>')

    # Data rows
    for ri, row_data in enumerate(rows):
        cells = []
        for ci, val in enumerate(row_data):
            col_letter = chr(65 + ci) if ci < 26 else "A"
            ref = col_letter + str(ri + 2)
            if isinstance(val, str):
                cells.append(f'<c r="{ref}" t="s"><v>{ss_idx(val)}</v></c>')
            elif isinstance(val, (int, np.integer)):
                cells.append(f'<c r="{ref}"><v>{int(val)}</v></c>')
            elif isinstance(val, (float, np.floating)):
                cells.append(f'<c r="{ref}"><v>{val}</v></c>')
            else:
                cells.append(f'<c r="{ref}"><v>{val}</v></c>')
        sheet_rows_xml.append(f'<row r="{ri + 2}">{"".join(cells)}</row>')

    sheet_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
    <sheetData>{"".join(sheet_rows_xml)}</sheetData>
    </worksheet>'''

    ss_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{len(shared_strings)}" uniqueCount="{len(shared_strings)}">
    {"".join(f"<si><t>{_xml_escape(s)}</t></si>" for s in shared_strings)}
    </sst>'''

    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
        <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
        <Default Extension="xml" ContentType="application/xml"/>
        <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
        <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
        <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
        </Types>''')
        z.writestr("_rels/.rels", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
        <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
        </Relationships>''')
        z.writestr("xl/workbook.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
        <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
        </workbook>''')
        z.writestr("xl/_rels/workbook.xml.rels", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
        <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
        </Relationships>''')
        z.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        z.writestr("xl/sharedStrings.xml", ss_xml)


def _xml_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_good_sample():
    """Synthetic three-set borehole log (60 rows, grade A).

    Deterministic (rng seed 42) three fracture families with Gaussian
    angular scatter -- fully data-self-sufficient, no external data.
    """
    rng = np.random.default_rng(42)
    n_sets, n_per = 3, 20
    families = [(30.0, 60.0), (150.0, 45.0), (280.0, 70.0)]  # (dip_dir, dip)
    dip_dir = np.empty(n_sets * n_per)
    dip = np.empty(n_sets * n_per)
    for k, (a0, d0) in enumerate(families):
        sl = slice(k * n_per, (k + 1) * n_per)
        dip_dir[sl] = (a0 + rng.normal(0.0, 6.0, n_per)) % 360.0
        dip[sl] = np.clip(d0 + rng.normal(0.0, 5.0, n_per), 0.0, 90.0)
    order = rng.permutation(n_sets * n_per)  # interleave families down-hole
    dip_dir, dip = dip_dir[order], dip[order]
    depth = np.arange(n_sets * n_per, dtype=float) * 0.25  # 0.25 m spacing

    headers = ["深度(m)", "倾角(°)", "倾向(°)"]
    rows = [[round(float(depth[i]), 2), round(float(dip[i]), 2),
             round(float(dip_dir[i]), 2)] for i in range(len(depth))]

    os.makedirs(DEMO_DIR, exist_ok=True)
    path = os.path.join(DEMO_DIR, "sample_borehole_good.xlsx")
    _build_minimal_xlsx(path, headers, rows)
    return path


def generate_dirty_sample():
    """构造一个有质量问题的样例 (30 行, grade B + 越界/重复)."""
    rng = np.random.default_rng(123)
    n = 30
    depth = np.arange(n, dtype=float) * 1.0
    dip = rng.uniform(10, 80, n)
    dip_dir = rng.uniform(0, 360, n)

    # 故意插入 3 个越界值
    dip[5] = 95   # > 90
    dip[12] = -3  # < 0
    dip_dir[20] = 400  # >= 360
    # 插入 2 个重复行
    depth[25] = depth[0]
    dip[25] = dip[0]
    dip_dir[25] = dip_dir[0]
    depth[26] = depth[1]
    dip[26] = dip[1]
    dip_dir[26] = dip_dir[1]

    headers = ["深度(m)", "倾角(°)", "倾向(°)"]
    rows = [[round(float(depth[i]), 2), round(float(dip[i]), 2),
             round(float(dip_dir[i]), 2)] for i in range(n)]

    os.makedirs(DEMO_DIR, exist_ok=True)
    path = os.path.join(DEMO_DIR, "sample_borehole_dirty.xlsx")
    _build_minimal_xlsx(path, headers, rows)
    return path


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="T3: 一键商务演示包")
    ap.add_argument("--input", default=None, help="输入 Excel/CSV (默认用内置好样例)")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT, help="输出目录")
    ap.add_argument("--K", type=int, default=3,
                    help="组数 K (内置样例为 3 组; 自定义输入请按场地实际组数指定)")
    ap.add_argument("--site-name", default="演示井", help="场地名称")
    ap.add_argument("--generate-samples", action="store_true",
                    help="仅生成样例文件, 不跑流程")
    args = ap.parse_args()

    # 生成样例
    if args.generate_samples:
        p1 = generate_good_sample()
        p2 = generate_dirty_sample()
        print(f"[demo] 好样例 → {p1}")
        print(f"[demo] 坏样例 → {p2}")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # 确定输入文件
    if args.input:
        input_path = args.input
    else:
        # 用内置样例
        input_path = os.path.join(DEMO_DIR, "sample_borehole_good.xlsx")
        if not os.path.isfile(input_path):
            print("[demo] 内置样例未找到, 自动生成 ...")
            input_path = generate_good_sample()
            print(f"[demo] 样例 → {input_path}")

    print(f"{'='*60}")
    print(f"  T3: 一键商务演示")
    print(f"  输入: {input_path}")
    print(f"  输出: {args.out_dir}")
    print(f"{'='*60}")

    # 串联: borehole_report.py 一条命令完成
    import subprocess
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "borehole_report.py"),
           "--input", input_path,
           "--out-dir", args.out_dir,
           "--K", str(args.K),
           "--site-name", args.site_name]
    print(f"[demo] 运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)

    if result.returncode != 0:
        print(f"[ERROR] borehole_report.py 失败 (exit {result.returncode})")
        sys.exit(result.returncode)

    # 检查输出
    expected_files = ["borehole_report.md", "rose_diagram.png",
                      "stereonet.png", "group_table.csv"]
    print()
    print(f"{'='*60}")
    print(f"  演示包产出")
    print(f"{'='*60}")
    for f in expected_files:
        fp = os.path.join(args.out_dir, f)
        exists = os.path.isfile(fp)
        status = "✓" if exists else "✗"
        print(f"  [{status}] {f}")
    print()
    print(f"  全部文件在: {os.path.abspath(args.out_dir)}")
    print(f"  可直接打包:  zip -r demo.zip {args.out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
