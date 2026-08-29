# -*- coding: utf-8 -*-
"""T2: 钻孔组系表中文报告生成.

串联: net dict (T1 输出) → set_table 估计 → 中文报告 (玫瑰图 + 极点图 + 组系表).

用法:
    # 从 T1 输出
    python scripts/borehole_report.py --net results/borehole_net.pt --out-dir results/demo/

    # 从 Excel 直接 (内部调 T1)
    python scripts/borehole_report.py --input data/编录表.xlsx --out-dir results/demo/

    # 指定 K
    python scripts/borehole_report.py --net results/x.pt --K 6 --out-dir results/demo/
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fractureflow.borehole_report import generate_borehole_report
from fractureflow.set_table_eval import estimate_set_table
from fractureflow.setlabel import spherical_kmeans, _sign_align


def main():
    ap = argparse.ArgumentParser(description="T2: 钻孔组系表中文报告")
    ap.add_argument("--net", default=None, help="T1 输出 net dict (.pt)")
    ap.add_argument("--input", default=None, help="Excel/CSV 文件 (内部调 T1)")
    ap.add_argument("--out-dir", default="results/demo", help="输出目录")
    ap.add_argument("--K", type=int, default=6, help="组数 K (默认 6)")
    ap.add_argument("--seed", type=int, default=42, help="k-means 种子 (默认 42)")
    ap.add_argument("--site-name", default="场地", help="场地名称")
    args = ap.parse_args()

    if args.net is None and args.input is None:
        ap.error("需要 --net 或 --input 之一")

    os.makedirs(args.out_dir, exist_ok=True)

    quality_report = None

    if args.input:
        # 内部调 T1 流程
        print(f"[T2] 从 {args.input} 读入 ...")
        from borehole_excel_entry import read_input, find_columns, validate_and_clean, build_net_dict

        df = read_input(args.input)
        depth_col, dip_col, dipdir_col, well_col = find_columns(df.columns)
        if depth_col is None or dip_col is None or dipdir_col is None:
            print("[ERROR] 找不到必要列 (深度/倾角/倾向)")
            sys.exit(1)

        clean_df, quality_report = validate_and_clean(
            df, depth_col, dip_col, dipdir_col, well_col)

        if quality_report["quality_grade"] == "F":
            print(f"[ERROR] 数据不足: {quality_report['n_valid']} < 20")
            sys.exit(1)

        nets = build_net_dict(clean_df, quality_report)
    else:
        # 从 T1 net dict 加载
        import torch
        print(f"[T2] 从 {args.net} 加载 net dict ...")
        nets = torch.load(args.net, weights_only=False)
        if isinstance(nets, dict):
            nets = [nets]

        # 尝试加载伴随的质量报告
        q_path = args.net.replace(".pt", "_quality.json")
        if os.path.isfile(q_path):
            with open(q_path, "r", encoding="utf-8") as f:
                quality_report = json.load(f)

    # 合并所有井的法向 (全量, 用于建组系表)
    all_nrm = np.vstack([np.asarray(net["nrm"], dtype=np.float64) for net in nets])
    print(f"[T2] 总裂隙数: {len(all_nrm)}, K={args.K}")

    # 估计组系表 (strict, obs-only 精神: 全量法向作组心)
    K = args.K
    pos = np.zeros((len(all_nrm), 3))
    occ = np.ones(len(all_nrm), dtype=bool)

    st = estimate_set_table(pos, all_nrm, occ, K=K, seed=args.seed, strict=True)
    centers = st["centers"]
    assign = st["assign"]
    print(f"[T2] 组系表: {centers.shape[0]} 组")

    # 生成报告
    report_md, md_path, rose_path, stereo_path = generate_borehole_report(
        nets, st, quality_report or {}, args.site_name, args.out_dir)

    print(f"[T2] 报告 → {md_path}")
    print(f"[T2] 玫瑰图 → {rose_path}")
    print(f"[T2] 极点图 → {stereo_path}")

    # 落盘组系表 CSV
    from fractureflow.borehole_report import build_group_table
    df_table = build_group_table(all_nrm, assign, centers, K)
    csv_path = os.path.join(args.out_dir, "group_table.csv")
    df_table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[T2] 组系表 CSV → {csv_path}")

    print("[T2] 完成.")


if __name__ == "__main__":
    main()
