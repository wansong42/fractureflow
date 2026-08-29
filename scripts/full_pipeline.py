# -*- coding: utf-8 -*-
"""一键全管线: LAS/CSV/npz → 打标 → DFN → 渗流 → 报告.

用法:
    # LAS 输入
    python scripts/full_pipeline.py --input-las data/x.las --domain 50 50 50

    # npz 输入 (wells x normals)
    python scripts/full_pipeline.py --set-table your_wells.npz --domain 50 50 50

    # 已有 set_ids 的 pt 文件
    python scripts/full_pipeline.py --set-table results/x_routeA.pt --domain 50 50 50
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))


def run_cmd(cmd, desc):
    """运行子命令并检查返回码."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[ERROR] 步骤失败 (exit {result.returncode}): {desc}")
        sys.exit(result.returncode)
    return result


def main():
    ap = argparse.ArgumentParser(description='一键全管线: 数据 → 打标 → DFN → 渗流 → 报告')
    ap.add_argument('--input-las', default=None, help='FMI LAS 文件路径')
    ap.add_argument('--set-table', default=None, help='含 set_ids 的 npz/pt 文件')
    ap.add_argument('--domain', nargs=3, type=float, default=[50, 50, 50],
                    help='域尺寸 (m)')
    ap.add_argument('--seeds', type=int, default=20, help='每 P32 点实现数')
    ap.add_argument('--beta', type=float, default=3.5, help='幂律指数 β')
    ap.add_argument('--site-name', default='场地', help='场地名称')
    ap.add_argument('--out-dir', default='results/full_pipeline', help='输出目录')
    args = ap.parse_args()

    if args.input_las is None and args.set_table is None:
        ap.error("需要 --input-las 或 --set-table")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.input_las:
        # 步骤 1: LAS → 打标 → pt (用 auto_label_borehole)
        las_basename = os.path.splitext(os.path.basename(args.input_las))[0]
        pt_path = os.path.join(args.out_dir, f"{las_basename}_routeA.pt")
        cmd1 = [sys.executable, "scripts/auto_label_borehole.py",
                "--input-las", args.input_las, "--out", pt_path]
        run_cmd(cmd1, f"Step 1: LAS → 打标 ({args.input_las})")
        set_table_path = pt_path
    else:
        set_table_path = args.set_table

    # 步骤 2: SetTable → DFN → 渗流 → 报告
    domain_str = " ".join(str(d) for d in args.domain)
    out_base = os.path.join(args.out_dir, "dfn_report")
    cmd2 = [sys.executable, "scripts/dfn_from_borehole.py",
            "--set-table", set_table_path,
            "--domain"] + [str(d) for d in args.domain] + [
            "--seeds", str(args.seeds),
            "--beta", str(args.beta),
            "--site-name", args.site_name,
            "--out", f"{out_base}.json"]
    run_cmd(cmd2, f"Step 2: DFN → 渗流 → 报告 (domain={domain_str})")

    print(f"\n{'='*60}")
    print(f"  ✓ 全管线完成!")
    print(f"  JSON: {out_base}.json")
    print(f"  报告: {out_base}_report.md")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
