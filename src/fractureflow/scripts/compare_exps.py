# -*- coding: utf-8 -*-
"""对比聚合: 读 results/*_eval.json, 统一逐点 acos 口径下各实验 vs 几何基线 top2_ens。

铁律: 绝不切换口径。所有数字均为隐伏点逐点 acos(|<pred,true>|) 度。
输出排序表: 实验名 | 模型 MAE | 基线 MAE | delta | P50 | P90, 一眼看出谁真在进步。

用法:
  cd src && python -m fractureflow.scripts.compare_exps [--dir results] [--glob "*_eval.json"]
"""

import argparse
import glob
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--glob", default="*_eval.json")
    args = ap.parse_args()
    pat = os.path.join(ROOT, args.dir, args.glob)
    files = sorted(glob.glob(pat))
    rows = []
    for fp in files:
        try:
            d = load(fp)
        except Exception as e:
            print(f"skip {fp}: {e}")
            continue
        if "model" not in d or "geometric_baseline_top2_ens" not in d:
            continue
        name = os.path.basename(fp).replace("_eval.json", "")
        m = d["model"]; g = d["geometric_baseline_top2_ens"]
        rows.append((name, m, g))
    if not rows:
        print("no eval json found")
        return
    rows.sort(key=lambda r: r[1]["mae"])
    print(f"\n{'experiment':32s} {'model':>7s} {'baseline':>8s} {'delta':>7s} {'p50':>6s} {'p90':>6s} {'n_hid':>6s}")
    print("-" * 78)
    for name, m, g in rows:
        delta = m["mae"] - g["mae"]
        print(f"{name:32s} {m['mae']:7.2f} {g['mae']:8.2f} {delta:+7.2f} "
              f"{m['p50']:6.2f} {m['p90']:6.2f} {m['n_hid']:6d}")
    print("-" * 78)
    print("全部为逐点 acos(|<pred,true>|) 度; delta<0 表示模型优于几何基线 top2_ens。")


if __name__ == "__main__":
    main()
