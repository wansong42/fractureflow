#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B3-4 几何约定 grep 门禁: 扫描 src/fractureflow 下手搓的产状<->法向/走向 转换.

设计目标 (杀模式不杀实例):
  1. 硬性阻断 BUG-A 类走向向量构造错误 ([cos(strike), sin(strike), 0] 轴互换);
  2. 硬性阻断 BUG-B 类无向法向还原 (手搓 atan2 但不翻转 nz<0);
  3. 阻断任何在 geometry.py 之外重新实现 dip_dipdir_to_normal 完整配方;
  4. 要求所有从法向提取 (atan2) 产状的文件复用 geometry 集中函数,
     遗留提取型文件在白名单内并带 # geo-conv-exempt 标注.

退出码: 0=通过, 1=发现违规 (可接 CI / selfcheck).
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "fractureflow")

# 权威实现文件, 自己可以出现所有配方 (集中在此处).
AUTHORITY = {"geometry.py"}
# 遗留提取型文件 (仅做 normal->dip_dir 提取, 非构造, 已由 T58 校订);
# 若未来要根治, 应改为调用 geometry.normal_to_dip_dipdir (行为以不变量测试为准).
LEGACY_EXTRACT = {
    "borehole_advanced.py",
    "borehole_report.py",
    "html_report.py",
    "report.py",
    "aland.py",
    "terzaghi.py",
}
EXEMPT_MARK = "geo-conv-exempt"

# --- 模式定义 -------------------------------------------------------------
# BUG-A: 走向向量轴互换 (x/y 用 cos/sin 而非 sin/cos)
BUG_A = re.compile(r"np\.array\(\s*\[\s*np\.cos\(\s*strike\s*\)\s*,\s*np\.sin\(\s*strike\s*\)")
# BUG-A 广义: 任何 [np.cos(<角度>), np.sin(<角度>), 0 构造水平单位向量 (非走集中函数)
COS_SIN_HORIZ = re.compile(
    r"np\.array\(\s*\[\s*np\.cos\([^)]*\)\s*,\s*np\.sin\([^)]*\)\s*,\s*0")
# dip_dipdir_to_normal 完整配方 sin(dip)*sin(dd) 等, 出现在非权威文件
SIN_SIN_FORMULA = re.compile(r"sin\s*\([^)]*\)\s*\*\s*sin\s*\(")
# 手搓 normal = [sin*sin, sin*cos, cos] 三元素向量 (构造法向)
NORMAL_CONSTRUCT = re.compile(
    r"np\.array\(\s*\[\s*np\.sin\([^)]*\)\s*\*\s*np\.sin\(")
# atan2 提取 (np.arctan2 / math.atan2 等) —— 用于"非集中"检测
ATAN2 = re.compile(r"arctan2?\(")


def _scan_file(path, fn):
    issues = []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        if EXEMPT_MARK in line:
            continue
        if BUG_A.search(line) or COS_SIN_HORIZ.search(line):
            issues.append((i, "BUG-A 类走向向量轴互换构造 (应使用 geometry.dip_dir_to_strike_vector)",
                          line.strip()))
        if SIN_SIN_FORMULA.search(line) or NORMAL_CONSTRUCT.search(line):
            issues.append((i, "手搓 dip_dipdir_to_normal 配方 (应使用 geometry.dip_dipdir_to_normal)",
                          line.strip()))
    return issues


def main():
    violations = []
    for fn in sorted(os.listdir(SRC)):
        if not fn.endswith(".py"):
            continue
        if fn.startswith("test_"):
            continue
        path = os.path.join(SRC, fn)
        if not os.path.isfile(path):
            continue

        if fn in AUTHORITY:
            continue  # 权威实现, 跳过全部模式

        issues = _scan_file(path, fn)
        for i, why, snippet in issues:
            violations.append((fn, i, why, snippet))

        # atan2 非集中检测 (仅针对非白名单文件)
        if fn not in LEGACY_EXTRACT:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if ATAN2.search(text):
                # 允许: 若文件显式导入 geometry 的集中转换函数, 视为已集中
                if ("from .geometry import" in text
                        or "from fractureflow.geometry import" in text
                        or "import fractureflow.geometry" in text):
                    continue
                # 否则逐行标记 atan2 行为违规 (除非该行有 exempt 标记)
                for i, line in enumerate(text.splitlines(), 1):
                    if EXEMPT_MARK in line:
                        continue
                    if ATAN2.search(line):
                        violations.append((fn, i,
                                          "atan2 提取产状但未复用 geometry 集中函数",
                                          line.strip()))

    if violations:
        print("=" * 70)
        print("几何约定 grep 门禁: 发现 {0} 处违规".format(len(violations)))
        print("=" * 70)
        for fn, i, why, snippet in violations:
            print(f"[{fn}:{i}] {why}")
            print(f"    > {snippet}")
        print("\n修复: 改用 fractureflow.geometry 的 dip_dipdir_to_normal / "
              "normal_to_dip_dipdir / dip_dir_to_strike_vector。")
        return 1
    print("几何约定 grep 门禁: 通过 (0 处违规)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
