# -*- coding: utf-8 -*-
"""T91 脏数据交付演练修复回归 — L5 铁律 (每修一个 bug 提炼成单测).

覆盖:
  1. _fmt_score 混合类型 scores 字典安全格式化 (dfn_from_borehole.py:137 崩溃回归)
  2. read_borehole_csv depth/dip/dip_direction 方言自动转换 (客户标准格式)
  3. read_borehole_csv 空值/非数字行跳过 + 计数警告
  4. read_borehole_csv 缺列时带实际列清单的明确报错
  5. read_single_las 缺 Azimuth/Dip 列 -> 行跳过而非 TypeError 崩溃
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from scripts.dfn_from_borehole import _fmt_score
from scripts.auto_label_borehole import read_borehole_csv
from scripts.read_forge_las import dip_dipdir_to_normal, read_single_las


def test_fmt_score_mixed_dict():
    """auto_select_K 返回混 float/int/str 的 scores; 旧 f'{v:.3f}' 对 str 崩."""
    scores = {2: 0.1234, 3: 0.5678, "K_silhouette": 3, "K_data_cap": 12,
              "K_fusion": 3, "fusion_rule": "min(K_sil=3, K_data=12)=3"}
    out = ", ".join(_fmt_score(k, v) for k, v in scores.items())
    assert "2:0.123" in out and "K_fusion:3" in out
    assert "min(K_sil=3" in out          # str 值原样保留
    assert ".000" not in out.split("K_fusion")[1][:6]  # int 不被强转浮点串


def test_read_csv_dialect_dip_dipdir(tmp_path):
    """depth/dip/dip_direction 编录表方言 -> 法向, 与参考转换逐点一致."""
    dips = [10.0, 45.0, 80.0, 30.0]
    dds = [0.0, 90.0, 200.5, 359.0]
    p = tmp_path / "dialect.csv"
    lines = ["depth,dip,dip_direction"]
    for i, (dp, dd) in enumerate(zip(dips, dds)):
        lines.append(f"{100 + i},{dp},{dd}")
    p.write_text("\n".join(lines), encoding="utf-8-sig")

    net = read_borehole_csv(str(p))
    ref = dip_dipdir_to_normal(np.array(dds), np.array(dips))
    assert np.allclose(net["nrm"], ref, atol=1e-9)


def test_read_csv_skips_bad_rows_with_warning(tmp_path, capsys):
    """空值/垃圾文本/零向量行跳过并打印计数警告."""
    p = tmp_path / "dirty.csv"
    rows = ["id,nx,ny,nz",
            "0,0.707,0,0.707",
            "1,,0.5,0.5",           # 空值
            "2,N/A,0.1,0.9",        # 垃圾文本
            "3,0,0,0",              # 零向量
            "4,0.6,0.8,0"]
    p.write_text("\n".join(rows), encoding="utf-8-sig")
    net = read_borehole_csv(str(p))
    assert len(net["nrm"]) == 2       # 行 0 和 4
    out = capsys.readouterr().out
    assert "跳过无效行 2 个" in out and "零向量行 1 个" in out


def test_read_csv_missing_columns_clear_error(tmp_path):
    """缺列时报错必须列出缺失列名与实际列清单 (不再是裸 KeyError)."""
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8-sig")
    try:
        read_borehole_csv(str(p))
        raise AssertionError("应抛 RuntimeError")
    except RuntimeError as e:
        msg = str(e)
        assert "缺少必需列" in msg and "nx" in msg and "'a', 'b'" in msg


def test_las_missing_orientation_columns(tmp_path):
    """LAS 有 ~CURVE 但无 Azimuth/Dip_TRU -> 行跳过 -> 明确 'no fracture rows'
    报错, 而非 parts[None] TypeError 崩溃 (T91 D11 回归)."""
    p = tmp_path / "no_orient.las"
    p.write_text("~CURVE\nMD.M\nGR.gAPI\n~ASCII\n100 55\n101 56\n",
                 encoding="utf-8")
    try:
        read_single_las(str(p))
        raise AssertionError("应抛 RuntimeError")
    except RuntimeError as e:
        assert "no fracture rows" in str(e)
