# -*- coding: utf-8 -*-
"""E3GTV4 判别式等变向量回归骨干 (注册为 "e3gt_v4")。

直接复用 src/fractureflow/e3gt_v4.py 的 E3GTV4 实现 (已修复 Eᵀ 逆旋转 bug)。
forward 返回 {pred, aniso, set_dir_pred}, 供通用训练器 + dir/set_dir 损失使用。
"""

from ..core import BACKBONES
from ..e3gt_v4 import E3GTV4 as _E3GTV4


@BACKBONES.register("e3gt_v4")
class E3GTBackbone(_E3GTV4):
    """等变向量传播回归 (判别)。见 e3gt_v4.py 全文说明。"""
    pass
