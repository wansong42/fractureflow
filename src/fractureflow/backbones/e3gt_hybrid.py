# -*- coding: utf-8 -*-
"""E3GT-Hybrid 骨干 (注册为 "e3gt_hybrid")。

复用 src/fractureflow/e3gt_hybrid.py 的混合注意力实现 (局部等变 + 全局自注意力
+ 交叉注意力读出)。与 e3gt_v4 同一 forward/loss 接口, 故可在同一固定指标下替换对比。
"""

from ..core import BACKBONES
from ..e3gt_hybrid import E3GTHybrid as _E3GTHybrid


@BACKBONES.register("e3gt_hybrid")
class E3GTHybridBackbone(_E3GTHybrid):
    """混合注意力等变回归。见 e3gt_hybrid.py 全文说明。"""
    pass
