# -*- coding: utf-8 -*-
"""E3GTHybrid v2 骨干 (注册为 "e3gt_hybrid_v2").

v2 = 混合注意力 (局部等变 + 全局自注意力 + 交叉注意力读出) + 站点级组方向校正
(Site-Aware Group Calibration) + 门控逐点残差. 路线A(set_ids) 下以神经网络突破几何
set_aware 17.14° 下界; 无 set_ids 时退化为几何 L1 后验等效方向 (不劣于 33.49°).

forward 返回 {pred, aniso, set_dir_pred, gate}, 与通用训练器 + dir/set_dir 损失兼容.
"""

from ..core import BACKBONES
from ..e3gt_hybrid_v2 import E3GTHybridV2 as _E3GTHybridV2


@BACKBONES.register("e3gt_hybrid_v2")
class E3GTHybridV2Backbone(_E3GTHybridV2):
    """混合注意力 v2 (站点级校正). 见 e3gt_hybrid_v2.py 全文说明。"""
    pass
