# -*- coding: utf-8 -*-
"""模块化核心: 注册表 + 通用工具。

设计目的(用户要求): 把"裂隙法向预测"做成乐高式可快速替换的实验框架。
- 每个可替换组件 (backbone / augment / loss) 用 `@REG.register("名字")` 登记。
- 实验配置(JSON)里只写组件名字 + 超参; 换组件 = 改配置一行, 不动代码。
- 这样"不行就换"能真落地, 大范围试错/多次重构成本低。

评测口径铁律: 所有指标一律逐点 acos(|<pred,true>|), 绝不切换口径凑达标
(详见 docs/开发日志.md §0 与 §6)。
"""

import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Registry:
    """极简名字->类 注册表。"""

    def __init__(self, name):
        self.name = name
        self._d = {}

    def register(self, key):
        def deco(cls):
            if key in self._d:
                raise ValueError(f"{self.name} 名字冲突: '{key}' 已注册")
            self._d[key] = cls
            cls._registry_name = key
            return cls
        return deco

    def __getitem__(self, key):
        if key not in self._d:
            avail = ", ".join(sorted(self._d))
            raise KeyError(f"未知 {self.name}: '{key}'。可用: [{avail}]")
        return self._d[key]

    def get(self, key):
        return self._d.get(key)

    def keys(self):
        return list(self._d)


# 三类可替换组件注册表
BACKBONES = Registry("backbone")
AUGMENTS = Registry("augment")
LOSSES = Registry("loss")


def unit(x, eps=1e-8):
    """单位化最后一维向量。"""
    return x / (x.norm(dim=-1, keepdim=True) + eps)
