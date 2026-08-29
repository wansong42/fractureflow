# -*- coding: utf-8 -*-
"""数据切分: 真实网络 80/10/10 (seed=42), 与官方协议一致。"""

import numpy as np
import torch


def load_split(real_path, seed=42):
    """返回 (train, val, test) 三个 net dict 列表。

    切分严格复刻 scripts/train_e3gt_v4.py: np.random.default_rng(42).permutation,
    保证与历史结果可比。
    """
    real = torch.load(real_path, weights_only=False)
    idx = np.random.default_rng(seed).permutation(len(real))
    n_tr = int(0.8 * len(real))
    n_va = int(0.1 * len(real))
    tr = [real[i] for i in idx[:n_tr]]
    va = [real[i] for i in idx[n_tr:n_tr + n_va]]
    te = [real[i] for i in idx[n_tr + n_va:]]
    return tr, va, te
