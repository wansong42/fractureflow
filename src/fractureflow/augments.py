# -*- coding: utf-8 -*-
"""训练期增广 (可替换, 注册式)。

- so3: 整网一致随机旋转 (pos/法向/应力同步)。模型等变 + 指标 |cos| 不变 => 合法,
       强制学"几何->法向"关系, 打破对训练网固定朝向的记忆 (治过拟合)。
- domain_rand: 域随机化 (抄旧项目 v22 最高 ROI 一招)。逐批随机 obs_frac + 给观测法向加
       高斯噪声, 覆盖真实井的块金噪声, 专治"合成->真实域偏移"。
"""

import math
import numpy as np
import torch

from .core import AUGMENTS, unit


def rand_rot_matrix(rng):
    """随机 SO(3) 旋转矩阵 (det=+1), 单位四元数法。"""
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    return R.astype(np.float32)


@AUGMENTS.register("so3")
def so3_augment(b, rng, **kw):
    R = torch.tensor(rand_rot_matrix(rng), dtype=torch.float32, device=b["pos"].device)
    # 所有方向张量都必须同步旋转, 否则 set_dirs 与旋转后的 pred/nrm 不在同一标架,
    # 会让 set_dir / memb 监督信号错位。
    for key in ("pos", "nrm", "nrm_full", "s1", "s3", "set_dirs"):
        if isinstance(b.get(key), torch.Tensor):
            b[key] = b[key] @ R.T
    return b


@AUGMENTS.register("domain_rand")
def domain_rand_augment(b, rng, obs_frac_min=0.2, obs_frac_max=0.6,
                        noise_deg_max=25.0, resample_frac=True, **kw):
    """逐批随机观测比例 + 观测法向加噪 (v22 域随机化)。

    - resample_frac=True (默认, 用于合成预训练): 重采样 obs_frac, 让模型学会
      适应不同观测密度 (治合成->真实域偏移)。
    - resample_frac=False (用于真实微调): 保留数据自带 obs_mask, 仅给观测法向加噪,
      模拟测量噪声、提升鲁棒性, 不破坏真实观测结构。
    """
    B, L = b["mask"].shape
    if resample_frac:
        frac = float(rng.uniform(obs_frac_min, obs_frac_max))
        new_mask = torch.as_tensor(rng.random((B, L)) < frac, dtype=torch.float32,
                                   device=b["mask"].device)
        b["mask"] = new_mask
        b["nrm"] = b["nrm_full"] * new_mask[..., None]
    if noise_deg_max > 0:
        sigma = math.radians(float(rng.uniform(0, noise_deg_max)))
        nrm = b["nrm"] + torch.randn_like(b["nrm"]) * sigma
        b["nrm"] = unit(nrm) * b["mask"][..., None]
    return b
