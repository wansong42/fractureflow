# -*- coding: utf-8 -*-
"""M3 裂隙组发现模块: 可学习原型池 + 软 vMF 隶属度 + 聚合组中心。

- mu0: K 个可学习原型 (单位球面)
- 观测点软隶属度 alpha_ik ∝ exp(temp * <n_i, mu0_k>)
- 组中心 mu_hat_k = SphNorm( sum_i alpha_ik * n_i ) (观测法向加权聚合)
- Anderson 物理正则: 组中心与安德森候选面夹角惩罚
- Gumbel-softmax 硬分配可选项 (仅推理/直方图用)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def unit_norm(x, eps=1e-7):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def anderson_candidates(s1, s3, theta_deg=30.0):
    """s1,s3 [B,3] -> [B,2,3]: n_A=s3, n_B 从 s1 向 s3 偏 theta (度)"""
    z = unit_norm(s3)
    tang = unit_norm(s1 - (s1 * z).sum(-1, keepdim=True) * z + 1e-9)
    th = torch.deg2rad(torch.tensor(theta_deg, device=s1.device, dtype=s1.dtype))
    nB = tang * torch.cos(th) + z * torch.sin(th)
    return torch.stack([z, nB], dim=1)  # [B,2,3]


class SetModule(nn.Module):
    """原型池 -> 软隶属度 -> 组中心 (数据驱动) + Anderson 正则。"""

    def __init__(self, K, temp=10.0):
        super().__init__()
        self.K = K
        self.temp = nn.Parameter(torch.tensor(float(temp)))
        with torch.no_grad():
            m = torch.randn(K, 3)
            self.mu0 = nn.Parameter(unit_norm(m))
        self.anderson_scale = nn.Parameter(torch.tensor(30.0))

    def forward(self, nrm, mask):
        """nrm [B,L,3](观测点法向, 未观测为0), mask [B,L]
        -> alpha [B,L,K], mu_hat [B,K,3], pi [B,K], w [B,L,K]
        """
        B, L, _ = nrm.shape
        n_u = unit_norm(nrm) * mask.unsqueeze(-1)
        alpha_raw = F.softplus(self.temp) * torch.einsum(
            "ble,ke->blk", n_u, self.mu0)
        alpha = F.softmax(alpha_raw, dim=-1)
        w = mask.unsqueeze(-1) * alpha
        num = torch.einsum("blk,ble->bke", w, nrm)
        mu_hat = unit_norm(num + 1e-8)
        pi = w.sum(1) / (mask.sum(1, keepdim=True) + 1e-8)
        return alpha, mu_hat, pi, w

    def hard_assign(self, nrm, mask, tau=1.0):
        """Gumbel-softmax 硬采样 (训练中的可微离散路径, 目前仅监控用)"""
        B, L, _ = nrm.shape
        n_u = unit_norm(nrm) * mask.unsqueeze(-1)
        logits = F.softplus(self.temp) * torch.einsum(
            "ble,ke->blk", n_u, self.mu0)
        return F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)

    def fit_centers(self, nrm, mask, iters=8, temp=30.0):
        """无梯度迭代拟合组中心 (对照真值用)。返回 [B,K,3] 收敛中心"""
        with torch.no_grad():
            n_u = unit_norm(nrm) * mask.unsqueeze(-1)
            mu = self.mu0.detach().clone()[None].expand(nrm.shape[0], -1, -1)
            for _ in range(iters):
                c = torch.einsum("ble,bke->blk", n_u, mu)
                a = torch.softmax(temp * c, -1) * mask.unsqueeze(-1)
                mu = unit_norm(torch.einsum("blk,ble->bke", a, nrm) + 1e-8)
            return mu

    def anderson_loss(self, mu_hat, s1, s3):
        """组中心与最近 Anderson 候选的夹角 -> 平滑惩罚"""
        cand = anderson_candidates(s1, s3)               # [B,2,3]
        cos = torch.einsum("bke,bme->bkm", mu_hat, cand)
        cos = cos.abs().clamp(-1, 1)
        ang = torch.arccos(cos.amax(-1))                 # [B,K]
        loss = (ang / torch.clamp(self.anderson_scale.abs(), 5.0, 60.0)).pow(2).mean()
        return loss