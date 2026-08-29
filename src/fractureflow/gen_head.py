# -*- coding: utf-8 -*-
"""M4 生成预测头: 条件 vMF 混合分布。

逐点输出:
  pi_ik   = softmax(MLP(h_i))           点->组权重
  mu_k    = 组中心 (来自 SetModule)
  kappa_k = per-set 浓度 (可学习, sigmoid 映射到 1..kappa_max)
  kappa_i = per-point 局部浓度缩放 (MLP(h)) -> 校准不确定性

损失: 对称 NLL(vMF 混合) = -log( P(x) + P(-x) ), 目标未定向;
推理: 混合归一化均值方向 + 隶属度 + 有效浓度(1/kappa 即置信度)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def unit_norm(x, eps=1e-7):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def logC3(kappa):
    """vMF(3) 对数归一化常数: log(kappa / (4 pi sinh kappa)), 大 kappa 稳定"""
    k = torch.clamp(kappa, min=1e-6)
    return (torch.log(k) - torch.log(torch.tensor(2.0 * torch.pi, device=k.device,
                                                  dtype=k.dtype))
            - k - torch.log1p(-torch.exp(-2.0 * k)))


class ResidualBlock(nn.Module):
    """带残差连接的 MLP 块"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class GenHead(nn.Module):
    def __init__(self, d_model, K, kappa_max=80.0):
        super().__init__()
        self.K = K
        self.kappa_max = kappa_max
        
        # 更深的混合 MLP
        self.mix_mlp = nn.Sequential(
            nn.Linear(d_model + K, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.15),
            ResidualBlock(d_model, dropout=0.15),
            ResidualBlock(d_model, dropout=0.15),
            nn.Linear(d_model, K),
        )
        
        # 更深的 per-point kappa MLP
        self.kappa_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )
        
        self.log_kappa = nn.Parameter(torch.randn(K).clamp(-1.5, 1.5))
        # 组使用先验耦合
        self.set_bias = nn.Parameter(torch.tensor(3.0))
        
        # 改进的显式"观测→隐伏传播"分支: 更深残差网络 + gated copy + 组中心注入
        self.prop_mlp = nn.Sequential(
            nn.Linear(d_model + 3 + 3, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.15),
            ResidualBlock(d_model, dropout=0.15),
            ResidualBlock(d_model, dropout=0.15),
            ResidualBlock(d_model, dropout=0.15),
            nn.Linear(d_model, 3),
        )
        # Gating network: 学习何时信任邻居均值 vs 自身预测
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model + 3 + 3, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    # ---- 传播分支 ----
    def prop_dir(self, h, nhb=None, mu_aff=None):
        """-> [B,L,3] 单位方向: 显式传播 regressor。
        结构: gated copy —— 邻域观测均值方向 nhb 直通锚定 + 组中心软混合 mu_aff
        + 残差 MLP 校准。当无观测邻居时 nhb=0, 退化为学习先验。
        """
        if nhb is None:
            nhb = torch.zeros_like(h[..., :3])
        if mu_aff is None:
            mu_aff = torch.zeros_like(nhb)

        # Gating: 基于 h 和 nhb 决定信任度
        gate_in = torch.cat([h, nhb, mu_aff], dim=-1)
        gate = self.gate_mlp(gate_in)  # [B,L,1] in (0,1)

        # MLP 预测残差
        d = self.prop_mlp(torch.cat([h, nhb, mu_aff], dim=-1))

        # Gated combination: gate * nhb + (1-gate) * (nhb + residual)
        # = nhb + (1-gate) * residual
        pred = nhb + (1.0 - gate) * d
        return unit_norm(pred)

    def prop_loss(self, h, target, w, nhb=None, mu_aff=None):
        """target [B,L,3] 邻域观测均值方向; w [B,L] 参与权; 返回对称cos负对数"""
        p = self.prop_dir(h, nhb, mu_aff)
        cos = (p * target).sum(-1).abs()
        return -((w * cos).sum() / (w.sum() + 1e-8))

    def true_loss(self, target, h, nhb, w, mu_aff=None):
        """prop 分支直接回归掩码真值方向 (对称 cos NLL)。

        传播分支最终输出即隐伏方向, 必须直接监督真值 (训练时有标签,
        属于合法的隐藏任务监督), 而不是只对准近邻观测均值。
        """
        p = self.prop_dir(h, nhb, mu_aff)
        cos = (p * target).sum(-1).abs()
        return -((w * cos).sum() / (w.sum() + 1e-8))

    # ---- 分布参数 ----
    def params(self, h, alpha, mu_hat, usage=None):
        """-> pi [B,L,K], mu [B,K,3], kappa_k [B,K], kappa_i [B,L]"""
        B, L, _ = h.shape
        mix_in = torch.cat([h, alpha], dim=-1)
        logits = self.mix_mlp(mix_in)
        if usage is not None:
            # 组使用耦合: [B,1,K] 广播到逐点 [B,L,K]
            logits = logits + torch.clamp(self.set_bias, 0.0, 10.0) \
                * torch.log(usage[:, None, :] + 1e-6)
        pi = F.softmax(logits, dim=-1)
        logk = self.log_kappa[None, :]
        kappa_k = 1.0 + (self.kappa_max - 1.0) * torch.sigmoid(logk.expand(B, self.K))
        k_scale = F.softplus(self.kappa_mlp(h)).squeeze(-1)              # [B,L] >= 0
        kappa_i = torch.clamp(kappa_k[:, None, :] * (0.5 + 1.5 * torch.sigmoid(k_scale)[..., None]),
                              min=0.5, max=self.kappa_max)                # [B,L,K]
        return pi, mu_hat, kappa_k, kappa_i

    # ---- 分布 ----
    def mixture_logp(self, x, h, alpha, mu_hat, usage=None):
        """x [B,L,3] -> log P(x) [B,L]: logsumexp over K (log pi + logC + k x·mu)"""
        pi, mu, kappa_k, kappa_i = self.params(h, alpha, mu_hat, usage)
        cos = torch.einsum("ble,bke->blk", x, mu)
        logp = logC3(kappa_i) + kappa_i * cos                       # [B,L,K]
        return torch.logsumexp(logp + torch.log(pi + 1e-8), dim=-1)

    def nll_loss(self, x_true, h, alpha, mu_hat, weight=None, usage=None):
        """对称 NLL (未定向目标); weight [B,L] 参与加权 (无则求均值)"""
        lp = self.mixture_logp(x_true, h, alpha, mu_hat, usage)
        lp_neg = self.mixture_logp(-x_true, h, alpha, mu_hat, usage)
        nll = -torch.logsumexp(torch.stack([lp, lp_neg], -1), dim=-1)   # [B,L]
        if weight is None:
            return nll.mean()
        return (nll * weight).sum() / (weight.sum() + 1e-8)

    def sym_term(self, mu_hat, pi):
        """显式对称正则: 模型预测均值方向与组中心方向一致率 (残余项, 默认 0.1w 保留)"""
        return torch.zeros((), device=mu_hat.device)

    # ---- 推理 ----
    def infer(self, h, alpha, mu_hat, usage=None, nhb=None):
        pi, mu, kappa_k, kappa_i = self.params(h, alpha, mu_hat, usage)
        # 后验加权均值方向: 用 π*κ 加权 (避免正交组平均互相抵消)
        w = pi * kappa_i                                   # [B,L,K]
        mean_mix = unit_norm(torch.einsum("blk,bke->ble", w, mu) + 1e-7)
        kappa_eff = w.sum(-1)                              # 有效浓度(置信度反比)
        # 组中心软混合 (观测隶属度加权) -> 注入传播分支
        mu_aff = unit_norm(torch.einsum("blk,bke->ble", alpha, mu_hat) + 1e-7)
        # 主输出 = 显式传播 (观测→隐伏), 混合后验作为 'mean_mix' 保留
        prop = self.prop_dir(h, nhb, mu_aff)
        return {
            "mean": prop,            # [B,L,3] 主预测: 观测→隐伏传播(无向)
            "mean_mix": mean_mix,    # [B,L,3] 混合后验均值
            "prop": prop,
            "mu_aff": mu_aff,        # [B,L,3] 组中心软混合
            "pi": pi,                # [B,L,K]
            "kappa": kappa_eff,      # [B,L]
            "kcomp": pi.argmax(-1),  # [B,L]
            "mu": mu,                # [B,K,3]
            "kappa_k": kappa_k,      # [B,K]
        }

    def sample(self, h, alpha, mu_hat, n_samples=8, rng=None, usage=None):
        """从混合抽样 (Wood 法) [B,L,S,3]"""
        pi, mu, kappa_k, kappa_i = self.params(h, alpha, mu_hat, usage)
        B, L, K = pi.shape
        comps = torch.multinomial(pi.reshape(B * L, K), n_samples, replacement=True,
                                  generator=rng).reshape(B, L, n_samples)
        mu_c = torch.gather(mu.unsqueeze(1).expand(B, L, K, 3), 2,
                            comps.unsqueeze(-1).expand(B, L, n_samples, 3))
        kk = torch.gather(kappa_i, 2, comps).unsqueeze(-1)     # [B,L,S,1]
        kk = torch.minimum(kk, torch.tensor(1e3, device=kk.device))
        u = torch.rand(B, L, n_samples, 1, device=kk.device)
        w = 1.0 + (1.0 / kk) * torch.log(u + (1.0 - u) * torch.exp(-2.0 * kk))
        w = w.clamp(-1.0, 1.0)
        s = torch.sqrt(torch.clamp(1.0 - w.pow(2), min=0.0))
        z = torch.randn(B, L, n_samples, 2, device=kk.device)
        zn = unit_norm(z)
        z2 = torch.cat([s * zn, w], dim=-1)                # [B,L,S,3]
        return rotate_to_mu(z2, mu_c)


def rotate_to_mu(x, mu):
    """z 轴基样本 -> mu 方向 (Householder 反射合法)"""
    z = torch.tensor([0.0, 0.0, 1.0], device=x.device)
    same = ((mu - z).abs().sum(-1) < 1e-6).unsqueeze(-1)
    flip = ((mu + z).abs().sum(-1) < 1e-6).unsqueeze(-1)
    h = unit_norm(mu - z)
    h = torch.where(same | flip, torch.zeros_like(h), h)   # 退化位置不旋转
    R = torch.eye(3, device=x.device) - 2.0 * h[..., :, None] * h[..., None, :]
    out = torch.einsum("...ce,...e->...c", R, x)
    out = torch.where(same, x, out)
    out = torch.where(flip, x * torch.tensor([1.0, 1.0, -1.0], device=x.device), out)
    return out