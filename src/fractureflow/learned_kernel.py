# -*- coding: utf-8 -*-
"""Learned Local Posterior Weighting (路线 A 之外的无标签学习方向).

把几何 SOTA `l1_local_dirs` (33.49°) 的固定核 (h=0.5, c=0.1) 升级为**逐点可学**
核宽 h_i 与收缩量 c_i: 19 维局部特征 -> 小 MLP -> 逐点 (h_i, c_i) -> 可微
Weiszfeld 解码 -> 直接对评测指标 acos|<pred,true>| 求梯度。

关键约束 (来自实现指南):
- 不改变 e3gt_hybrid 或任何已有模型文件, 不改 inference._l1_median_batch。
- 初始化 h_i≈0.5, c_i≈0.1 -> 起手即 l1_local 基线, 训练只能往下压 (不假突破)。
- 解码器与 inference._l1_median_batch 逐点等价 (仅 h,c 变为逐点), 故 h,c 全取
  常数 0.5/0.1 时应复现 l1_local_dirs。

泄漏安全性: 输入 nrm 对隐伏点置零; 训练 loss 只统计隐伏点 (目标 nrm_full 不参与
前向, 仅作真值); 特征只用观测信息 + 局部几何。
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import (knn_graph, local_frames, build_graph_features,
                        unit_n)


# ----------------------- KernelNet (逐点学 h_i, c_i) -----------------------
class KernelNet(nn.Module):
    def __init__(self, in_dim=20, hidden=32, clamp_h=(0.2, 1.0),
                 clamp_c=(0.02, 0.3), random_init=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
        self.clamp_h = tuple(clamp_h)
        self.clamp_c = tuple(clamp_c)
        # 起手 = l1_local 基线: h≈0.5, c≈0.1。零权重 => 初始输出完全由 bias 决定。
        if random_init:
            # 审查报告建议: 最后一层 weight 加 N(0,0.01) 打破对称 (bias 仍锁定基线起点)。
            nn.init.normal_(self.net[-1].weight, 0.0, 0.01)
        else:
            self.net[-1].weight.data.zero_()
            self.net[-1].bias.data = torch.tensor([
                math.log(0.5),   # log_h -> h = softplus(log(0.5)) ≈ 0.505
                math.log(0.1),   # log_c -> c = softplus(log(0.1)) ≈ 0.105
            ])

    def forward(self, feat):
        out = self.net(feat)                       # [B, L, 2]
        log_h, log_c = out[..., 0], out[..., 1]
        # 钳制在初始化 (h≈0.5, c≈0.1) 附近: 防 Weiszfeld 奇异梯度把 h,c 推到极端
        # (1/sin 项在 C≈±1 处梯度巨大, 大 lr 会一步陷入坏盆, 误差反而恶化)。
        h = (F.softplus(log_h) + 0.1).clamp(*self.clamp_h)      # 默认围绕 0.5
        c = (F.softplus(log_c) + 0.01).clamp(*self.clamp_c)     # 默认围绕 0.1
        return h, c


# ----------------------- 权重计算 -----------------------
def compute_weights(pos, h, c, mask):
    """pos[B,L,3], h[B,L], c[B,L], mask[B,L] -> W[B,L,L] (仅观测 j 列非零)。

    距离用 sqrt(||Δ||²+1e-6) 而非 torch.cdist: cdist 在重合点的反向梯度
    为 1/dist → NaN, 会污染参数 (前向有限、反向 NaN, isnan(loss) 拦不住)。
    """
    B = pos.shape[0]
    radius = pos.std(1).mean(-1).clamp_min(1e-3)               # [B]
    diff = pos.unsqueeze(1) - pos.unsqueeze(2)                 # [B,L,L,3]
    d = torch.sqrt(diff.pow(2).sum(-1) + 1e-6) / radius.view(B, 1, 1)  # [B,L,L]
    W = torch.exp(-d / h.unsqueeze(-1)) + c.unsqueeze(-1)       # [B,L,L]
    W = W * mask.unsqueeze(1).float()                           # 隐伏 j 不贡献
    return W


# ----------------------- 可微 Weiszfeld 解码 (对标 _l1_median_batch) -----------------------
def weiszfeld_decode(pos, nrm, mask, W, iters=40, eps=1e-3, v_init=None, gn_eps=1e-12):
    """可微 Weiszfeld 射影 L1 中位数。返回 pred[B,L,3] (观测点=真值)。

    与 inference._l1_median_batch 逐点等价: 对隐伏点 i, 以 W[i,j]>0 的观测法向为样本,
    迭代收敛到最小化 sum_j w_ij acos|<v, n_j>| 的 v。

    v_init: 可选 [B,L,3] 初值 (如 set_aware 组模态 μ_k)。默认 None 用观测均值初值。
    注意: 组内外选择完全由 W 决定 (组内 W 对异组/隐伏 j 为零), v_init 仅影响起手点,
    不改梯度可学性 —— 组内精修的全部信号来自 h,c 经 W 回传。
    """
    B, L, _ = pos.shape
    obs = mask.bool()
    hid = ~obs

    if v_init is None:
        # 观测均值初始化隐伏点。退化网 (观测点=0) 回退到 batch 全局观测均值, 杜绝 0/0=NaN。
        nobs = mask.sum(1, keepdim=True)                                    # [B,1]
        gmean = (nrm * mask.unsqueeze(-1)).sum((0, 1), keepdim=True) / \
            mask.sum().clamp_min(1e-6)                                      # [1,1,3]
        pernet = (nrm * mask.unsqueeze(-1)).sum(1) / nobs.clamp_min(1e-6)   # [B,3]
        gmean_b = gmean.reshape(1, 3).expand(B, 3)                           # [B,3]
        mean_obs = torch.where(nobs > 0, pernet, gmean_b)
        V = torch.where(hid.unsqueeze(-1), mean_obs.unsqueeze(1).expand(B, L, 3), nrm.clone())
    else:
        # 组内初值 (组模态 μ_k): 隐伏点用其组初值, 观测点用真值。
        V = torch.where(hid.unsqueeze(-1), v_init, nrm.clone())

    for _ in range(iters):
        C = torch.einsum("bik,bjk->bij", V, nrm)               # [B,L,L]
        C = C * mask.unsqueeze(1).float()                      # 仅观测 j
        sin = torch.sqrt(torch.clamp(1.0 - C * C, eps ** 2, None))
        signs = torch.sign(C + 1e-12)                          # [B,L,L]
        coeff = (W / sin) * signs                              # [B,L,L]
        G = torch.einsum("bij,bjk->bik", coeff, nrm)           # [B,L,3]
        Gn = G.norm(dim=-1, keepdim=True)
        V_new = torch.where(Gn > gn_eps, G / Gn.clamp_min(gn_eps), V)
        V = torch.where(hid.unsqueeze(-1), V_new, V)           # 只更新隐伏点

    pred = V
    pred = torch.where(obs.unsqueeze(-1), nrm, pred)           # 观测点锁定真值
    return pred


# ----------------------- 顶层模型 -----------------------
class LearnedL1Model(nn.Module):
    def __init__(self, k_knn=16, hidden=32, iters=40, clamp_h=(0.2, 1.0),
                 clamp_c=(0.02, 0.3), random_init=False):
        super().__init__()
        self.k = k_knn
        self.iters = iters
        self.kernel = KernelNet(hidden=hidden, clamp_h=clamp_h,
                                clamp_c=clamp_c, random_init=random_init)

    def forward(self, pos, nrm, mask, s1, s3, log_len, lith):
        """全张量输入: pos,nrm[B,L,3]; mask[B,L]; s1,s3[B,3];
        log_len[B,L]; lith[B,L] (int). 返回 pred[B,L,3], h[B,L], c[B,L]。"""
        B, L, _ = pos.shape
        idx, dist = knn_graph(pos, self.k)
        idx3 = idx.unsqueeze(-1).expand(B, L, self.k, 3)
        knn_pos = torch.gather(pos.unsqueeze(1).expand(B, L, L, 3), 2, idx3)
        evec, eval_ratio = local_frames(knn_pos, self.k)
        graph = (idx, dist, knn_pos, evec, eval_ratio)
        fnode, _, _, _ = build_graph_features(
            pos, nrm, log_len, lith, mask, s1, s3, self.k, graph=graph)

        # 19 维特征 (fnode 布局来自 geometry.build_graph_features 行 105-117):
        #  0..2  pos_local | 3..5 nrm_local | 7 mask | 8..10 eig | 11 aniso
        #  18..20 nrm_obs_mean | 21 obs_frac | 22 dist_obs_min | 12..14 s1_local | 15..17 s3_local
        obs_consistency = fnode[..., 18:21].norm(dim=-1, keepdim=True)  # [B,L,1]
        feat = torch.cat([
            fnode[..., 0:3],     # pos_local (3)
            fnode[..., 3:6],     # nrm_local (3)
            fnode[..., 7:8],     # mask (1)
            fnode[..., 8:11],    # eig_ratio (3)
            fnode[..., 11:12],   # aniso (1)
            obs_consistency,     # obs_consistency (1)
            fnode[..., 21:22],   # obs_frac (1)
            fnode[..., 22:23],   # dist_obs_min (1)
            fnode[..., 12:15],   # s1_local (3)
            fnode[..., 15:18],   # s3_local (3)
        ], dim=-1)              # [B, L, 19]

        h, c = self.kernel(feat)
        W = compute_weights(pos, h, c, mask)
        pred = weiszfeld_decode(pos, nrm, mask, W, iters=self.iters)
        return pred, h, c

    def loss(self, pos, nrm, mask, s1, s3, log_len, lith, nrm_full):
        pred, h, c = self.forward(pos, nrm, mask, s1, s3, log_len, lith)
        cos = (pred * nrm_full).sum(-1).abs().clamp(-1, 1)
        err = torch.rad2deg(torch.acos(cos))                    # [B,L]
        hid = (1.0 - mask) > 0.5
        if hid.any():
            total = err[hid].mean()
        else:
            total = err.mean() * 0.0
        return total, {"l_hid": float(err[hid].mean()) if hid.any() else 0.0,
                       "h_mean": float(h.mean()), "c_mean": float(c.mean())}


# ----------------------- 推理接口 (与 inference.*_dirs 对齐) -----------------------
def learned_l1_dirs(pos, nrm_raw, obs_mask, model, device="cuda",
                    s1=None, s3=None, log_len=None, lith=None):
    """单网络推理。pos/nrm_raw[L,3] numpy, obs_mask[L] bool/int, s1,s3[3],
    log_len/lith[L]。返回 (dirs[L,3] float32, labels[L] int=-1)。"""
    pos = np.asarray(pos, dtype=np.float32)
    nrm_raw = np.asarray(nrm_raw, dtype=np.float32)
    mask = np.asarray(obs_mask, dtype=bool).astype(np.float32)
    L = pos.shape[0]

    pos_t = torch.as_tensor(pos).unsqueeze(0).to(device)
    # 隐伏点法向置零 (与训练一致)
    nrm_t = torch.as_tensor(nrm_raw * mask[:, None]).unsqueeze(0).to(device)
    mask_t = torch.as_tensor(mask).unsqueeze(0).to(device)
    s1_t = torch.as_tensor(np.asarray(s1, dtype=np.float32)).unsqueeze(0).to(device)    # [1,3]
    s3_t = torch.as_tensor(np.asarray(s3, dtype=np.float32)).unsqueeze(0).to(device)    # [1,3]
    log_len_t = torch.as_tensor(np.asarray(log_len, dtype=np.float32)).unsqueeze(0).to(device)  # [1,L]
    lith_t = torch.as_tensor(np.asarray(lith, dtype=np.int64)).unsqueeze(0).to(device)          # [1,L]

    model.eval()
    with torch.no_grad():
        pred, _, _ = model(pos_t, nrm_t, mask_t, s1_t, s3_t, log_len_t, lith_t)
    dirs = pred[0].cpu().numpy().astype(np.float32)
    labels = np.full(L, -1, dtype=int)
    return dirs, labels
