# -*- coding: utf-8 -*-
"""plain_mlp 对照骨干 (注册为 "plain_mlp")。

目的: 隔离"等变消息传播"这一变量的贡献。
- 与 e3gt_v4 共用同一套 kNN-PCA 局部标架 + 不变节点特征 (geometry.build_graph_features)。
- 但**去掉跨点等变向量消息传递**: 用逐点 MLP (带残差) 在局部标架内回归方向, 再旋回全局。
- 不含 set_dir 辅助头 => set_dir 损失对其自动跳过。

注意: 它仍"碰巧"等变 (因在局部标架内运算并旋回), 但它测的是
"没有迭代式等变向量消息传递/NormAct/叉积消息"时还能走多远。
若 plain_mlp 与 e3gt_v4 差距很小 => 等变消息传播没在干活, 需换思路;
若差距大 => 等变消息传播确有价值, 保留 e3gt_v4 路线。
"""

import torch
import torch.nn as nn

from ..core import BACKBONES, unit
from ..geometry import knn_graph, local_frames, build_graph_features


@BACKBONES.register("plain_mlp")
class PlainBackbone(nn.Module):
    def __init__(self, d_model=256, k_knn=16, n_layers=4, dropout=0.1, obs_drop=0.25):
        super().__init__()
        self.k = k_knn
        self.d = d_model
        self.obs_drop = obs_drop

        # 31 维不变节点特征 + 3 维观测法向(局部标架) 协变通道
        self.lift = nn.Sequential(
            nn.Linear(31 + 3, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model), nn.Dropout(dropout))
            for _ in range(n_layers)])

        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 3))

    def _dropped_nrm(self, nrm, mask):
        if self.training and self.obs_drop > 0:
            drop = (torch.rand_like(mask) < self.obs_drop) & (mask > 0.5)
            return torch.where(drop.unsqueeze(-1), torch.zeros_like(nrm), nrm)
        return nrm

    def forward(self, b):
        pos = b["pos"]; nrm = b["nrm"]; mask = b["mask"]
        log_len = b["log_len"]; lith = b["lith"]; s1 = b["s1"]; s3 = b["s3"]
        B, L, _ = pos.shape

        nrm_in = self._dropped_nrm(nrm, mask)
        idx, dist = knn_graph(pos, self.k)
        idx3 = idx.unsqueeze(-1).expand(B, L, self.k, 3)
        knn_pos = torch.gather(pos.unsqueeze(1).expand(B, L, L, 3), 2, idx3)
        evec, eval_ratio = local_frames(knn_pos, self.k)            # [B,L,3,3] [B,L,3]
        graph = (idx, dist, knn_pos, evec, eval_ratio)

        fnode, _, _, _ = build_graph_features(
            pos, nrm_in, log_len, lith, mask, s1, s3, self.k,
            graph=graph)                                           # [B,L,31]

        # 观测法向旋入局部标架 (协变通道), 隐伏点=0
        nrm_u = unit(torch.where(nrm_in.norm(dim=-1, keepdim=True) > 1e-6,
                                 nrm_in, torch.zeros_like(nrm_in)))
        has_obs = (nrm_in.norm(dim=-1, keepdim=True) > 1e-6).float()
        v0 = torch.einsum("blc,blcd->bld", nrm_u, evec) * has_obs  # [B,L,3]

        x = self.lift(torch.cat([fnode, v0], -1))
        for layer in self.layers:
            x = layer(x) + x                                        # 残差
        n_local = self.readout(x)                                  # [B,L,3]
        n_global = torch.einsum("blcd,bld->blc", evec, n_local)     # 旋回全局
        pred = unit(n_global)

        aniso = (eval_ratio[:, :, 0] - eval_ratio[:, :, 2]).clamp(0.0, 1.0)
        return {"pred": pred, "aniso": aniso}
