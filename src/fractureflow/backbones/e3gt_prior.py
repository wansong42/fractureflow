# -*- coding: utf-8 -*-
"""E3GTPrior 骨干 (注册为 "e3gt_prior")。

核心改动 (实打实改模型, 非换口径):
  - 在 forward 内用当前 (pos,nrm,mask) 现场算几何先验 gp = top2_ens (31.1° 地板)。
  - 隐伏点向量通道 v0 不再从 0 起步, 而是从 gp 的局部标架投影起步;
    并把 gp 局部投影作为额外不变量注入标量通道 s (让网络显式"看见"强先验)。
  - 输出改为残差校正: pred = normalize(gp_global + evec @ corr_local)。
    校正读头末层零初始化 => 起手 pred == gp == 31.1° 地板, 训练只能往下压。

等变性: 旋转输入 => gp 协变旋转, corr_local 为不变量, evec -> R·evec,
故 pred 协变; 配合 |cos| 无向度量, 评测对旋转一致。
"""

import torch
import torch.nn as nn

from ..core import BACKBONES
from ..e3gt_v4 import E3GTV4, unit
from ..geometry import knn_graph, local_frames, build_graph_features
from ..geo_prior import geo_prior_dirs_gpu


@BACKBONES.register("e3gt_prior")
class E3GTPrior(E3GTV4):
    def __init__(self, d_model=256, k_knn=16, n_layers=4, dropout=0.1,
                 obs_drop=0.25):
        super().__init__(d_model=d_model, k_knn=k_knn, n_layers=n_layers,
                         dropout=dropout, obs_drop=obs_drop)
        # 训练期用廉价几何先验 (fast=True) 作残差基底以提速; 评测期必须精确 (fast=False)
        # 以保证 "起手即 34.68° 地板"。trainer 会在训练步置 True、评测前置 False。
        self.prior_fast = False
        # 把几何先验(局部标架投影)注入标量通道
        self.prior_proj = nn.Sequential(
            nn.Linear(3, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        # 残差校正读头: 在局部标架内回归修正向量; 末层零初始化 => 起手即几何先验
        self.corr = nn.Sequential(
            nn.Linear(d_model + 1 + 3, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 3))
        nn.init.zeros_(self.corr[-1].weight)
        nn.init.zeros_(self.corr[-1].bias)

    def forward(self, b):
        pos = b["pos"]; nrm = b["nrm"]; mask = b["mask"]
        log_len = b["log_len"]; lith = b["lith"]; s1 = b["s1"]; s3 = b["s3"]
        B, L, _ = pos.shape

        nrm_in = self._dropped_nrm(nrm, mask)

        # 图 + 局部标架 (一次性计算)
        idx, dist = knn_graph(pos, self.k)
        idx3 = idx.unsqueeze(-1).expand(B, L, self.k, 3)
        knn_pos = torch.gather(pos.unsqueeze(1).expand(B, L, L, 3), 2, idx3)
        evec, eval_ratio = local_frames(knn_pos, self.k)            # [B,L,3,3]
        graph = (idx, dist, knn_pos, evec, eval_ratio)
        fnode, fedge, _, _ = build_graph_features(
            pos, nrm_in, log_len, lith, mask, s1, s3, self.k,
            graph=graph)                                           # [B,L,31]
        radius = pos.std(1).mean(-1, keepdim=True).clamp_min(1e-3)  # [B,1]

        # ---- 几何先验 (GPU, 与 numpy top2_ens 一致) ----
        # 注意: gp 必须 detach。它是"强归纳偏置/引导特征", 不是可学组件;
        # 若不 detach, 反向会穿过 kmeans/argsort/gather 的大图, batch32 下单步 30~60s
        # 训练不可行。detach 后仍满足: 校正读头零初始化 => 起手 pred==gp==地板;
        # 残差 c_local 照常可学 (梯度只走 corr 读头与 E3GT 各层)。
        gp = geo_prior_dirs_gpu(pos, nrm, mask, fast=self.prior_fast).detach()  # [B,L,3] 全局 (常量)
        gp_local = torch.einsum("blcd,blc->bld", evec, gp).detach()  # 局部标架投影 (常量)

        # 向量通道起手: 观测点用真实观测法向, 隐伏点用几何先验方向
        nrm_u = unit(torch.where(nrm_in.norm(dim=-1, keepdim=True) > 1e-6,
                                 nrm_in, torch.zeros_like(nrm_in)))
        has_obs = (nrm_in.norm(dim=-1, keepdim=True) > 1e-6).float()
        v0 = torch.where(has_obs > 0.5,
                         torch.einsum("blc,blcd->bld", nrm_u, evec),
                         gp_local)

        s = self.lift(fnode) + self.prior_proj(gp_local)           # [B,L,d] 注入先验
        v = v0                                                     # [B,L,3]

        rel = knn_pos - pos.unsqueeze(2)                           # [B,L,k,3]
        rel_local = torch.einsum("blke,blcd->blkd", rel, evec)     # [B,L,k,3]

        for layer in self.layers:
            s, v = layer(s, v, idx, dist, rel_local, evec, radius)

        # ---- 残差校正: pred = normalize(gp_global + evec @ corr_local) ----
        nv = v.norm(dim=-1, keepdim=True)                          # [B,L,1]
        c_local = self.corr(torch.cat([s, nv, v], -1))             # [B,L,3]
        pred = unit(gp + torch.einsum("blcd,bld->blc", evec, c_local))

        aniso = (eval_ratio[:, :, 0] - eval_ratio[:, :, 2]).clamp(0.0, 1.0)  # [B,L]
        set_dir_pred = unit(self.set_dir_head(torch.cat([s, v], -1)))         # [B,L,3]

        return {"pred": pred, "v": v, "s": s, "aniso": aniso, "set_dir_pred": set_dir_pred}
