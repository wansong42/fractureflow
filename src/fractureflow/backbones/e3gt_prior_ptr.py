# -*- coding: utf-8 -*-
"""E3GTPriorPtr 骨干 (注册 "e3gt_prior_ptr")。

在 e3gt_prior 的"几何先验残差基底"之上, 追加两套直接攻击**指派瓶颈**的机制。
指派瓶颈是逼近 16° / 13.76° 解码下界的最大障碍 (AGENTS.md: 错指派上界 ~42°,
真值指派+观测中心 ~23°)。仅靠方向残差 (e3gt_prior) 只能"微调角度", 无法"换组",
故不可达 16°。

  (A) 指针 (pointer): 把几何先验产出的每点候选裂隙组方向
      (各 K 的 primary/secondary 组方向 + 最近观测方向, 共 C=2*len(Ks)+1 个)
      **再加上 k 个最近观测法向 (原始观测, 非组均值)** 作为候选集,
      网络学一组 tanh 门控 w_i, 把 pred 从 gp 推向选中的候选:
          pred = normalize( gp + Σ_i tanh(w_i)·(cand_i − gp) + evec·c_local )
      指针末层零初始化 => tanh(0)=0 => 起手 pred == gp == 34.68° 地板
      (不假突破)。这是 fracgen/v25 "pointer+top2" 思路的纯等变实现: 网络可
      "换组", 不被几何先验的固定 top2 混合锁死 —— 指派瓶颈的唯一结构性解法。
      加入原始观测候选后, 指针还能把隐伏点拉向其"最佳匹配观测法向", 这是逼近
      ~9-14° 信息下界 (见 results/ceiling_probe2.json) 的关键动作。

  (B) 组隶属 (membership) 头: 合成数据带 set_ids, 直接用交叉熵监督逐点组隶属
      logits (out["memb"]), 让表征显式学到"每个点属于哪个裂隙组", 与 dir 协同
      把每个点拉向正确全局原型方向。

等变性: gp / cand 协变旋转; tanh(w) 为不变量标量门控; evec·c_local 协变; 故 pred 协变。
"""

import torch
import torch.nn as nn

from ..core import BACKBONES, unit
from ..geometry import knn_graph, local_frames, build_graph_features
from ..geo_prior import (geo_prior_dirs_gpu, geo_prior_candidates_gpu,
                         geo_prior_obsneighbor_candidates_gpu)
from .e3gt_prior import E3GTPrior


@BACKBONES.register("e3gt_prior_ptr")
class E3GTPriorPtr(E3GTPrior):
    def __init__(self, d_model=256, k_knn=16, n_layers=4, dropout=0.1,
                 obs_drop=0.25, k_obs=8, n_groups=6, Ks=(3, 4, 5, 6)):
        super().__init__(d_model=d_model, k_knn=k_knn, n_layers=n_layers,
                         dropout=dropout, obs_drop=obs_drop)
        self.k_obs = k_obs
        self.Ks = Ks
        self.n_geom = 2 * len(Ks) + 1           # 几何候选数 (primary+secondary+ref)
        self.n_cand = self.n_geom + k_obs       # 总候选数

        # (A) 指针: (s,|v|,v) -> n_cand 个 tanh 门控 (零初始化 => 起手全 0 => 地板)
        self.ptr = nn.Sequential(
            nn.Linear(d_model + 1 + 3, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, self.n_cand))
        nn.init.zeros_(self.ptr[-1].weight)
        nn.init.zeros_(self.ptr[-1].bias)

        # (B) 组隶属头: (s,v) -> n_groups logits (仅合成数据有 set_ids 时监督)
        self.memb_head = nn.Sequential(
            nn.Linear(d_model + 3, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, n_groups))

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

        # ---- 几何先验 + 候选集 (均必须 detach, 见 e3gt_prior 说明) ----
        gp = geo_prior_dirs_gpu(pos, nrm, mask, fast=self.prior_fast).detach()  # [B,L,3]
        gp_local = torch.einsum("blcd,blc->bld", evec, gp).detach()
        cand_g = geo_prior_candidates_gpu(pos, nrm, mask,
                                         fast=self.prior_fast, Ks=self.Ks).detach()  # [B,L,9,3]
        cand_o = geo_prior_obsneighbor_candidates_gpu(
            pos, nrm, mask, k=self.k_obs, gp=gp,
            fast=self.prior_fast).detach()                                  # [B,L,k_obs,3]
        cand = torch.cat([cand_g, cand_o], dim=2)                          # [B,L,n_cand,3]
        assert cand.shape[2] == self.n_cand, \
            f"candidate count {cand.shape[2]} != n_cand {self.n_cand}"

        # 向量通道起手: 观测点真实观测法向, 隐伏点用几何先验方向 (同 e3gt_prior)
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

        # ---- 残差校正 + 指针 ----
        nv = v.norm(dim=-1, keepdim=True)                          # [B,L,1]
        c_local = self.corr(torch.cat([s, nv, v], -1))            # [B,L,3] 零初始化

        # (A) 指针门控: 把 pred 从 gp 推向选中的候选 (起手 gates=0 => pred==gp)
        ptr_logits = self.ptr(torch.cat([s, nv, v], -1))           # [B,L,n_cand]
        gates = torch.tanh(ptr_logits)                             # [B,L,n_cand] ∈[-1,1]
        cand_off = cand - gp.unsqueeze(-2)                         # [B,L,n_cand,3]
        shift = (gates.unsqueeze(-1) * cand_off).sum(-2)           # [B,L,3]
        pred = unit(gp + shift + torch.einsum("blcd,bld->blc", evec, c_local))

        aniso = (eval_ratio[:, :, 0] - eval_ratio[:, :, 2]).clamp(0.0, 1.0)  # [B,L]
        set_dir_pred = unit(self.set_dir_head(torch.cat([s, v], -1)))      # [B,L,3]
        memb = self.memb_head(torch.cat([s, v], -1))                        # [B,L,n_groups]

        return {"pred": pred, "v": v, "s": s, "aniso": aniso,
                "set_dir_pred": set_dir_pred, "memb": memb}
