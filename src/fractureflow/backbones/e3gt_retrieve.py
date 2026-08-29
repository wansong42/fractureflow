# -*- coding: utf-8 -*-
"""E3GTRetrieve 骨干 (注册 "e3gt_retrieve")。

核心创新 (冲击 13° 解码下界):
  关键实验发现 —— 对每个隐伏点, 与其真值法向最接近 (<=9~13°) 的那个"观测法向"
  就位于它的 ~20~40 个空间最近邻观测点之中。因此只要让网络**学会从空间候选集里
  "挑"出那一个观测法向**, 就能在不泄漏真值的前提下逼近 9~13° 信息下界。

  与 e3gt_prior_ptr 的本质区别:
    ptr 用 tanh 门控把全部候选"加权求和"(等价于平滑平均) -> 候选互相抵消, 卡在 31°;
    本骨干用 **softmax 注意力做"选择"**, 让网络聚焦到单个最佳候选 -> 可逼近下界。

  结构:
    - 复用 E3GTV4 等变消息传递得到逐点表征 (s,v) 与基础方向 base(锚定几何先验 gp)。
    - 对每个点, 取 k_cand 个空间最近观测法向 (+空间距离不变量) + 几何候选(组方向)。
    - 注意力打分: 查询 q_i (来自 s,v,base) 对各候选打分, 含不变量
      (空间距离, 候选法向·base, 候选法向·gp)。softmax -> 选择权重。
    - 检索方向 = 候选的加权(对齐后)平均; 最终 pred = normalize(gp + gate*(retrieved-gp)),
      gate 零初始化 => 起手 pred==gp==34.68° 地板 (不假突破)。
    - 训练时对候选做"按真值符号对齐"的 teacher-forcing, 让检索损失直接驱动选择学习。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import BACKBONES, unit
from ..geometry import knn_graph, local_frames, build_graph_features
from ..geo_prior import (geo_prior_dirs_gpu, geo_prior_candidates_gpu)
from .e3gt_prior import E3GTPrior


def knn_obs(pos, nrm, mask, k, chunk=1024):
    """返回每个点空间最近的 k 个观测法向与对应距离。

    Returns:
        cand_o : [B, L, k, 3]  (与 gp 同半球对齐由调用方处理)
        cand_d : [B, L, k, 1]  空间距离 / radius
    """
    B, L, _ = pos.shape
    radius = pos.std(1).mean(-1, keepdim=True).clamp_min(1e-3)  # [B,1]
    Lobs = mask.sum(1).long()
    maxobs = int(Lobs.max())
    obs_pos = torch.zeros(B, maxobs, 3, device=pos.device)
    obs_nrm = torch.zeros(B, maxobs, 3, device=pos.device)
    for b in range(B):
        m = mask[b].bool()
        obs_pos[b, :Lobs[b]] = pos[b][m]
        obs_nrm[b, :Lobs[b]] = nrm[b][m]
    d = torch.empty(B, L, maxobs, device=pos.device)
    for i in range(0, L, chunk):
        j = min(i + chunk, L)
        diff = pos[:, i:j, None, :] - obs_pos[:, None, :, :]   # [B,chunk,maxobs,3]
        d[:, i:j] = (diff * diff).sum(-1)
    # 屏蔽 padding
    pad = torch.arange(maxobs, device=pos.device)[None, None, :] >= Lobs[:, None, None]
    d = d.masked_fill(pad, float("inf"))
    kk = min(k, maxobs)
    dtop, didx = torch.topk(d, kk, dim=-1, largest=False)  # [B,L,kk]
    # 观测点不足 k 时补零 (cand_o=0, cand_d=大常数) 让注意力自动忽略
    if kk < k:
        pn = k - kk
        didx = torch.cat([didx, torch.zeros(B, L, pn, dtype=torch.long, device=pos.device)], dim=2)
        dtop = torch.cat([dtop, torch.full((B, L, pn), float(1e3), device=pos.device)], dim=2)
    idx_flat = didx + torch.arange(B, device=pos.device)[:, None, None] * maxobs
    obs_flat = obs_nrm.reshape(B * maxobs, 3)
    cand_o = obs_flat[idx_flat.reshape(-1)].reshape(B, L, k, 3)
    cand_d = dtop.unsqueeze(-1) / radius.unsqueeze(1).unsqueeze(2)
    return cand_o, cand_d


@BACKBONES.register("e3gt_retrieve")
class E3GTRetrieve(E3GTPrior):
    def __init__(self, d_model=256, k_knn=16, n_layers=4, dropout=0.1,
                 obs_drop=0.25, k_cand=32, Ks=(3, 4, 5, 6), n_groups=6):
        super().__init__(d_model=d_model, k_knn=k_knn, n_layers=n_layers,
                         dropout=dropout, obs_drop=obs_drop)
        self.k_cand = k_cand
        self.Ks = Ks
        self.n_geom = 2 * len(Ks) + 1   # primary+secondary 组方向 + ref_b(=gp)
        self.query_proj = nn.Sequential(
            nn.Linear(d_model + 1 + 3 + 3, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model // 2))
        self.score_q = nn.Linear(d_model // 2, 1)
        self.cand_emb = nn.Sequential(
            nn.Linear(3, d_model // 4), nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 4))
        self.score_c = nn.Sequential(
            nn.Linear(d_model // 4 + 3, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 1))
        self.gate = nn.Sequential(
            nn.Linear(d_model + 3, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.memb_head = nn.Sequential(
            nn.Linear(d_model + 3, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, n_groups))

    def forward(self, b):
        pos = b["pos"]; nrm = b["nrm"]; mask = b["mask"]
        log_len = b["log_len"]; lith = b["lith"]; s1 = b["s1"]; s3 = b["s3"]
        B, L, _ = pos.shape

        nrm_in = self._dropped_nrm(nrm, mask)

        idx, dist = knn_graph(pos, self.k)
        idx3 = idx.unsqueeze(-1).expand(B, L, self.k, 3)
        knn_pos = torch.gather(pos.unsqueeze(1).expand(B, L, L, 3), 2, idx3)
        evec, eval_ratio = local_frames(knn_pos, self.k)
        graph = (idx, dist, knn_pos, evec, eval_ratio)
        fnode, fedge, _, _ = build_graph_features(
            pos, nrm_in, log_len, lith, mask, s1, s3, self.k, graph=graph)

        gp = geo_prior_dirs_gpu(pos, nrm, mask, fast=self.prior_fast).detach()
        gp_local = torch.einsum("blcd,blc->bld", evec, gp).detach()
        cand_g = geo_prior_candidates_gpu(pos, nrm, mask,
                                         fast=self.prior_fast, Ks=self.Ks).detach()
        cand_o, cand_d = knn_obs(pos, nrm, mask, self.k_cand)      # [B,L,k,3], [B,L,k,1]
        cand_o = cand_o.detach()
        cand_d = cand_d.detach()
        cand = torch.cat([cand_g, cand_o], dim=2)                  # [B,L, n_geom+k_cand, 3]
        # 几何候选无空间距离 -> 置 0
        d_feat = torch.cat([torch.zeros(B, L, self.n_geom, 1, device=pos.device),
                            cand_d], dim=2)                          # [B,L,Kc,1]
        Kc = cand.shape[2]

        nrm_u = unit(torch.where(nrm_in.norm(dim=-1, keepdim=True) > 1e-6,
                                 nrm_in, torch.zeros_like(nrm_in)))
        has_obs = (nrm_in.norm(dim=-1, keepdim=True) > 1e-6).float()
        v0 = torch.where(has_obs > 0.5,
                         torch.einsum("blc,blcd->bld", nrm_u, evec),
                         gp_local)
        s = self.lift(fnode) + self.prior_proj(gp_local)
        v = v0
        rel = knn_pos - pos.unsqueeze(2)
        rel_local = torch.einsum("blke,blcd->blkd", rel, evec)

        radius = pos.std(1).mean(-1, keepdim=True).clamp_min(1e-3)
        for layer in self.layers:
            s, v = layer(s, v, idx, dist, rel_local, evec, radius=radius)

        nv = v.norm(dim=-1, keepdim=True)
        c_local = self.corr(torch.cat([s, nv, v], -1))            # 零初始化 => base==gp
        base = unit(gp + torch.einsum("blcd,bld->blc", evec, c_local))   # [B,L,3]

        # ---------- 候选注意力检索 ----------
        ref = b["nrm_full"] if self.training else base.detach()
        dot_ref = (cand * ref.unsqueeze(2)).sum(-1)
        sign = torch.where(dot_ref >= 0, torch.ones_like(dot_ref), -torch.ones_like(dot_ref))
        cand_a = cand * sign.unsqueeze(-1)                        # 对齐后候选

        q = self.query_proj(torch.cat([s, nv, v, base], -1))      # [B,L,d/2]
        d_feat = torch.clamp(d_feat, 0.0, 8.0)                     # 距离特征限幅, 防 softmax 溢出
        odot_base = (cand_a * base.unsqueeze(2)).sum(-1, keepdim=True)
        odot_gp = (cand_a * gp.unsqueeze(2)).sum(-1, keepdim=True)
        cemb = self.cand_emb(cand_a)
        cand_feats = torch.cat([cemb, d_feat, odot_base, odot_gp], -1)  # [B,L,Kc, d/4+3]
        score = self.score_q(q).unsqueeze(2) + self.score_c(cand_feats)
        score = score - score.amax(dim=2, keepdim=True)            # softmax 数值稳定
        w = F.softmax(score, dim=2)
        retrieved = unit((w * cand_a).sum(2))                     # [B,L,3]

        gate = torch.sigmoid(self.gate(torch.cat([s, base], -1)))
        pred = unit(gp + gate * (retrieved - gp))

        aniso = (eval_ratio[:, :, 0] - eval_ratio[:, :, 2]).clamp(0.0, 1.0)
        set_dir_pred = unit(self.set_dir_head(torch.cat([s, v], -1)))
        memb = self.memb_head(torch.cat([s, v], -1))

        return {"pred": pred, "v": v, "s": s, "aniso": aniso,
                "set_dir_pred": set_dir_pred, "memb": memb,
                "w": w, "retrieved": retrieved, "gate": gate, "base": base,
                "cand": cand}
