# -*- coding: utf-8 -*-
"""重构核心: 裂隙组指派模型 (SetProp)。

设计动机 (见 results/v25_gran_dump.json):
  decode_ceiling_true_assign_k6 = 13.76°  -- 这是"真值指派 + 观测组中心"的下限。
  当前 SOTA (v25) = 31.10°, 差距 17° 完全是"指派"质量造成的:
  一旦能像 oracle 一样把每个隐伏点指到正确裂隙组, 用该组观测均值方向预测即得 13.76°。

本模型把问题显式化为"指派":
  1. 几何法从"观测法向"发现 K=6 裂隙组中心 mu_k (可微球面 k-means, 与天花板协议一致)。
  2. 逐点交叉注意力头 a_i = softmax(<h_i, key(mu_k)>) 学习每个点的组隶属度。
     - 观测点: 用几何标签做强监督 CE (只用观测, 无泄漏)。
     - 合成数据隐伏点: 用真值组做直接 CE 监督。
     - 真实数据隐伏点: 用方向 NLL 路由 (迫使 a_i 指向"均值方向对齐真值"的组)。
  3. 解码 d_i = unit(Σ_k a_ik mu_k)  -- 完美指派的极限即 13.76° 天花板。
  4. (可选) 组内残差 res_i = MLP(h_i), d_i = unit(Σ a_ik mu_k + res_i) 以突破组均值下限。

全程只用到观测法向/位置/应力, 隐伏标签仅用于训练损失, 评测严格不泄漏。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import build_graph_features
from .field_encoder import FieldEncoder


def unit_norm(x, eps=1e-9):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def geometric_sets(nrm, mask, K, iters=6, temp=20.0):
    """可微球面 k-means: 仅用观测法向 (mask) 发现 K 个组中心 mu [B,K,3]。
    与 v25 天花板协议 (K=6) 一致。"""
    B, L, _ = nrm.shape
    n_u = unit_norm(nrm) * mask.unsqueeze(-1)
    with torch.no_grad():
        mu = torch.randn(B, K, 3, device=nrm.device)
        mu = unit_norm(mu)
        for b in range(B):
            obs_idx = mask[b].nonzero(as_tuple=False).flatten()
            if len(obs_idx) >= 1:
                sel = obs_idx[torch.randperm(len(obs_idx))[:min(K, len(obs_idx))]]
                mu[b, :len(sel)] = n_u[b, sel]
    for _ in range(iters):
        cos = torch.einsum("ble,bke->blk", n_u, mu)          # [B,L,K]
        w = F.softmax(temp * cos, -1) * mask.unsqueeze(-1)   # [B,L,K]
        num = torch.einsum("blk,ble->bke", w, nrm)
        mu = unit_norm(num + 1e-9)
    return mu


def nearest_obs_dir(nrm, mask, pos):
    """每点取最近观测点的法向方向 [B,L,3] (隐伏点=最近观测法向, 观测点=自身)。"""
    B, L, _ = nrm.shape
    d2 = ((pos[:, :, None, :] - pos[:, None, :, :]) ** 2).sum(-1)   # [B,L,L]
    BIG = 1e9
    d2 = d2 + (1.0 - mask[:, None, :]) * BIG
    nn_idx = d2.argmin(-1)                                     # [B,L]
    return torch.gather(nrm, 1, nn_idx.unsqueeze(-1).expand(B, L, 3))


class SetProp(nn.Module):
    def __init__(self, K=6, d_model=256, k_knn=16, gnn_layers=3, dropout=0.15,
                 n_proto_init=6, use_residual=True, use_nn_anchor=True):
        super().__init__()
        self.K = K
        self.k_knn = k_knn
        self.d_model = d_model
        self.use_residual = use_residual
        self.use_nn_anchor = use_nn_anchor
        self.encoder = FieldEncoder(
            d_in=31, d_edge=4, k=k_knn, d_model=d_model,
            n_layers=gnn_layers, dropout=dropout, use_attention=True)

        # 组方向 -> key, 与点特征交叉注意力得到隶属度
        self.key_proj = nn.Linear(3, d_model)
        self.attn_scale = float(d_model) ** -0.5
        self.assign_mlp = nn.Sequential(
            nn.Linear(d_model + 3, d_model), nn.LayerNorm(d_model), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1))

        if use_residual:
            self.resid_mlp = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 3))

    def sets(self, nrm, mask):
        return geometric_sets(nrm, mask, self.K)

    def forward(self, b):
        pos = b["pos"]; nrm = b["nrm"]; mask = b["mask"]
        log_len = b["log_len"]; lith = b["lith"]; s1 = b["s1"]; s3 = b["s3"]
        B, L, _ = nrm.shape

        mu = self.sets(nrm, mask)                          # [B,K,3] 组中心
        nn = nearest_obs_dir(nrm, mask, pos)               # [B,L,3] 最近观测方向
        # 锚点 = K 个组中心 (+ 可选最近观测方向作为局部松弛, 类比 v25 候选)
        if self.use_nn_anchor:
            anchors = torch.cat([mu.unsqueeze(1).expand(B, L, self.K, 3),
                                 nn.unsqueeze(2)], dim=2)  # [B,L,K+1,3]
        else:
            anchors = mu.unsqueeze(1).expand(B, L, self.K, 3)  # [B,L,K,3] 纯组指派
        self._M = anchors.shape[2]

        fnode, fedge, idx, _ = build_graph_features(
            pos, nrm, log_len, lith, mask, s1, s3, self.k_knn)
        h = self.encoder(fnode, fedge, idx)                # [B,L,d]

        # 交叉注意力: 每个点对 (K+1) 个锚点的打分
        h_exp = h.unsqueeze(2).expand(B, L, self._M, self.d_model)   # [B,L,M,d]
        anc_exp = anchors                                          # [B,L,M,3]
        score = self.assign_mlp(torch.cat([h_exp, anc_exp], -1)).squeeze(-1)  # [B,L,M]
        log_a = score * self.attn_scale
        a = F.softmax(log_a, -1)                           # [B,L,M] 软隶属度

        # 解码: 锚点混合方向
        mix = torch.einsum("blm,blme->ble", a, anchors)   # [B,L,3]
        if self.use_residual:
            res = self.resid_mlp(h)
            mix = mix + res
        pred = unit_norm(mix)                              # [B,L,3]

        out = {"pred": pred, "assign": a, "mu": mu, "nn": nn, "h": h}
        return out

    # ---------------- 损失 ----------------
    def geom_label(self, nrm, mask, mu):
        """观测点几何组标签 (只用观测法向, 无泄漏) -> [B,L] long (0..K-1)"""
        gcos = torch.einsum("ble,bke->blk", unit_norm(nrm) * mask.unsqueeze(-1), mu)
        return gcos.abs().argmax(-1)

    def loss(self, b, w_obs_ce=1.0, w_dir=1.0, w_setg=1.0, w_res=0.1, w_ent=0.05,
             w_oracle=0.0):
        out = self.forward(b)
        pred = out["pred"]; a = out["assign"]; mu = out["mu"]
        mask = b["mask"]; nrm_full = b["nrm_full"]
        B, L, _ = pred.shape
        K = self.K
        a_set = a[..., :K]                                # 仅对 K 个组中心算组监督
        hid = (1.0 - mask) > 0.5
        obs = mask > 0.5

        total = 0.0
        comp = {}

        # (1) 观测点几何组 CE (强监督, 无泄漏)
        if obs.any():
            gl = self.geom_label(b["nrm"], mask, mu)
            log_a = torch.log(a_set + 1e-8)
            l_obs = -(log_a[obs] * F.one_hot(gl[obs], K).float()).sum(-1).mean()
            total = total + w_obs_ce * l_obs
            comp["l_obs_ce"] = float(l_obs)

        # (2) 方向损失 (隐伏点): 1 - |cos(pred, true)|, 对称
        if hid.any():
            cos = (pred * nrm_full).sum(-1).abs().clamp(0, 1)   # [B,L]
            l_dir = (1.0 - cos[hid]).mean()
            total = total + w_dir * l_dir
            comp["l_dir"] = float(l_dir)

        # (3) 合成真值组 CE (隐伏点直接监督, 仅 synth 有 set_ids)
        if "set_ids" in b and "set_dirs" in b and hid.any():
            set_ids = b["set_ids"]; set_dirs = b["set_dirs"]
            G = set_dirs.shape[1]
            cos_gk = torch.einsum("bge,bke->bgk", set_dirs, mu).abs()  # [B,G,K]
            g2k = cos_gk.argmax(-1)                                   # [B,G]
            sid = set_ids.clamp(min=0, max=G - 1)
            tgt = g2k.gather(1, sid)                                  # [B,L]
            valid = set_ids >= 0
            log_a = torch.log(a_set + 1e-8)
            l_setg = -(log_a[hid & valid] *
                       F.one_hot(tgt[hid & valid], K).float()).sum(-1).mean()
            total = total + w_setg * l_setg
            comp["l_setg"] = float(l_setg)

        # (4) 自监督稀疏化: 降低隐伏点指派熵, 迫使每个隐伏点"认领"一个组
        #     (真实数据隐伏点无组标签, 仅靠方向损失+熵正则逼近 oracle 提交式指派)
        if w_ent > 0 and hid.any():
            p = a[hid]                                    # [N, M]
            ent = -(p * torch.log(p + 1e-8)).sum(-1).mean()
            total = total + w_ent * ent
            comp["l_ent"] = float(ent)

        # (5) 真实隐伏点 oracle 组伪标签: 用真值方向 + 几何组中心推导"应属哪组"
        #     仅当无真值 set_ids 时启用 (即真实数据). 这是逼近 13° 天花板的关键监督:
        #     目标 = argmax_k |cos(真方向, 几何组k均值)|, 与天花板"真指派+观测组中心"一致.
        has_synth_set = ("set_ids" in b and "set_dirs" in b)
        if w_oracle > 0 and hid.any() and not has_synth_set:
            cos_k = torch.einsum("ble,bke->blk",
                                 unit_norm(nrm_full), mu).abs()       # [B,L,K]
            oracle_set = cos_k.argmax(-1)                            # [B,L]
            log_a = torch.log(a_set + 1e-8)
            l_oracle = -(log_a[hid] *
                         F.one_hot(oracle_set[hid], K).float()).sum(-1).mean()
            total = total + w_oracle * l_oracle
            comp["l_oracle"] = float(l_oracle)

        comp["total"] = float(total)
        return total, comp
