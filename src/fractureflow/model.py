# -*- coding: utf-8 -*-
"""FracGen 完整模型: FieldEncoder + SetModule + GenHead。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import build_graph_features
from .field_encoder import FieldEncoder
from .set_module import SetModule
from .gen_head import GenHead

F_NODE_DIM = 31   # 见 geometry.build_graph_features
F_EDGE_DIM = 4


class FracGen(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = FieldEncoder(
            d_in=F_NODE_DIM, d_edge=F_EDGE_DIM, k=cfg.k_knn,
            d_model=cfg.d_model, n_layers=cfg.gnn_layers, dropout=cfg.dropout,
            use_attention=cfg.use_attention)
        self.set_module = SetModule(cfg.n_proto, temp=cfg.temp_assign)
        self.head = GenHead(cfg.d_model, cfg.n_proto, kappa_max=cfg.kappa_max)

    def encode_features(self, b):
        """b: dict 批量 -> (h [B,L,d], alpha [B,L,K], mu_hat [B,K,3],
        usage [B,K], aniso [B,L], idx [B,L,k])"""
        fnode, fedge, idx, knn_pos = build_graph_features(
            b["pos"], b["nrm"], b["log_len"], b["lith"], b["mask"], b["s1"], b["s3"],
            self.cfg.k_knn)
        h = self.encoder(fnode, fedge, idx)
        alpha, mu_hat, usage, w = self.set_module(b["nrm"], b["mask"])
        aniso = fnode[:, :, 11]  # F 布局: .. 18..20 nhb_nrm(全局), 21 obs占比
        nhb_nrm = fnode[:, :, 18:21]   # 近邻观测均值方向 (传播分支直瞄目标)
        cov = fnode[:, :, 21]          # 近邻观测占比
        return h, alpha, mu_hat, usage, aniso, idx, nhb_nrm, cov

    def neighbor_assign(self, alpha, mask, idx):
        """标签传播: 隐伏点的"邻居观测隶属度共识" y_i = mean_j α_j (j∈kNN, 仅观测)。

        这正是"把观测到的裂隙组方向传播到未探测区"的显式监督。
        返回 [B,L,K] 归一化软标签 (无观测邻居时为 0)。
        """
        B, L, K = alpha.shape
        a_obs = alpha * mask.unsqueeze(-1)                     # 观测点 α, 隐伏 0
        aj = torch.gather(a_obs.unsqueeze(1).expand(B, L, L, K), 2,
                          idx.unsqueeze(-1).expand(B, L, self.cfg.k_knn, K))
        y = aj.mean(2)                                        # [B,L,K]
        has_nbr = (aj.sum(-1) > 1e-8).any(2).float()          # [B,L] 有观测邻居
        y = y / (y.sum(-1, keepdim=True) + 1e-8)
        return y, has_nbr

    def forward(self, b):
        h, alpha, mu_hat, usage, aniso, idx, nhb_nrm, cov = self.encode_features(b)
        out = self.head.infer(h, alpha, mu_hat, usage, nhb_nrm)
        out["aniso"] = aniso
        out["h"] = h
        return out

    def loss(self, b):
        """b 含 'nrm_full' (全点真值)。返回 (total, dict(分量))"""
        h, alpha, mu_hat, usage, aniso, idx, nhb_nrm, cov = self.encode_features(b)
        mask = b["mask"]
        target = b["nrm_full"]
        aniso_w = torch.clamp(aniso / 0.6, 0.2, 1.0) * self.cfg.w_aniso
        hid_w = (1.0 - mask) * aniso_w
        obs_w = mask

        l_hid = self.head.nll_loss(target, h, alpha, mu_hat, hid_w, usage)
        l_obs = self.head.nll_loss(target, h, alpha, mu_hat, obs_w, usage)
        l_and = self.set_module.anderson_loss(mu_hat, b["s1"], b["s3"])
        l_ent = -1.0 * (alpha * torch.log(alpha + 1e-8)).sum(-1).mean()  # -H(alpha), 鼓励确定性

        # 组标签传播 KL (隐伏点 π 对齐邻居观测共识)
        pi, _, _, _ = self.head.params(h, alpha, mu_hat, usage)
        y, has_nbr = self.neighbor_assign(alpha, mask, idx)
        kl = -(torch.log(pi + 1e-8) * y).sum(-1)              # KL(pi||y) 交叉熵
        n_hid_nb = (has_nbr * (1.0 - mask)).sum().clamp_min(1.0)
        l_assign = (kl * has_nbr * (1.0 - mask)).sum() / n_hid_nb

        # 显式传播分支: 隐伏点方向 → 近邻观测均值 (结构化copy 监督)
        wp = (1.0 - mask) * cov
        l_prop = self.head.prop_loss(h, nhb_nrm, wp, nhb_nrm, mu_aff=None)
        # 传播分支直接回归掩码真值 (最终输出即隐伏方向)
        l_true = self.head.true_loss(target, h, nhb_nrm, hid_w)

        # 真值组指派监督 (仅合成数据有 set_ids/set_dirs)
        l_setg = torch.zeros((), device=target.device)
        if "set_ids" in b and "set_dirs" in b:
            l_setg = self._set_group_loss(alpha, mu_hat, b, mask)

        loss = l_hid + float(self.cfg.w_obs) * l_obs + float(self.cfg.w_anderson) * l_and \
               + float(self.cfg.w_ent) * l_ent + float(self.cfg.w_assign) * l_assign \
               + float(self.cfg.w_prop) * l_prop + float(self.cfg.w_true) * l_true \
               + float(self.cfg.w_setg) * l_setg
        return loss, {
            "l_hid": l_hid.item(), "l_obs": l_obs.item(),
            "l_and": l_and.item(), "l_ent": l_ent.item(), "l_assign": l_assign.item(),
            "l_prop": l_prop.item(), "l_true": l_true.item(), "l_setg": l_setg.item(),
        }

    def _set_group_loss(self, alpha, mu_hat, b, mask):
        """合成真值组 -> 最近原型匹配 -> 隐伏点 CE(alpha, onehot) 指派监督"""
        set_ids = b["set_ids"]                              # [B,L]
        set_dirs = b["set_dirs"]                            # [B,G,3]
        B, L, K = alpha.shape
        # 真组 g -> 原型 p* = argmax_p |cos(mu_hat_p, set_dir_g)|
        cos = torch.einsum("bpe,bge->bpg", mu_hat, set_dirs).abs()  # [B,P,G]
        pstar = cos.argmax(1)                               # [B,G]
        y = torch.zeros(B, L, K, device=alpha.device)
        ok = torch.zeros(B, L, dtype=torch.bool, device=alpha.device)
        for bi in range(B):
            G = set_dirs.shape[1]
            valid = (set_ids[bi] >= 0) & (set_ids[bi] < G)
            g = set_ids[bi][valid]
            if len(g) == 0:
                continue
            y[bi][valid, pstar[bi, g]] = 1.0
            ok[bi] = valid
        ce = -(torch.log(alpha + 1e-8) * y).sum(-1)         # [B,L]
        w = ok.float() * (1.0 - mask)
        den = w.sum().clamp_min(1.0)
        return (ce * w).sum() / den