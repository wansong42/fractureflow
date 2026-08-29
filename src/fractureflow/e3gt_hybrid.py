# -*- coding: utf-8 -*-
"""E3GT-Hybrid — 局部等变消息传递 + 全局自注意力 + 交叉注意力读出 (混合注意力).

v2 (路线 A / set-aware 升级, 2026-08-14)
------------------------------------------------------------------
在 v1 混合注意力基础上接入"组标签条件信号" set_ids (线路 A 离线球面 k-means
生成的组隶属, 真实数据全点已知, 不泄漏隐伏法向本身):

  (1) 条件注入: set_ids 经 embedding 加到旋转不变标量 s, 让每一层的等变消息
      传递都携带"这条点属于哪个裂隙组"的组身份 (set_ids 是旋转不变量, 加标量
      不破坏等变).
  (2) set-aware 读出: 当 batch 同时带 set_ids + set_dirs 时, 最终 pred 直接取
      set_dir_head 预测的"组方向" (端到端版 set_aware_dirs), 而非逐点自由回归.
      这把"组隶属"这个战役确认的唯一可解瓶颈, 以条件信号形式喂给模型.
  (3) 监督: 用 set_ids + 观测法向构造的组内方向 (set_dirs) 监督 set_dir_head,
      塑造表征显式学到组结构. 与几何 set_aware (17.14°) 同口径对照.

输出接口与 E3GTV4 完全一致: {"pred":[B,L,3], "aniso":[B,L], "set_dir_pred":[B,L,3]}.
主指标不变 (隐伏点 acos(|<pred,true>|)), 与几何基线/旧模型在同一固定口径公平对比.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .e3gt_v4 import EquivLayer, NormAct, unit
from .geometry import knn_graph, local_frames, build_graph_features

EPS = 1e-8


class GlobalSelfAttn(nn.Module):
    """全局 MH 自注意力 (set transformer): 在 [B,L,d] 的点集序列上做置换等变注意力.

    s 是旋转不变标量 (边距离/度/各向异性...), 故全局注意力不破坏等变性.
    """

    def __init__(self, d, heads=4, ff=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * ff), nn.GELU(),
            nn.Linear(d * ff, d), nn.Dropout(dropout))
        self.dp = nn.Dropout(dropout)

    def forward(self, s):  # s:[B,L,d]
        h = self.ln1(s)
        a, _ = self.attn(h, h, h)
        s = s + self.dp(a)
        s = s + self.ffn(self.ln2(s))
        return s


class CrossAttnReadout(nn.Module):
    """交叉注意力读出: query/key 来自标量 s, value = 局部标架法向 v.

    输出 ctx [B,L,d] = 对全体点法向的注意力池化 (局部标架). 注意力权重由标量 s
    决定 (学到的"指派核"), 故隐伏点会按学到的相关性聚合对应组的观测法向.
    """

    def __init__(self, d, heads=4):
        super().__init__()
        self.heads = heads
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(3, d // heads)  # 局部法向 -> 每头 hd 维

    def forward(self, s, v):  # s:[B,L,d]  v:[B,L,3] 局部标架法向
        B, L, d = s.shape
        hd = d // self.heads
        Q = self.Wq(s).view(B, L, self.heads, hd).permute(0, 2, 1, 3)  # [B,H,L,hd]
        K = self.Wk(s).view(B, L, self.heads, hd).permute(0, 2, 1, 3)
        V = self.Wv(v).unsqueeze(2).expand(B, L, self.heads, hd).permute(0, 2, 1, 3)
        logits = torch.einsum("bhqd,bhkd->bhqk", Q, K) / (hd ** 0.5)
        A = torch.softmax(logits, dim=-1)                          # [B,H,L,L]
        ctx = torch.einsum("bhqk,bhkd->bhqd", A, V)                 # [B,H,L,hd]
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, L, d)             # [B,L,d]
        return ctx


class E3GTHybrid(nn.Module):
    def __init__(self, d_model=256, k_knn=16, n_layers=4, n_global=2,
                 attn_heads=4, dropout=0.1, obs_drop=0.25, n_sets=8):
        super().__init__()
        self.k = k_knn
        self.d = d_model
        self.obs_drop = obs_drop
        self.n_sets = n_sets

        self.lift = nn.Sequential(
            nn.Linear(31, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))

        # 组标签条件 embedding (set_ids 旋转不变, 加标量 s 不破坏等变)
        self.set_embed = nn.Embedding(n_sets + 1, d_model)

        # 局部等变栈 (复用, 局部几何注意力)
        self.layers = nn.ModuleList(
            [EquivLayer(d_model, k_knn, dropout=dropout) for _ in range(n_layers)])

        # 混合注意力: 全局自注意力 (整网组结构) + 交叉注意力读出 (学指派核)
        self.global_attn = nn.ModuleList(
            [GlobalSelfAttn(d_model, attn_heads, dropout=dropout)
             for _ in range(n_global)])
        self.cross = CrossAttnReadout(d_model, attn_heads)

        # 读头: (s_hybrid, |v|, v, ctx) -> 局部标架内 3D 方向
        self.readout = nn.Sequential(
            nn.Linear(2 * d_model + 1 + 3, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 3))

        # set-aware 残差头: 在几何组方向(geom_dir = set_dirs 观测均值, 即 set_aware
        # 基线 17.14°) 之上学"组内局部偏移"残差。强先验保证下界=几何, 且可更低。
        self.residual_head = nn.Sequential(
            nn.Linear(d_model + 3 + d_model, d_model // 2), nn.ReLU(),
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

        # 图 + 局部标架 (与 E3GTV4 完全一致)
        idx, dist = knn_graph(pos, self.k)
        idx3 = idx.unsqueeze(-1).expand(B, L, self.k, 3)
        knn_pos = torch.gather(pos.unsqueeze(1).expand(B, L, L, 3), 2, idx3)
        evec, eval_ratio = local_frames(knn_pos, self.k)
        graph = (idx, dist, knn_pos, evec, eval_ratio)
        fnode, _, _, _ = build_graph_features(
            pos, nrm_in, log_len, lith, mask, s1, s3, self.k, graph=graph)
        radius = pos.std(1).mean(-1, keepdim=True).clamp_min(1e-3)

        nrm_u = unit(torch.where(nrm_in.norm(dim=-1, keepdim=True) > 1e-6,
                                 nrm_in, torch.zeros_like(nrm_in)))
        has_obs = (nrm_in.norm(dim=-1, keepdim=True) > 1e-6).float()
        v0 = torch.einsum("blc,blcd->bld", nrm_u, evec) * has_obs  # [B,L,3]

        s = self.lift(fnode)
        v = v0

        # ---- 组标签条件注入 (路线 A): set_ids -> embedding 加到标量 s ----
        sids = b.get("set_ids")
        use_set = (sids is not None) and ("set_dirs" in b)
        if use_set:
            sidx = sids.clamp(0, self.n_sets)            # [B,L]
            s = s + self.set_embed(sidx)                 # 旋转等变: 标量+标量

        rel = knn_pos - pos.unsqueeze(2)
        rel_local = torch.einsum("blke,blcd->blkd", rel, evec)

        # ---- 局部等变消息传递 (局部注意力) ----
        for layer in self.layers:
            s, v = layer(s, v, idx, dist, rel_local, evec, radius)

        # ---- 混合注意力 ----
        s_g = s
        for ga in self.global_attn:
            s_g = ga(s_g)
        s_hybrid = s + s_g                              # 局部 + 全局残差
        ctx = self.cross(s_hybrid, v)                   # 交叉注意力读出 (局部标架)

        # ---- 读头: 局部标架内回归方向, 旋回全局 ----
        nv = v.norm(dim=-1, keepdim=True)
        n_local = self.readout(torch.cat([s_hybrid, nv, v, ctx], -1))
        n_global = torch.einsum("blcd,bld->blc", evec, n_local)

        aniso = (eval_ratio[:, :, 0] - eval_ratio[:, :, 2]).clamp(0.0, 1.0)

        # set-aware 读出 v4: 几何组方向(geom_dir = set_dirs 观测均值, 即 set_aware 基线)
        # + 神经网络残差 (学组内局部偏移)。强先验保证下界=几何 17.14°, 且可更低。
        if use_set:
            sid = sids.clamp(0, b["set_dirs"].shape[1] - 1)
            geom_dir = torch.gather(b["set_dirs"], 1,
                                    sid.unsqueeze(-1).expand(B, L, 3).to(torch.long))
            residual = self.residual_head(torch.cat([s_hybrid, v, ctx], -1))
            set_dir_pred = unit(geom_dir + 0.5 * residual)
            pred = set_dir_pred
        else:
            set_dir_pred = torch.zeros_like(unit(n_global))
            pred = unit(n_global)

        return {"pred": pred, "v": v, "s": s_hybrid,
                "aniso": aniso, "set_dir_pred": set_dir_pred,
                "n_global": unit(n_global)}

    # ---------------- 损失 (逐字复用 E3GTV4, 含 set_dir 辅助) ----------------
    def loss(self, b, w_hid=1.0, w_obs=0.3, w_aniso=1.0, aniso_floor=0.2,
             w_set=1.0):
        out = self.forward(b)
        pred = out["pred"]
        mask = b["mask"]
        nrm_full = b["nrm_full"]
        aniso = out["aniso"]
        hid = (1.0 - mask) > 0.5
        obs = mask > 0.5

        # set-aware 模式 pred = 组方向+逐点残差门控融合, 受 dir 损失(逐点真值);
        # 组方向纯度由 set_dir 损失单独保证。无标签模式 pred = 逐点读出 n_global。
        cos = (pred * nrm_full).sum(-1).abs().clamp(0, 1)
        err1 = 1.0 - cos

        total = 0.0
        comp = {}
        if hid.any():
            aw = torch.clamp(aniso / 0.5, aniso_floor, 1.0) * w_aniso
            l_hid = (err1[hid] * aw[hid]).mean() / max(float(aw[hid].mean()), 1e-3)
            total = total + w_hid * l_hid
            comp["l_hid"] = float(err1[hid].mean())
        if obs.any():
            l_obs = err1[obs].mean()
            total = total + w_obs * l_obs
            comp["l_obs"] = float(l_obs)

        if "set_ids" in b and "set_dirs" in b:
            # ⚠️ P17 审计注记: 本项监督目标 tgt = gather(set_dirs) 与读出路径
            # 的 geom_dir 输入是**同一个向量**, 唯一极小解是 residual -> 0
            # (恒等复述输入), 与 dir 损失对抗。重训入口已默认 --w-set 0。
            Bb, Lb, _ = b["pos"].shape
            set_ids = b["set_ids"]
            set_dirs = b["set_dirs"]
            # 过滤指向零向量组 (构造失败/超出 max_sets) 的点, 避免 90° 伪梯度
            # 污染训练 (与 e3gt_hybrid_v2.loss 同一修复)
            sd_norm = set_dirs.norm(dim=-1)                       # [B, max_sets]
            valid_group = sd_norm > 1e-3                          # 有方向的组
            sid_clamped = set_ids.clamp(0, set_dirs.shape[1] - 1)
            valid = (set_ids >= 0) & valid_group.gather(1, sid_clamped)
            sid = sid_clamped
            idx = sid.unsqueeze(-1).expand(Bb, Lb, 3).to(torch.long)
            tgt = torch.gather(set_dirs, 1, idx)
            cos_s = (out["set_dir_pred"] * tgt).sum(-1).abs().clamp(0, 1)
            err_s = 1.0 - cos_s
            if valid.any():
                l_set = err_s[valid].mean()
                total = total + w_set * l_set
                comp["l_set"] = float(l_set)

        comp["total"] = float(total)
        return total, comp
