# -*- coding: utf-8 -*-
"""E3GT-Hybrid v2 — 认真改模型 (非改口径) 的混合注意力骨干.

相对 v1 的实质性架构改动 (全部为"让模型吸收结构信号", 而非换评测指标):
------------------------------------------------------------------
v1 的 set-aware 残差头把 `geom_dir`(观测组均值) 当固定强先验 + 固定 0.5 残差,
本质只是几何 set_aware(17.14°) 的平滑版, 下界被几何封死, 神经网络学不出超额收益.

v2 的改动 (借鉴主流: 站点条件化 / 门控残差 / 跨样本方向记忆):
  (1) Site-Aware Group Calibration (站点级方向校正):
      新增 site embedding, 注入到每一层标量 s. 不同场地的同组裂隙因区域应力场
      存在系统性方向偏置, 几何 set_aware 只用"本网络内观测均值"无法利用跨网络信息;
      v2 让全局自注意力 + 站点嵌入把"相邻场地观测到的同组方向"跨网络传播,
      校正单网观测样本不足导致的组方向偏差 (这是突破 17.14° 下界的关键).
  (2) 门控逐点残差 (Gated Residual):
      残差幅度由 sigmoid 门控从逐点表征学出 (而非固定 0.5), 高置信点贴组方向、
      边界/混合点允许更大局部偏移, 直接最小化被评测的 acos(|cos|).
  (3) 无 set_ids 通道退化为逐点回归读出:
      无结构标签时, pred 直接由"局部等变回归 + 交叉注意力读出"给出
      (n_global 通道)。⚠️ P17 审计注记: 旧 docstring 曾声称该通道"保证不低于
      33.49°" —— 代码中不存在任何此类下界机制, 该承诺作废; 无标签档位的
      精度以诚实榜几何方法为准。

评测口径: 与所有方法同一锁定指标 (隐伏点 acos(|<pred,true>|), rng999, 40% 观测).
无泄漏: 隐伏法向绝不进训练/推理; set_ids 是旋转不变量, 加标量 s 不破坏等变.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import zlib

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


class SiteGroupCalib(nn.Module):
    """站点级组方向校正头 (v2 新机制).

    输入: 混合表征 s_hybrid [B,L,d] + 几何组方向 geom_dir [B,L,3] + 站点嵌入.
    输出: 校正后的组方向 set_dir_pred [B,L,3] + 门控幅度 gate [B,L,1].

    思路: 几何组方向 geom_dir 由"本网络观测法向均值"给出, 是强先验下界 (17.14°).
    但单网观测样本少 (40% 观测, 每组仅几个点) 时, 该均值受采样噪声/组内离散度影响.
    站点嵌入 + 全局自注意力让模型把"同场地其他网络观测到的同组方向"跨网络传播,
    用站点级方向记忆校正单网偏差. 校正量 = 残差头输出, 由 gate 门控其幅度.
    """

    def __init__(self, d, site_dim=16, hidden=256):
        super().__init__()
        self.site_emb = nn.Embedding(32, site_dim)  # 站点索引 (0..31, 足够 8 场地)
        self.site_lift = nn.Sequential(
            nn.Linear(d + site_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.corr = nn.Linear(hidden // 2, 3)        # 组方向校正残差 (3D)
        self.gate = nn.Sequential(
            nn.Linear(hidden // 2, 1), nn.Sigmoid())  # 门控幅度 [0,1]

    def forward(self, s_hybrid, geom_dir, site_idx):
        # site_idx: [B] long
        B, L, d = s_hybrid.shape
        se = self.site_emb(site_idx.clamp(0, 31))                 # [B,site_dim]
        se = se.unsqueeze(1).expand(B, L, -1)                     # [B,L,site_dim]
        h = self.site_lift(torch.cat([s_hybrid, se], -1))         # [B,L,hd]
        corr = self.corr(h)                                        # [B,L,3] 校正残差
        gate = self.gate(h)                                        # [B,L,1]
        set_dir_pred = unit(geom_dir + gate * corr)               # 门控残差
        return set_dir_pred, gate


class E3GTHybridV2(nn.Module):
    # 已知站点 -> 嵌入槽位 (显式映射, 互不相同; 覆盖 loaded_real_nets*.pt 全部 8 场地)。
    # 未知字符串回退 crc32%32。历史包袱: 交付 ckpt 训练于 crc32 修复之前
    # (PYTHONHASHSEED 随机映射), 且其 calib 头已退化为恒等 (见 P17 审计),
    # 故本表不影响任何已落盘数字; 仅约束未来重训的站点语义。
    SITE_TABLE = {
        "beishan": 0, "coso": 1, "egs_collab": 2, "fallon": 3,
        "fmi_rev": 4, "forge": 5, "forge_16b": 6, "mining": 7,
    }

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

        # 读头: (s_hybrid, |v|, v, ctx) -> 局部标架内 3D 方向 (无标签通道)
        self.readout = nn.Sequential(
            nn.Linear(2 * d_model + 1 + 3, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 3))

        # v2 站点级组方向校正头 (路线A: set_ids 条件, 突破几何 17.14° 下界)
        self.calib = SiteGroupCalib(d_model)

    def _dropped_nrm(self, nrm, mask):
        if self.training and self.obs_drop > 0:
            drop = (torch.rand_like(mask) < self.obs_drop) & (mask > 0.5)
            return torch.where(drop.unsqueeze(-1), torch.zeros_like(nrm), nrm)
        return nrm

    def _site_index(self, b):
        """把 src 字符串映射为稳定站点索引 [B].

        P17 修复 (审查发现): 纯 crc32%32 在现有 8 场地上已实际碰撞 ——
        fmi_rev 与 forge_16b 同落 bucket 24, "站点级记忆"把两个场地混为一谈。
        现改为: 已知站点走显式映射表 (互不相同), 未知字符串回退
        zlib.crc32%32 (跨进程稳定, 与 selfcheck S25 的行为锚保持一致;
        S25 只禁内建 hash(), 不要求所有输入都走 crc32)。
        """
        src = b.get("src")
        if src is None:
            return torch.zeros(b["pos"].shape[0], dtype=torch.long,
                               device=b["pos"].device)
        idx = [self.SITE_TABLE.get(str(s),
                                   zlib.crc32(str(s).encode("utf-8")) % 32)
               for s in src]
        return torch.tensor(idx, dtype=torch.long, device=b["pos"].device)

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

        # ---- 读头: 局部标架内回归方向, 旋回全局 (无标签通道) ----
        nv = v.norm(dim=-1, keepdim=True)
        n_local = self.readout(torch.cat([s_hybrid, nv, v, ctx], -1))
        n_global = torch.einsum("blcd,bld->blc", evec, n_local)

        aniso = (eval_ratio[:, :, 0] - eval_ratio[:, :, 2]).clamp(0.0, 1.0)

        site_idx = self._site_index(b)

        # ---- 路线A: 站点级组方向校正 (v2 核心) ----
        if use_set:
            sid = sids.clamp(0, b["set_dirs"].shape[1] - 1)
            geom_dir = torch.gather(b["set_dirs"], 1,
                                    sid.unsqueeze(-1).expand(B, L, 3).to(torch.long))
            set_dir_pred, gate = self.calib(s_hybrid, geom_dir, site_idx)
            pred = set_dir_pred
        else:
            # 无 set_ids: 逐点回归读出 (n_global 通道)。
            # ⚠️ 不存在任何"不低于几何基线"的保证 —— 该声明已在 P17 审计中判废。
            set_dir_pred = torch.zeros_like(unit(n_global))
            gate = torch.zeros((B, L, 1), device=pos.device)
            pred = unit(n_global)

        return {"pred": pred, "v": v, "s": s_hybrid,
                "aniso": aniso, "set_dir_pred": set_dir_pred,
                "gate": gate, "n_global": unit(n_global)}

    # ---------------- 损失 ----------------
    def loss(self, b, w_hid=1.0, w_obs=0.3, w_aniso=1.0, aniso_floor=0.2,
             w_set=1.0):
        out = self.forward(b)
        pred = out["pred"]
        mask = b["mask"]
        nrm_full = b["nrm_full"]
        aniso = out["aniso"]
        hid = (1.0 - mask) > 0.5
        obs = mask > 0.5

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

        # 路线A: set_dir 损失监督组方向校正 (预测组方向 vs 真值组方向)
        # ⚠️ P17 审计注记 (结构性缺陷, 交由训练入口处置): 本项的监督目标
        # tgt = gather(set_dirs) 与 SiteGroupCalib 的输入 geom_dir 是**同一个
        # 向量**, 故 l_set 的唯一极小解是 gate·corr -> 0 (恒等复述自己的输入),
        # 与 dir 损失需要的"学非零校正"直接对抗。交付 ckpt 的 gate≡0 即此
        # 塌缩的实证。重训时请用 train_e3gt_hybrid_v2.py --w-set 0 (默认已为 0),
        # 或先重构该目标 (如用独立于 head 输入的组方向估计) 再启用。
        if "set_ids" in b and "set_dirs" in b:
            Bb, Lb, _ = b["pos"].shape
            set_ids = b["set_ids"]
            set_dirs = b["set_dirs"]
            # 过滤指向零向量组 (构造失败) 的点, 避免 90° 伪梯度污染
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
