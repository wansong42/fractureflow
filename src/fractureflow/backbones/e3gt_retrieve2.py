# -*- coding: utf-8 -*-
"""E3GTRetrieve2 骨干 (注册 "e3gt_retrieve2") —— 冲击 13° 的第二代检索头。

背景 / 已验证事实:
  - 探针 cand_test.py: 对每个隐伏点, 与其真值法向最接近的"观测法向"就在其
    ~20~40 个空间最近邻观测点里 (非泄漏下界 9.14°, oracle 最优候选 6.56°)。
  - 第一代 e3gt_retrieve (softmax 加权检索) 实测 23.39° (基线 34.68°),
    但停在"指派完美地板 23.15°"附近上不去。

第一代的三处根因 (本文件逐条修掉):
  1. **查询对排序完全无影响**: 原打分 score = score_q(q) + score_c(c),
     score_q(q) 是逐点常数, softmax 后被约掉 => 选哪个候选只由候选特征决定,
     网络其实无法"因点而异"地检索。本代改为真正的点积注意力
     score = <q_i, k_c>/sqrt(d) + b_c, 查询才真正参与排序。
  2. **观测点自泄漏捷径**: 原 knn_obs 对观测点会把"它自己"选成距离 0 的候选,
     l_obs 项 (w=0.3) 于是奖励"永远挑最近候选"这一退化捷径, 反过来污染隐伏点。
     本代显式剔除自身 (d<eps 置 inf), 观测点变成真正的自监督任务。
  3. **候选特征没有判别力**: 原特征只有 (距离, cand·base, cand·gp),
     其中 cand·base 的 argmax 天生就是"最像 gp 的候选" => 网络最多学到
     "贴着 gp 走", 这正是卡在 ~23° 的原因。本代注入真正的**物理判别量**:
     若隐伏点 i 与观测点 j 同属一条裂隙 (平面), 则 (p_j - p_i) ⟂ n_j,
     即 共面残差 |<p_j-p_i, n_j>| ≈ 0。这是"i 是否落在 j 的裂隙面上"的直接证据,
     也是把 23° 推向 13° 所缺的那一味信息。

其他改进:
  - **无符号聚合 (scatter + 幂迭代)**: 轴向数据的正确聚合是加权散布矩阵
    S = Σ w_c o_c o_c^T 的主特征向量, 天然符号不变。于此彻底去掉第一代
    "训练期用 nrm_full 对齐候选符号"的 train/test 口径错配 (也顺手消除该泄漏)。
  - **两级级联**: 第一遍以 base(≈gp, 34.68°) 为锚做检索得 pred1; 第二遍改用
    pred1 作参考重算判别特征再检索。锚点从 34.68° 提升到 ~23° 后,
    共面/夹角类特征的信噪比大幅提升。
  - **可学 softmax 温度**: 让网络自行从"平滑平均"锐化到"近 argmax 选择"。
  - 全部候选打分特征都是旋转不变量 (局部标架投影 / 内积), 保持等变性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import BACKBONES, unit
from ..geometry import knn_graph, local_frames, build_graph_features
from ..geo_prior import geo_prior_dirs_gpu, geo_prior_candidates_gpu
from .e3gt_prior import E3GTPrior


def knn_obs_rel(pos, nrm, mask, k, chunk=1024):
    """每点空间最近的 k 个**观测点** (剔除自身), 返回法向与相对位移。

    Returns:
        cand_o : [B, L, k, 3]  观测法向 (原始符号, 后续用无符号聚合)
        rel    : [B, L, k, 3]  p_j - p_i (全局系)
        radius : [B, 1, 1, 1]  网络尺度 (用于无量纲化)
    """
    B, L, _ = pos.shape
    dev = pos.device
    radius = pos.std(1).mean(-1).clamp_min(1e-3)                  # [B]
    Lobs = mask.sum(1).long()
    maxobs = max(int(Lobs.max()), 1)
    obs_pos = torch.zeros(B, maxobs, 3, device=dev)
    obs_nrm = torch.zeros(B, maxobs, 3, device=dev)
    for b in range(B):
        n = int(Lobs[b])
        if n > 0:
            m = mask[b].bool()
            obs_pos[b, :n] = pos[b][m]
            obs_nrm[b, :n] = nrm[b][m]
    d = torch.empty(B, L, maxobs, device=dev)
    for i in range(0, L, chunk):
        j = min(i + chunk, L)
        diff = pos[:, i:j, None, :] - obs_pos[:, None, :, :]
        d[:, i:j] = (diff * diff).sum(-1)
    pad = torch.arange(maxobs, device=dev)[None, None, :] >= Lobs[:, None, None]
    d = d.masked_fill(pad, float("inf"))
    # 关键: 剔除"自己" (观测点到自身距离 0) —— 否则 l_obs 会奖励"总挑最近候选"的退化捷径
    d = d.masked_fill(d < 1e-10, float("inf"))
    kk = min(k, maxobs)
    dtop, didx = torch.topk(d, kk, dim=-1, largest=False)
    if kk < k:
        pn = k - kk
        didx = torch.cat([didx, didx[:, :, -1:].repeat(1, 1, pn)], dim=2)
        dtop = torch.cat([dtop, dtop[:, :, -1:].repeat(1, 1, pn)], dim=2)
    off = torch.arange(B, device=dev)[:, None, None] * maxobs
    fl = (didx + off).reshape(-1)
    cand_o = obs_nrm.reshape(B * maxobs, 3)[fl].reshape(B, L, k, 3)
    cand_p = obs_pos.reshape(B * maxobs, 3)[fl].reshape(B, L, k, 3)
    rel = cand_p - pos.unsqueeze(2)
    # 观测点全被 inf 屏蔽的极端情形 (网络里只有 1 个观测点): rel/cand 归零, 由注意力自行忽略
    bad = torch.isinf(dtop).unsqueeze(-1)
    cand_o = torch.where(bad, torch.zeros_like(cand_o), cand_o)
    rel = torch.where(bad, torch.zeros_like(rel), rel)
    return cand_o, rel, radius[:, None, None, None]


def _power_iter(S, v0, iters=6):
    """加权散布矩阵 S 的主特征向量 (幂迭代, 由 v0 定半球)。"""
    v = v0
    for _ in range(iters):
        v = torch.einsum("blij,blj->bli", S, v)
        v = unit(v)
    return v


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


N_CFEAT = 14


@BACKBONES.register("e3gt_retrieve2")
class E3GTRetrieve2(E3GTPrior):
    def __init__(self, d_model=256, k_knn=16, n_layers=4, dropout=0.1,
                 obs_drop=0.25, k_cand=32, Ks=(3, 4, 5, 6), n_groups=6,
                 d_head=64, pow_iters=6, n_pass=2):
        super().__init__(d_model=d_model, k_knn=k_knn, n_layers=n_layers,
                         dropout=dropout, obs_drop=obs_drop)
        self.k_cand = k_cand
        self.Ks = tuple(Ks)
        self.n_geom = 2 * len(self.Ks) + 1
        self.d_head = d_head
        self.pow_iters = pow_iters
        self.n_pass = n_pass

        self.cfeat = _mlp(N_CFEAT, d_head, d_head)          # 候选不变量编码 (两遍共享)
        self.key = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(n_pass)])
        self.cbias = nn.ModuleList([nn.Linear(d_head, 1) for _ in range(n_pass)])
        self.qproj = nn.ModuleList([_mlp(d_model + 1 + 3 + 3, d_model, d_head)
                                    for _ in range(n_pass)])
        self.log_tau = nn.Parameter(torch.zeros(n_pass))     # 可学锐化温度
        gates = []
        for _ in range(n_pass):
            g = _mlp(d_model + 1 + 3 + 3, d_model // 2, 1)
            nn.init.zeros_(g[-1].weight)
            nn.init.zeros_(g[-1].bias)
            gates.append(g)
        self.gate = nn.ModuleList(gates)
        self.memb_head = _mlp(d_model + 3, d_model // 2, n_groups)

    # ---------- 候选不变量 ----------
    def _cand_feats(self, cand, rel_u, dist_n, plane_off, rank, ref, gp, evec):
        """全部为旋转不变标量 => 保持等变性。"""
        ab = lambda x: x.abs()
        f = [
            dist_n,                                              # 1 归一化距离
            torch.log1p(dist_n),                                 # 2
            rank,                                                # 3 近邻序号
            plane_off,                                           # 4 共面残差 |<p_j-p_i, n_j>|/r  ★
            ab((rel_u * cand).sum(-1, keepdim=True)),            # 5 单位共面残差 ★
            ab((cand * ref.unsqueeze(2)).sum(-1, keepdim=True)),  # 6 候选·参考方向
            ab((cand * gp.unsqueeze(2)).sum(-1, keepdim=True)),   # 7 候选·几何先验
            ab((rel_u * ref.unsqueeze(2)).sum(-1, keepdim=True)),  # 8 位移·参考方向
            ab(torch.einsum("blkc,blcd->blkd", cand, evec)),      # 9-11 候选在局部标架投影
            ab(torch.einsum("blkc,blcd->blkd", rel_u, evec)),     # 12-14 位移在局部标架投影
        ]
        return torch.cat(f, dim=-1)

    def _retrieve(self, p, s, nv, v, evec, cand, rel_u, dist_n, plane_off, rank,
                  anchor, gp):
        ref = anchor.detach()
        feats = self._cand_feats(cand, rel_u, dist_n, plane_off, rank, ref, gp, evec)
        ce = self.cfeat(feats)                                    # [B,L,Kc,dh]
        kc = self.key[p](ce)
        anchor_local = torch.einsum("blc,blcd->bld", ref, evec)
        q = self.qproj[p](torch.cat([s, nv, v, anchor_local], -1))  # [B,L,dh]
        score = (kc * q.unsqueeze(2)).sum(-1) / (self.d_head ** 0.5)
        score = score + self.cbias[p](ce).squeeze(-1)
        score = score * self.log_tau[p].exp().clamp(0.05, 20.0)
        score = score - score.amax(dim=2, keepdim=True)
        w = F.softmax(score, dim=2)                               # [B,L,Kc]

        cw = cand * w.unsqueeze(-1)
        S = torch.einsum("blki,blkj->blij", cw, cand)             # 加权散布矩阵
        ret = _power_iter(S, anchor.detach(), self.pow_iters)
        ret = ret * torch.sign((ret * anchor.detach()).sum(-1, keepdim=True) + 1e-12)

        dot_ra = (ret * anchor.detach()).sum(-1, keepdim=True)
        wmax = w.max(dim=2, keepdim=False)[0].unsqueeze(-1)
        went = -(w.clamp_min(1e-9) * w.clamp_min(1e-9).log()).sum(2, keepdim=True)
        g = torch.sigmoid(self.gate[p](torch.cat([s, nv, v, dot_ra, wmax, went], -1)))
        pred = unit(anchor + g * (ret - anchor))
        return pred, w, ret, g

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

        # ---- 候选集: 几何组方向 + 空间最近观测法向 (剔除自身) ----
        cand_g = geo_prior_candidates_gpu(pos, nrm, mask, fast=self.prior_fast,
                                          Ks=self.Ks).detach()
        cand_o, rel, radius4 = knn_obs_rel(pos, nrm, mask, self.k_cand)
        cand_o = cand_o.detach(); rel = rel.detach()
        cand = torch.cat([cand_g, cand_o], dim=2).detach()         # [B,L,Kc,3]
        Kc = cand.shape[2]
        ng = cand_g.shape[2]

        dlen = rel.norm(dim=-1, keepdim=True)
        rel_u = rel / dlen.clamp_min(1e-9)
        dist_o = (dlen / radius4).clamp(0.0, 8.0)
        plane_o = ((rel * cand_o).sum(-1, keepdim=True).abs() / radius4).clamp(0.0, 8.0)
        rank_o = (torch.arange(self.k_cand, device=pos.device, dtype=pos.dtype)
                  / max(self.k_cand - 1, 1))[None, None, :, None].expand(B, L, self.k_cand, 1)
        z = torch.zeros(B, L, ng, 1, device=pos.device, dtype=pos.dtype)
        dist_n = torch.cat([z, dist_o], 2)
        plane_off = torch.cat([z, plane_o], 2)
        rank = torch.cat([z, rank_o], 2)
        rel_u_all = torch.cat([torch.zeros(B, L, ng, 3, device=pos.device,
                                           dtype=pos.dtype), rel_u], 2)

        # ---- 等变消息传递 ----
        nrm_u = unit(torch.where(nrm_in.norm(dim=-1, keepdim=True) > 1e-6,
                                 nrm_in, torch.zeros_like(nrm_in)))
        has_obs = (nrm_in.norm(dim=-1, keepdim=True) > 1e-6).float()
        v0 = torch.where(has_obs > 0.5,
                         torch.einsum("blc,blcd->bld", nrm_u, evec), gp_local)
        s = self.lift(fnode) + self.prior_proj(gp_local)
        v = v0
        rel_knn = knn_pos - pos.unsqueeze(2)
        rel_local = torch.einsum("blke,blcd->blkd", rel_knn, evec)
        radius = pos.std(1).mean(-1, keepdim=True).clamp_min(1e-3)
        for layer in self.layers:
            s, v = layer(s, v, idx, dist, rel_local, evec, radius=radius)

        nv = v.norm(dim=-1, keepdim=True)
        c_local = self.corr(torch.cat([s, nv, v], -1))
        base = unit(gp + torch.einsum("blcd,bld->blc", evec, c_local))

        # ---- 级联检索 ----
        anchor = base
        preds = []; ws = []; gates = []
        for p in range(self.n_pass):
            anchor, w, ret, g = self._retrieve(
                p, s, nv, v, evec, cand, rel_u_all, dist_n, plane_off, rank,
                anchor, gp)
            preds.append(anchor); ws.append(w); gates.append(g)

        aniso = (eval_ratio[:, :, 0] - eval_ratio[:, :, 2]).clamp(0.0, 1.0)
        set_dir_pred = unit(self.set_dir_head(torch.cat([s, v], -1)))
        memb = self.memb_head(torch.cat([s, v], -1))

        out = {"pred": preds[-1], "v": v, "s": s, "aniso": aniso,
               "set_dir_pred": set_dir_pred, "memb": memb,
               "w": ws[-1], "cand": cand, "base": base,
               "gate": gates[-1]}
        for i, pv in enumerate(preds[:-1]):
            out[f"pred{i + 1}"] = pv
        return out
