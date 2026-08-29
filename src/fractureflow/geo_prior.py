# -*- coding: utf-8 -*-
"""GPU 版几何先验 top2_ens (与 inference.top2_ens_dirs 同算法, 纯 torch 批量化实现)。

为何存在
--------
E3GTV4 连续回归在隐伏点上从 0 起步预测方向, 而几何基线 top2_ens 用"附近观测方向
的组合"作强归纳偏置, 反而更强 (34.68° vs 模型 39.9°)。本模块把同一套几何先验在
**模型 forward 内、用当前增广后的 (pos,nrm,mask) 现场算出来**, 作为:
  (1) 隐伏点向量通道的起手方向 (不再从 0 起步);
  (2) 残差校正的基底: pred = normalize(gp + 修正)。
=> 模型起手即 34.68° 地板, 只能往上逼近, 不会假突破。

训练期性能: 精确版 (默认 fast=False) 与 numpy top2_ens 逐点一致, 但每个前向都要
算球面 K-means (批量 Python 循环), 很慢 (~17s/step @ b32)。**本版已批量化**:
kmeans / kNN / top2 混合全部在 [B,...] 上向量化, 去掉了逐网 Python 循环,
训练期快约一个数量级。评测期 (fast=False) 仍与 numpy top2_ens 一致。

关键不变量: geo_prior_dirs_gpu 在 fast=False 下 == inference.top2_ens_dirs (差 <0.3°)
—— ⚠️ **该保证严格成立仅限 B=1 或批内位置 0**。H1 known-fail (2026-08 实测,
selfcheck S30 档案): _kmeans_dirs_batched 共享 default_rng(seed) 按批内顺序消费,
第 b 个网的初始化依赖其批位置; 组系模糊数据上位置 >=1 网与逐网参考最大可差
~41° (合成 40 点/50% 观测实测)。诚实榜 8 方法全 numpy 路径不受影响;
计算冻结不动 (修复会断 e3gt 系 checkpoint 复现), 推迟至重训时代 —— 对外
表述"模型起手==地板"仅限 B=1 口径。详见 docs/评测标准与口径锁定.md §geo_prior。
候选集 (candidates/obsneighbor) 仅用于指针头的候选加权, 不影响地板
(指针零门控时 pred==gp)。
"""

import numpy as np
import torch
import torch.nn.functional as F

from .core import unit


# ---------------------------------------------------------------------------
# 标量版本 (保留用于数值验证 / 兜底): 与 numpy top2_ens 逐点一致
# ---------------------------------------------------------------------------
def _kmeans_dirs(nrm_o, K, iters=60, seed=0):
    """球面 k-means, 返回 [K,3] 单位中心 (与 inference.kmeans_dirs 同算法)。"""
    pts = unit(nrm_o)
    n = pts.shape[0]
    Kk = min(K, n)
    if Kk <= 1:
        return unit(pts.mean(0, keepdim=True))
    gen = torch.Generator(device=pts.device).manual_seed(seed)
    sel = torch.randperm(n, device=pts.device, generator=gen)[:Kk]
    centers = pts[sel].clone()                              # [Kk,3]
    for _ in range(iters):
        cos = pts @ centers.T                              # [n,Kk]
        assign = cos.argmax(1)                             # 有向指派
        for k in range(Kk):
            m = assign == k
            if m.any():
                centers[k] = unit(pts[m].mean(0))
    return centers


def _pipe_top2(pos_b, nrm_o, occ, nrm_b, K, knn, min_frac, top2, iters=60):
    """单 K 的 top2 混合管道 (对应 inference._pipe_top2), 返回 [L,3]。"""
    L = pos_b.shape[0]
    nocc = nrm_o.shape[0]
    Kk = min(K, nocc)
    cents = _kmeans_dirs(nrm_o, Kk, iters=iters)            # [Kk,3]
    g_obs = (nrm_o @ cents.T).abs().argmax(1)              # [nocc] 无向指派
    mu = torch.zeros(Kk, 3, device=pos_b.device)
    for c in range(Kk):
        m = g_obs == c
        mu[c] = unit(nrm_o[m].mean(0)) if m.any() else cents[c]
    d2 = ((pos_b[:, None] - pos_b[occ][None]) ** 2).sum(-1)   # [L, nocc]
    kn = min(knn, nocc)
    kk = d2.argsort(1)[:, :kn]                            # [L, kn]
    top = g_obs[kk]                                       # [L, kn]
    wcnt = torch.stack([(top == c).sum(1) for c in range(Kk)], 1)  # [L, Kk]
    order = wcnt.argsort(1, descending=True)              # [L, Kk]
    g_pred = order[:, 0]                                  # [L]
    denom = wcnt.sum(1, keepdims=True) + 1e-9
    frac = wcnt.gather(1, g_pred.unsqueeze(1)) / denom    # [L,1]
    nn = d2.argmin(1)                                     # [L]
    ref = nrm_o[nn]                                       # [L,3]
    bb = frac / max(min_frac, 1e-9)
    bb = bb.clamp(0.0, 1.0)                               # [L,1]
    gmu = mu[g_pred]                                      # [L,3]
    g2 = order[:, 1]                                      # [L]
    f2 = wcnt.gather(1, g2.unsqueeze(1)) / denom          # [L,1]
    d1 = unit(gmu * torch.sign((gmu * ref).sum(-1, keepdims=True) + 1e-12))
    d2_ = unit(mu[g2] * torch.sign((mu[g2] * ref).sum(-1, keepdims=True) + 1e-12))
    t = (top2 * f2 / (frac + f2 + 1e-9)).clamp(0.0, 1.0)  # [L,1]
    v = unit(d1 * (1.0 - t) + d2_ * t)
    v = unit(v * bb + ref * (1.0 - bb))
    out = torch.zeros(L, 3, device=pos_b.device)
    out[occ] = nrm_b[occ]
    out[~occ] = v[~occ]
    return out


def geo_prior_dirs_gpu_scalar(pos, nrm, mask, Ks=(3, 4, 5, 6), knn=24,
                              min_frac=0.8, top2=0.6, iters=60, fast=False):
    """标量版 (保留验证用): 与 numpy top2_ens 逐点一致。"""
    if fast:
        Ks = (4,); knn = 16; iters = 12
    B, L, _ = pos.shape
    dev = pos.device
    out = torch.zeros(B, L, 3, device=dev)
    for b in range(B):
        occ = mask[b] > 0.5
        nocc = int(occ.sum())
        if nocc == 0:
            continue
        nrm_b = nrm[b]
        if nocc < 4:
            v = unit(nrm_b[occ].mean(0, keepdim=True)).repeat(L, 1)
            out[b] = v
            continue
        nrm_o = unit(nrm_b[occ])                          # [nocc,3]
        pipes = [_pipe_top2(pos[b], nrm_o, occ, nrm_b, K, knn, min_frac, top2, iters)
                 for K in Ks]
        ref_idx = Ks.index(4) if 4 in Ks else 0
        ref = pipes[ref_idx]
        hid = ~occ
        acc = torch.zeros(int(hid.sum()), 3, device=dev)
        for p in pipes:
            sgn = torch.sign((p[hid] * ref[hid]).sum(-1, keepdims=True) + 1e-12)
            acc = acc + p[hid] * sgn
        v_hid = unit(acc)
        out[b, occ] = nrm_b[occ]
        out[b, hid] = v_hid
    return out


# ---------------------------------------------------------------------------
# 批量化核心 (训练/评测统一走这里, 去掉逐网 Python 循环)
# ---------------------------------------------------------------------------
def _observed_padded(nrm, mask):
    """取出每网观测法向并 pad 到 nmax。

    返回 nrm_o [B,nmax,3] (pad 处为 0), obs_valid [B,nmax] (bool),
    obs_idx [B,nmax] (观测点在 L 中的索引)。collate 同组 L 一致, 仅 nocc 可变。
    """
    B, L, _ = nrm.shape
    dev = nrm.device
    occ_list = [torch.where(mask[b] > 0.5)[0] for b in range(B)]
    nmax = max((idx.numel() for idx in occ_list), default=0)
    nrm_o = torch.zeros(B, nmax, 3, device=dev)
    obs_valid = torch.zeros(B, nmax, dtype=torch.bool, device=dev)
    obs_idx = torch.zeros(B, nmax, dtype=torch.long, device=dev)
    for b, idx in enumerate(occ_list):
        k = idx.numel()
        if k > 0:
            nrm_o[b, :k] = nrm[b][idx]
            obs_valid[b, :k] = True
            obs_idx[b, :k] = idx
    return nrm_o, obs_valid, obs_idx


def _kmeans_dirs_batched(nrm_o, obs_valid, K, iters=60, seed=0, cluster_cap=None):
    """向量化球面 k-means: nrm_o [B,nmax,3] (pad 处 obs_valid=False),
    返回 centers [B,K,3]。K 直接使用调用方给定值 (调用方已按 min/max_valid 处理退化);
    观测数 < K 的网用重复初值, 空簇由 cnt.clamp_min 兜底 (mu=0, 不被选中)。

    BUG-6 修复 (2026-08): cluster_cap [B] 为各网有效簇数上限 (min(K, n_occ), 下限 2),
    超限簇在指派阶段被屏蔽 (永不分到点)。旧实现整批共用 min(n_occ) 个簇 ——
    批内只要有一个低观测网, 所有网的 k-means 组数都被压到 ≤2, 偏离 numpy
    版逐网行为 (domain_rand 低 frac 训练下会触发)。"""
    Bb, n, _ = nrm_o.shape
    Kk = K
    dev = nrm_o.device
    if cluster_cap is not None:
        cmask = torch.arange(Kk, device=dev)[None, :] < cluster_cap[:, None]   # [B,K]
    else:
        cmask = None
    # 关键修复: 必须与 inference.kmeans_dirs 用完全相同的初始化 (np.random.default_rng(seed).choice),
    # 否则 torch.randperm 与 numpy 选择不同初值, 球面 k-means 在真实数据上对初值敏感,
    # 会导致 geo_prior_dirs_gpu 与 top2_ens_dirs 逐点差 ~6.7° (模型 gp 比基线弱/不一致)。
    sel = torch.zeros(Bb, Kk, dtype=torch.long, device=dev)
    np_rng = np.random.default_rng(seed)
    for b in range(Bb):
        k = int(obs_valid[b].sum())
        if k > 0:
            valid_idx = torch.where(obs_valid[b])[0].cpu().numpy()
            kk_b = min(Kk, len(valid_idx))
            perm = np_rng.choice(len(valid_idx), kk_b, replace=False)
            sel[b, :kk_b] = torch.tensor(perm, device=dev, dtype=torch.long)
            if kk_b < Kk:   # 观测数 < Kk: 复制首心, 空簇由 cnt.clamp_min 兜底
                sel[b, kk_b:] = sel[b, 0]
    centers = nrm_o.gather(1, sel.unsqueeze(-1).expand(Bb, Kk, 3))  # [B,Kk,3]
    neg = torch.tensor(-1.0, device=dev)
    for _ in range(iters):
        cos = nrm_o @ centers.transpose(1, 2)              # [B,n,Kk]
        cos = torch.where(obs_valid.unsqueeze(-1), cos, neg)
        if cmask is not None:
            cos = torch.where(cmask.unsqueeze(1), cos, neg)
        assign = cos.argmax(-1)                            # [B,n]
        onehot = F.one_hot(assign, Kk).float()            # [B,n,Kk]
        onehot = onehot * obs_valid.unsqueeze(-1).float()
        cnt = onehot.sum(1)                                # [B,Kk]
        nonempty = cnt > 0
        acc = onehot.transpose(1, 2) @ nrm_o               # [B,Kk,3]
        new_centers = unit(acc / cnt.unsqueeze(-1).clamp_min(1.0))
        # 空簇保持上一轮 centroid (与 inference.kmeans_dirs 行为一致: 空簇不更新);
        # 否则 torch 会把空簇 centroid 置 0, 与 numpy 产生 ~2.5° 残差。
        centers = torch.where(nonempty.unsqueeze(-1), new_centers, centers)
    return centers


def _pipe_top2_batched(pos, nrm_o, obs_valid, obs_idx, nrm_b, mask,
                       K, knn, min_frac, top2, iters=60):
    """单 K 的 top2 混合管道 (批量化)。返回 (out [B,L,3], d1 [B,L,3], d2_ [B,L,3], ref [B,L,3])。

    d1/d2_ 为 primary/secondary 组方向 (供指针候选), ref 为最近观测法向。
    """
    B, L, _ = pos.shape
    dev = pos.device
    n_occ = obs_valid.sum(1)                                          # [B]
    if bool((n_occ > 0).any()):
        max_valid = int(n_occ.max().item())
        # 各网有效簇数 = min(K, n_occ) (对齐 numpy 版逐网行为), 下限 2
        # (order[:,:,1] 取次级方向需要 Kk>=2; 全批无观测时才退化为 1)。
        cap = torch.clamp(n_occ, min=2).clamp(max=K)
        Kk = int(cap.max().item()) if max_valid >= 2 else 1
        cluster_cap = cap if max_valid >= 2 else None
    else:
        Kk = 1
        cluster_cap = None
    big = torch.tensor(1e9, device=dev)
    neg = torch.tensor(-1.0, device=dev)

    cents = _kmeans_dirs_batched(nrm_o, obs_valid, Kk, iters=iters,
                                 cluster_cap=cluster_cap)             # [B,Kk,3]
    cmask = (torch.arange(Kk, device=dev)[None, :] < cluster_cap[:, None]) \
        if (cluster_cap is not None and Kk > 1) else None
    cos_ob = (nrm_o @ cents.transpose(1, 2)).abs()                    # [B,nmax,Kk]
    cos_ob = torch.where(obs_valid.unsqueeze(-1), cos_ob, neg)
    if cmask is not None:
        cos_ob = torch.where(cmask.unsqueeze(1), cos_ob, neg)
    g_obs = cos_ob.argmax(-1)                                         # [B,nmax]
    onehot = F.one_hot(g_obs, Kk).float() * obs_valid.unsqueeze(-1).float()  # [B,nmax,Kk]
    cnt = onehot.sum(1)                                              # [B,Kk]
    nonempty = cnt > 0
    mu_new = unit((onehot.transpose(1, 2) @ nrm_o) / cnt.unsqueeze(-1).clamp_min(1.0))
    # 空簇 mu 用 k-means centroid cents (与 numpy _pipe_top2 一致: 空簇取 cents[c]);
    # 不能用 0, 否则 d1/d2_ 被污染 (~1.8° 残差来源)。
    mu = torch.where(nonempty.unsqueeze(-1), mu_new, cents)           # [B,Kk,3]

    pos_obs = pos.gather(1, obs_idx.unsqueeze(-1).expand(B, obs_idx.shape[1], 3))  # [B,nmax,3]
    d2 = ((pos.unsqueeze(2) - pos_obs.unsqueeze(1)) ** 2).sum(-1)      # [B,L,nmax]
    d2 = torch.where(obs_valid.unsqueeze(1), d2, big)
    kn = min(knn, int(obs_valid.sum(1).max().item()))
    kk = d2.argsort(2)[:, :, :kn]                                     # [B,L,kn]
    # g_obs 是 [B,nmax] 的逐观测组指派; kk 是 [B,L,kn] 对观测维度的索引。
    # 先把 g_obs 沿 L 维展开到 [B,L,nmax], 再沿 dim=2 用 kk 收集 => [B,L,kn]
    g_obs_exp = g_obs.unsqueeze(1).expand(B, L, g_obs.shape[1])      # [B,L,nmax]
    top = g_obs_exp.gather(2, kk)                                    # [B,L,kn]
    wcnt = torch.stack([(top == c).sum(2) for c in range(Kk)], 2)      # [B,L,Kk]
    order = wcnt.argsort(2, descending=True)                          # [B,L,Kk]
    g_pred = order[:, :, 0]                                           # [B,L]
    denom = wcnt.sum(2, keepdims=True) + 1e-9                         # [B,L,1]
    frac = wcnt.gather(2, g_pred.unsqueeze(-1)) / denom               # [B,L,1]
    nn = d2.argmin(2)                                                 # [B,L]
    ref = nrm_o.gather(1, nn.unsqueeze(-1).expand(B, L, 3))           # [B,L,3]
    bb = (frac / max(min_frac, 1e-9)).clamp(0.0, 1.0)                 # [B,L,1]
    gmu = torch.gather(mu, 1, g_pred.unsqueeze(-1).expand(B, L, 3))   # [B,L,3]
    g2 = order[:, :, 1] if Kk >= 2 else g_pred   # Kk==1 时退化为 primary (d2_=d1)
    f2 = wcnt.gather(2, g2.unsqueeze(-1)) / denom                     # [B,L,1]
    d1 = unit(gmu * torch.sign((gmu * ref).sum(-1, keepdims=True) + 1e-12))
    mu_g2 = torch.gather(mu, 1, g2.unsqueeze(-1).expand(B, L, 3))
    d2_ = unit(mu_g2 * torch.sign((mu_g2 * ref).sum(-1, keepdims=True) + 1e-12))
    t = (top2 * f2 / (frac + f2 + 1e-9)).clamp(0.0, 1.0)              # [B,L,1]
    v = unit(d1 * (1.0 - t) + d2_ * t)
    v = unit(v * bb + ref * (1.0 - bb))
    out = torch.where(mask.bool().unsqueeze(-1), nrm_b, v)            # 观测点取观测法向
    return out, d1, d2_, ref


def geo_prior_dirs_gpu(pos, nrm, mask, Ks=(3, 4, 5, 6), knn=24,
                       min_frac=0.8, top2=0.6, iters=60, fast=False):
    """批量几何先验 (与 inference.top2_ens_dirs 同算法, 向量化)。

    输入 pos/nrm/mask: [B,L,3]/[B,L,3]/[B,L]; nrm 为掩码后观测法向 (隐伏点=0)。
    返回 gp [B,L,3] 单位方向 (全局系)。fast=True 仅用于训练期提速 (单 K=4 管道 + 少迭代)。
    """
    if fast:
        Ks = (4,); knn = 16; iters = 12
    # 几何核心用 float64 计算, 输出转回 float32 — 避免 float32 在 cos argmax 近平局点
    # 翻转指派 (这是 batched 与 numpy top2_ens 出现 ~0.8° MAE 差距的主因)。
    pos = pos.double(); nrm = nrm.double()
    B, L, _ = pos.shape
    dev = pos.device
    nrm_o, obs_valid, obs_idx = _observed_padded(nrm, mask)
    pipes = []
    details = []
    for K in Ks:
        o, d1, d2_, ref = _pipe_top2_batched(pos, nrm_o, obs_valid, obs_idx,
                                             nrm, mask, K, knn, min_frac, top2, iters)
        pipes.append(o)
        details.append((d1, d2_, ref))
    ref_idx = Ks.index(4) if 4 in Ks else 0
    ref = pipes[ref_idx]
    hid = (~mask.bool()).unsqueeze(-1)
    acc = torch.zeros_like(ref)
    for p in pipes:
        sgn = torch.sign((p * hid * ref * hid).sum(-1, keepdims=True) + 1e-12)
        acc = acc + p * sgn
    v = unit(acc)
    out = torch.where(mask.bool().unsqueeze(-1), nrm, v)
    return out.float()


def geo_prior_candidates_gpu(pos, nrm, mask, Ks=(3, 4, 5, 6), knn=24,
                             min_frac=0.8, top2=0.6, iters=60, fast=False):
    """每点候选裂隙组方向集 [B,L,C,3] (供指针头做候选加权), C=2*len(Ks)+1。

    候选 = 每个 K 的 primary 组方向 + secondary 组方向 + 最近观测方向。
    与 geo_prior_dirs_gpu 同源 (同 _pipe_top2 算法), 故指针头起手
    (tanh 门控=0) 时 pred==gp 数学成立 => 34.68° 地板保证不被破坏。

    fast=True: 仅降 iters/knn 提速 (保留 Ks 不变, 以保证 C 在训练/评测期一致);
    评测请用 fast=False 与 geo_prior_dirs_gpu 保持一致。
    """
    if fast:
        knn = 16
        iters = 12
    pos = pos.double(); nrm = nrm.double()
    B, L, _ = pos.shape
    dev = pos.device
    nrm_o, obs_valid, obs_idx = _observed_padded(nrm, mask)
    cands = []
    ref_b = None
    for K in Ks:
        o, d1, d2_, ref = _pipe_top2_batched(pos, nrm_o, obs_valid, obs_idx,
                                             nrm, mask, K, knn, min_frac, top2, iters)
        cands.append(d1)
        cands.append(d2_)
        ref_b = ref
    cands.append(ref_b)
    cand = torch.stack(cands, dim=2)                               # [B,L,C,3]
    return cand.float()


def geo_prior_obsneighbor_candidates_gpu(pos, nrm, mask, k=8, gp=None, fast=False):
    """每点 k 个最近观测法向, 作为指针的*原始观测候选* [B,L,k,3]。

    与 gp 同半球定向 (sign 相对 gp)。候选值只来自观测法向 (无泄漏)。
    指针零门控时 pred==gp 仍成立。fast 不影响候选 (kNN 本身极快)。
    """
    pos = pos.double(); nrm = nrm.double()
    B, L, _ = pos.shape
    dev = pos.device
    if gp is None:
        gp = geo_prior_dirs_gpu(pos, nrm, mask, fast=fast).detach()
    gp = gp.double()  # 与上面 double 计算对齐, 避免 dtype 不匹配
    nrm_o, obs_valid, obs_idx = _observed_padded(nrm, mask)
    pos_obs = pos.gather(1, obs_idx.unsqueeze(-1).expand(B, obs_idx.shape[1], 3))  # [B,nmax,3]
    d2 = ((pos.unsqueeze(2) - pos_obs.unsqueeze(1)) ** 2).sum(-1)  # [B,L,nmax]
    d2 = torch.where(obs_valid.unsqueeze(1), d2, torch.tensor(1e9, device=dev))
    kn = min(k, int(obs_valid.sum(1).max().item()))
    kk = d2.argsort(2)[:, :, :kn]                                 # [B,L,kn]
    # g_obs 同型 gather: 把 nrm_o 展开到 [B,L,nmax,3] 再沿 dim=2 用 kk 收集
    nrm_o_exp = nrm_o.unsqueeze(1).expand(B, L, nrm_o.shape[1], 3)  # [B,L,nmax,3]
    nb = nrm_o_exp.gather(2, kk.unsqueeze(-1).expand(B, L, kn, 3))   # [B,L,kn,3]
    sgn = torch.sign((nb * gp.unsqueeze(2)).sum(-1, keepdims=True) + 1e-12)
    nb = nb * sgn
    if kn < k:
        nb = torch.cat([nb, nb[:, :, -1:].repeat(1, 1, k - kn, 1)], 2)
    return nb.float()
