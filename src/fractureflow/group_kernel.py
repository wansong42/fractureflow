# -*- coding: utf-8 -*-
"""路线 A + 可微核组内精修 (groupkernel)。

核心思想: 在 set_ids 锁定的裂隙组内做可微 Weiszfeld 精修。

- 权重 w_ij 只在 set_ids 组内计算, 不跨组: 对隐伏点 i, 只考虑同组观测点 j
  (set_ids[j] == set_ids[i]), w_ij = exp(-d_ij / h_i) + c_i。
- 解码头初值 = set_aware 组模态 μ_k (观测法向 sign_align 均值), 加速收敛。
- 特征从「全局邻域」改成「组内邻域」: 组内观测密度、组内法向一致性 (球面方差)、
  到组质心归一化距离、到组质心方向与 μ_k 夹角。h_i,c_i 由 KernelNet 学。
- 直接对隐伏点 acos|cos| 求梯度, 训练只能往下压 set_aware 基线。

复用 learned_kernel.KernelNet / weiszfeld_decode。不依赖 l1learn 的 clamp/lr/init 开关。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .learned_kernel import KernelNet, weiszfeld_decode


# ----------------------- 组模态 μ_k (numpy, 几何基线) -----------------------
def group_modes_from_ids(nrm, mask, set_ids, kmax):
    """观测法向 sign_align 组均值 (对齐 inference._group_dirs_from_ids)。

    nrm[L,3] float, mask[L] bool (观测), set_ids[L] int (-1=无组)。
    返回 mus[K,3] 单位向量; 空组返回零向量。
    """
    L = nrm.shape[0]
    mus = []
    for k in range(kmax):
        j = (set_ids == k) & mask
        if j.sum() == 0:
            mus.append(np.zeros(3, dtype=np.float32))
            continue
        nj = nrm[j]
        mu = nj.mean(0)
        mu_n = np.linalg.norm(mu)
        if mu_n < 1e-8:
            mus.append(np.zeros(3, dtype=np.float32))
        else:
            mus.append(mu / mu_n)
    return np.stack(mus).astype(np.float32)


import numpy as np


# ----------------------- 组内特征 (torch) -----------------------
def group_features(pos, nrm, mask, set_ids, kmax, group_mu_t, radius):
    """组内邻域特征 [B,L,F]。

    每个隐伏点 i 的候选 = 同组观测 j。特征:
      f0 组内观测密度: 同组观测点数 / L  (log1p 压缩)
      f1 组内法向一致性 (球面方差): 1 - ||sign_align_mean(同组观测 nrm)||
          0=完全一致, 1=混乱 (组内离散大 -> c_i 应大)
      f2 到组质心归一化距离: ||p_i - p_center_k|| / radius  (远 -> h_i 应大)
      f3 到组质心方向与 μ_k 夹角余弦: <unit(p_i - p_center_k), μ_k>
          (该点是否落在组的「主轴平面」上, 角距小->贴组)
    另附 pos_local[3], s1_local[3], s3_local[3] 供核对组优势方向。
    """
    B, L, _ = pos.shape
    device = pos.device
    obs = mask > 0.5

    # 组质心 (组内观测位置均值) [B,K,3]
    pcenter = torch.zeros(B, kmax, 3, device=device)
    cnt = torch.zeros(B, kmax, 1, device=device)
    sid_exp = set_ids.unsqueeze(-1)                      # [B,L,1]
    obs_f = obs.float().unsqueeze(-1)                    # [B,L,1]
    for k in range(kmax):
        sel = (sid_exp == k).float() * obs_f            # [B,L,1]
        pcenter[:, k] = (pos * sel).sum(1) / sel.sum(1).clamp_min(1e-6)
        cnt[:, k] = sel.sum(1)

    feats = []
    for i in range(B):
        si = set_ids[i]                                  # [L]
        ki = si.clamp_min(0)
        mu_i = group_mu_t[i][ki]                        # [L,3] 每点所属组模态
        pc_i = pcenter[i][ki]                           # [L,3] 每点所属组质心
        # 组内观测 mask [L,L]
        same = (si.unsqueeze(0) == si.unsqueeze(1)) & obs[i].unsqueeze(0)  # [L,L]
        # 密度
        dens = torch.log1p(same.sum(1).float()) / math.log(L + 1)         # [L]
        # 组内观测法向 sign_align 均值 (einsum 避免 matmul 维度歧义)
        nj = nrm[i] * obs[i].unsqueeze(-1).float()       # 隐伏=0, [L,3]
        w = same.float()                                # [L,L] 等权
        wsum = w.sum(1, keepdim=True).clamp_min(1e-6)   # [L,1]
        gmean = torch.einsum("ij,jk->ik", w, nj) / wsum  # [L,3]
        consistency = 1.0 - gmean.norm(dim=-1)          # 球面方差, 0=一致 [L]
        # 到组质心距离
        dp = pos[i] - pc_i                              # [L,3]
        dist_c = dp.norm(dim=-1) / radius[i]            # [L]
        # 到组质心方向与 μ_k 夹角余弦
        dp_u = dp / dp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ang = (dp_u * mu_i).sum(-1)                     # [L] cos
        feats.append(torch.stack([dens, consistency, dist_c, ang], -1))
    fgroup = torch.stack(feats, 0)                      # [B,L,4]

    # 附 pos_local / s1_local / s3_local 供 KernelNet 参考组优势方向
    center = pos.mean(1, keepdim=True)                  # [B,1,3]
    pos_local = (pos - center) / radius.view(B, 1, 1)   # [B,L,3]
    # s1/s3 由 forward 传入, 这里只组 4 维; s1_local/s3_local 在 forward 里拼
    return fgroup, pos_local


# ----------------------- 组内核权重 -----------------------
def group_weights(pos, h, c, mask, set_ids, kmax):
    """组内权重 [B,L,L]。

    w_ij = exp(-d_ij / h_i) + c_i, 但仅当 set_ids[j]==set_ids[i] 且 j 为观测点。
    异组 / 隐伏 j 权重置零。退化 (某隐伏点无同组观测) 时回退: 全量观测等权 (c_i 主导)。
    """
    B = pos.shape[0]
    radius = pos.std(1).mean(-1).clamp_min(1e-3)        # [B]
    diff = pos.unsqueeze(1) - pos.unsqueeze(2)          # [B,L,L,3]
    d = torch.sqrt(diff.pow(2).sum(-1) + 1e-6) / radius.view(B, 1, 1)  # [B,L,L]
    W = torch.exp(-d / h.unsqueeze(-1)) + c.unsqueeze(-1)             # [B,L,L]

    same_group = (set_ids.unsqueeze(1) == set_ids.unsqueeze(2)) & \
        (set_ids.unsqueeze(1) >= 0)                     # 排除无组 (-1)
    same_group = same_group & (mask.unsqueeze(1) > 0.5)  # 仅观测 j 贡献
    W = W * same_group.float()

    # 退化回退: 隐伏点 i 若无同组观测, 用全量观测等权 (避免 W 全零 -> NaN)
    has_group_obs = same_group.any(2)                   # [B,L] 隐伏点是否有同组观测
    hid = (~(mask > 0.5))
    fallback = (mask.unsqueeze(1) > 0.5).float() * c.unsqueeze(-1)  # 全量观测 + c
    use_fallback = hid.unsqueeze(-1) & (~has_group_obs).unsqueeze(-1)
    W = torch.where(use_fallback, fallback, W)
    return W


# ----------------------- 顶层模型 -----------------------
class GroupKernelModel(nn.Module):
    """组内可微核精修。

    KernelNet 输入 = 组内特征(4) + pos_local(3) + s1_local(3) + s3_local(3) = 13 维。
    h_i,c_i 经组内权重 W 回传梯度 -> 直接优化隐伏点 acos|cos|。
    """

    def __init__(self, k_knn=16, hidden=32, iters=40, in_dim=13,
                 clamp_h=(0.2, 1.0), clamp_c=(0.02, 0.3), random_init=True,
                 decode_eps=0.05, decode_gn_eps=1e-3):
        super().__init__()
        self.k = k_knn
        self.iters = iters
        self.decode_eps = decode_eps
        self.decode_gn_eps = decode_gn_eps
        self.kernel = KernelNet(in_dim=in_dim, hidden=hidden,
                                clamp_h=clamp_h, clamp_c=clamp_c,
                                random_init=random_init)

    def _kmax(self, set_ids):
        m = int(set_ids.max().item()) if set_ids.numel() else -1
        return max(m + 1, 1)

    def forward(self, pos, nrm, mask, s1, s3, log_len, lith, set_ids):
        B, L, _ = pos.shape
        kmax = self._kmax(set_ids)
        radius = pos.std(1).mean(-1).clamp_min(1e-3)

        # 组模态 μ_k (numpy 几何, 不追踪梯度) -> [B,K,3] -> [B,L,3] (每点所属组)
        sid_np = set_ids.detach().cpu().numpy().astype(np.int64)
        nrm_np = nrm.detach().cpu().numpy().astype(np.float32)
        mask_np = (mask.detach().cpu().numpy() > 0.5)
        mu_list = []
        for i in range(B):
            mu_k = group_modes_from_ids(nrm_np[i], mask_np[i], sid_np[i], kmax)
            mu_list.append(torch.as_tensor(mu_k, device=pos.device))
        group_mu_t = torch.stack(mu_list, 0)            # [B,K,3]
        mu_per_pt = group_mu_t[torch.arange(B).unsqueeze(-1), set_ids.clamp_min(0)]  # [B,L,3]

        # 组内特征
        fgroup, pos_local = group_features(pos, nrm, mask, set_ids, kmax, group_mu_t, radius)
        s1_local = s1.unsqueeze(1).expand(B, L, 3)
        s3_local = s3.unsqueeze(1).expand(B, L, 3)
        feat = torch.cat([fgroup, pos_local, s1_local, s3_local], -1)  # [B,L,13]

        h, c = self.kernel(feat)                        # [B,L]
        W = group_weights(pos, h, c, mask, set_ids, kmax)

        # 软化 Weiszfeld 奇点: 批量下组内法向近抵消使 G 范数→0 (1/Gn² 爆) 与
        # C≈±1 (1/sin³ 爆); 较软的 eps + 更大 gn_eps 限制单次放大因子, 减少
        # 多迭代梯度累积溢出。对 l1learn 等价性无影响 (l1learn 用默认 eps=1e-3)。
        pred = weiszfeld_decode(pos, nrm, mask, W, iters=self.iters,
                                v_init=mu_per_pt, eps=self.decode_eps,
                                gn_eps=self.decode_gn_eps)
        return pred, h, c

    def loss(self, pos, nrm, mask, s1, s3, log_len, lith, nrm_full, set_ids):
        pred, h, c = self.forward(pos, nrm, mask, s1, s3, log_len, lith, set_ids)
        hid = (~(mask > 0.5))
        # cos 夹到 [-1+1e-4, 1-1e-4]: acos 在 cos=±1 处梯度 1/sqrt(1-cos²)→inf,
        # 而组模态初值本就接近真值 (cos≈1) 会触发 NaN。留 1e-4 余量压住梯度上限。
        cos = (pred * nrm_full).sum(-1).abs().clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        err = torch.rad2deg(torch.acos(cos))            # [B,L]
        err_hid = err[hid]
        mae = err_hid.mean() if err_hid.numel() else torch.tensor(0.0, device=pos.device)
        return mae, dict(h_mean=h.mean().item(), c_mean=c.mean().item(),
                         h_min=h.min().item(), c_max=c.max().item())


# ----------------------- 推理接口 (对齐 learned_l1_dirs) -----------------------
def groupkernel_dirs(pos, nrm_raw, obs_mask, set_ids, s1=None, s3=None,
                     log_len=None, lith=None, device="cpu", model=None,
                     kmax=None, iters=40):
    """单网络推理。返回 (dirs[L,3] float32, labels[L] int=-1)。

    model: 训练好的 GroupKernelModel (GPU/CPU)。未提供则退化为 set_aware 组模态 (几何基线)。
    """
    pos = np.asarray(pos, dtype=np.float32)
    nrm_raw = np.asarray(nrm_raw, dtype=np.float32)
    mask = np.asarray(obs_mask, dtype=bool).astype(np.float32)
    set_ids = np.asarray(set_ids, dtype=np.int64)
    L = pos.shape[0]

    if model is None:
        # 几何基线: 直接返回 set_aware 组模态 (无需 torch)
        if kmax is None:
            kmax = int(set_ids.max()) + 1 if set_ids.size else 0
        mu = group_modes_from_ids(nrm_raw, mask > 0.5, set_ids, kmax)
        dirs = np.zeros((L, 3), dtype=np.float32)
        for i in range(L):
            if not (mask[i] > 0.5) and set_ids[i] >= 0:
                dirs[i] = mu[set_ids[i]]
            elif mask[i] > 0.5:
                dirs[i] = nrm_raw[i]
        return dirs.astype(np.float32), np.full(L, -1, dtype=int)

    pos_t = torch.as_tensor(pos).unsqueeze(0).to(device)
    nrm_t = torch.as_tensor(nrm_raw * mask[:, None]).unsqueeze(0).to(device)
    mask_t = torch.as_tensor(mask).unsqueeze(0).to(device)
    set_ids_t = torch.as_tensor(set_ids).unsqueeze(0).to(device)
    s1_t = torch.as_tensor(np.asarray(s1, dtype=np.float32)).unsqueeze(0).to(device)
    s3_t = torch.as_tensor(np.asarray(s3, dtype=np.float32)).unsqueeze(0).to(device)
    log_len_t = torch.as_tensor(np.asarray(log_len, dtype=np.float32)).unsqueeze(0).to(device)
    lith_t = torch.as_tensor(np.asarray(lith, dtype=np.int64)).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred, _, _ = model(pos_t, nrm_t, mask_t, s1_t, s3_t, log_len_t, lith_t, set_ids_t)
    dirs = pred[0].cpu().numpy().astype(np.float32)
    labels = np.full(L, -1, dtype=int)
    return dirs, labels
