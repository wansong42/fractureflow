# -*- coding: utf-8 -*-
"""E3GTAssign 骨干 (注册为 "e3gt_assign")。

复用 E3GTV4 等变向量传播主干, 新增 **混合方向赋值头**, 直击"指派瓶颈":

思路
----
把隐伏点法向预测重写成 **K 个全局原型的混合**:
  - 原型 p_k: K 个可学习方向 (在每点局部标架内, 固定参数)。网络旋转 R 时
    evec -> R·evec, 故全局组方向 set_dir_k = evec @ p_k 随之旋转 => **等变**。
  - 逐点隶属 logits: logits_i = (v_i · p_k) + ctx_i  (v_i 与 p_k 同处局部标架,
    点积为不变量 => 隶属权重 w_ik 在旋转下不变)。
  - 预测方向 pred_assign_i = normalize( Σ_k w_ik · set_dir_k_global_i )。
    (= 不变量权重 × 协变方向 之和, 整体等变, 配合 |cos| 度量对增广一致)。

监督 (协同)
-----------
  - dir 损失: 把 pred_assign 拉向真值法向 (逐点去噪, 安全网)。
  - memb 损失 (仅合成有 set_ids): 让每个点分到真值所属组 -> 原型 p_k 被拉到
    真实组方向, 形成"少数连贯组方向"的混合, 而非逐点散乱回归。
  - set_dir 损失: pred_assign 对齐其真值组方向 (额外聚合力)。

为何能攻指派瓶颈
----------------
E3GTV4 连续回归 (39°) 劣于几何赋值基线 top2_ens (31°), 说明逐点回归没学会
"一个点属于哪个组"。本模型把输出**显式约束到 K 个全局连贯方向** —— 等价于
让网络先学会"指派", 再报方向, 正对应天花板分解里最大的瓶颈项。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import BACKBONES
from ..e3gt_v4 import E3GTV4, unit
from ..geometry import knn_graph, local_frames, build_graph_features


def fibonacci_sphere(K):
    """K 个近似均匀分布在单位球上的方向 (初始化原型)。"""
    idx = torch.arange(K, dtype=torch.float32)
    phi = torch.acos(1.0 - 2.0 * (idx + 0.5) / K)
    theta = torch.pi * (1.0 + 5.0 ** 0.5) * idx
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack([x, y, z], -1)


@BACKBONES.register("e3gt_assign")
class E3GTAssign(E3GTV4):
    def __init__(self, d_model=256, k_knn=16, n_layers=4, dropout=0.1,
                 obs_drop=0.25, K=8):
        super().__init__(d_model=d_model, k_knn=k_knn, n_layers=n_layers,
                         dropout=dropout, obs_drop=obs_drop)
        self.K = K
        # 局部标架内的 K 个原型方向 (可学习)。
        self.prototypes = nn.Parameter(fibonacci_sphere(K))        # [K,3]
        # 逐点隶属 logits 的标量上下文。
        self.memb_ctx = nn.Linear(d_model + 1 + 3, K)

    def forward(self, b):
        pos = b["pos"]; nrm = b["nrm"]; mask = b["mask"]
        log_len = b["log_len"]; lith = b["lith"]; s1 = b["s1"]; s3 = b["s3"]
        B, L, _ = pos.shape

        nrm_in = self._dropped_nrm(nrm, mask)

        # 图 + 局部标架 (一次性计算, 供消息传递与节点特征共用, 避免重复 kNN/PCA)
        idx, dist = knn_graph(pos, self.k)
        idx3 = idx.unsqueeze(-1).expand(B, L, self.k, 3)
        knn_pos = torch.gather(pos.unsqueeze(1).expand(B, L, L, 3), 2, idx3)
        evec, eval_ratio = local_frames(knn_pos, self.k)            # [B,L,3,3]
        graph = (idx, dist, knn_pos, evec, eval_ratio)
        fnode, fedge, _, _ = build_graph_features(
            pos, nrm_in, log_len, lith, mask, s1, s3, self.k,
            graph=graph)                                           # [B,L,31]
        radius = pos.std(1).mean(-1, keepdim=True).clamp_min(1e-3)  # [B,1]

        nrm_u = unit(torch.where(nrm_in.norm(dim=-1, keepdim=True) > 1e-6,
                                 nrm_in, torch.zeros_like(nrm_in)))
        has_obs = (nrm_in.norm(dim=-1, keepdim=True) > 1e-6).float()
        v0 = torch.einsum("blc,blcd->bld", nrm_u, evec) * has_obs  # [B,L,3]

        s = self.lift(fnode)                                       # [B,L,d]
        v = v0                                                     # [B,L,3]

        rel = knn_pos - pos.unsqueeze(2)                           # [B,L,k,3]
        rel_local = torch.einsum("blke,blcd->blkd", rel, evec)     # [B,L,k,3]

        for layer in self.layers:
            s, v = layer(s, v, idx, dist, rel_local, evec, radius)

        nv = v.norm(dim=-1, keepdim=True)                          # [B,L,1]

        # 连续回归 (辅助, 不用于最终口径)
        n_local = self.readout(torch.cat([s, nv, v], -1))          # [B,L,3]
        n_global = torch.einsum("blcd,bld->blc", evec, n_local)    # [B,L,3]
        pred_cont = unit(n_global)

        # ---------- 混合方向赋值 ----------
        # 相似度 (局部标架内, 不变量): v·p_k
        sim = torch.einsum("bld,kd->blk", v, self.prototypes)      # [B,L,K]
        ctx = self.memb_ctx(torch.cat([s, nv, v], -1))             # [B,L,K]
        logits = sim + ctx                                          # [B,L,K] 不变
        w = F.softmax(logits, dim=-1)                              # [B,L,K] 不变
        # 全局组方向: evec @ p_k -> [B,L,K,3] (随 evec 旋转 => 等变)
        set_dir_global = torch.einsum("blcd,kd->blkc", evec, self.prototypes)
        # 加权求和 (不变权重 × 协变方向 = 协变)
        pred_assign = torch.einsum("blk,blkc->blc", w, set_dir_global)
        pred_assign = unit(pred_assign)

        aniso = (eval_ratio[:, :, 0] - eval_ratio[:, :, 2]).clamp(0.0, 1.0)  # [B,L]

        return {
            "pred": pred_assign,          # 最终口径用: 赋值感知方向
            "pred_cont": pred_cont,       # 辅助连续回归
            "memb": logits,               # 逐点隶属 logits (memb 损失)
            "set_dir_pred": pred_assign,  # 供 set_dir 损失
            "aniso": aniso, "v": v, "s": s,
        }
