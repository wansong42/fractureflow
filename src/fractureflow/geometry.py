# -*- coding: utf-8 -*-
"""M2 几何特征编码: 显式局部参考框架 (kNN -> PCA), 无任何等变库。

将 (pos, nrm, len, lith, obs_mask, s1, s3) 变成逐点不变特征 (节点)
与邻域相对特征 (边), 供 FieldEncoder 图消息传递使用。
"""

import torch

# ---------------------------------------------------------------------------
# 训练帧策略护栏 (T82 后续, 十四期): 新训练必须显式确认标架策略.
# ---------------------------------------------------------------------------

_TRAINING_FRAME_ACK = False
_DEFAULT_LEGACY = True   # 模块级默认帧: True=冻结行为 (legacy); ack('fixed') 后翻 False


def ack_training_frame_policy(policy: str) -> None:
    """训练入口护栏: 新训练必须显式选择 fixed 标架后方可开跑.

    背景: local_frames 默认 legacy=True 是八期冻结行为 —— 已知**非等变**
    (BUG-2: 帧实为特征向量置换混合, 旋转输入下逐点偏差可达 76°), 仅允许
    用于冻结 checkpoint 复现与漂移对照。任何新训练若忘记传 legacy=False
    会静默用非等变帧, 本断言把该事故挡在训练入口。

    行为语义 (2026-08-28 修复"护栏空转"): 确认 'fixed' 后, 模块级默认帧
    切换为修正版 —— 之后所有未显式传 legacy 的 local_frames 调用 (含模型
    内部调用点) 都走几何正确标架。冻结 checkpoint 复现/评测路径从不调用
    本函数, 默认仍为 legacy, 锚点数字零漂移。显式传 legacy=True/False
    的调用点 (selfcheck S27 / t82 漂移闸门 / 单测) 不受影响。
    """
    global _TRAINING_FRAME_ACK, _DEFAULT_LEGACY
    if policy == "fixed":
        _TRAINING_FRAME_ACK = True
        _DEFAULT_LEGACY = False
    elif policy == "legacy":
        raise RuntimeError(
            "训练帧策略拒绝 'legacy': legacy 标架非等变 (BUG-2), 只允许冻结 "
            "checkpoint 复现, 禁止新训练。如确需复现实验请走评测路径, 不经此处。")
    else:
        raise ValueError(f"未知帧策略 {policy!r} (仅接受 'fixed')")


def training_frame_acked() -> bool:
    return _TRAINING_FRAME_ACK


def unit_n(x, eps=1e-7):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def knn_graph(pos, k):
    """逐样本 kNN: pos [B,L,3] -> idx [B,L,k], dist [B,L,k]"""
    d2 = (pos.unsqueeze(1) - pos.unsqueeze(2)).pow(2).sum(-1)  # [B,L,L]
    eye = torch.eye(d2.shape[1], device=pos.device)
    d2 = d2 - eye.unsqueeze(0) * 1e6
    dist2, idx = torch.topk(d2, k, dim=-1, largest=False)
    return idx, (dist2.clamp_min(0) + 1e-10).sqrt()


def _local_frames_legacy(knn_pos, k, deterministic_e2=False):
    """八期冻结行为存档 (T82 前)。已知非等变, 仅供冻结 checkpoint 复现与漂移对照。

    缺陷 (BUG-2 真因, 2026-08 定位):
      1. eigh 输出布局为 [B,L,分量,特征向量] (列=特征向量), 旧代码
         `v[:, :, idx]` 重排的是**分量轴**, 相当于对每个特征向量的坐标做置换;
      2. 后续 `v[:, :, k]` 读出的是"第 k 个分量跨三个特征向量", 并非任何
         特征向量 —— 标架是几何上无意义的混合;
      3. 实测旋转输入后逐点帧偏差可达 76° (等变性完全破坏),
         所有依赖它的模型 (e3gt_v4/hybrid/hybrid_v2) 的"严格等变"声明不成立。
    """
    B, L, k, C = knn_pos.shape
    mean = knn_pos.mean(2, keepdim=True)
    x = knn_pos - mean
    cov = torch.einsum("blkc,blkd->blcd", x, x) / max(int(k), 2)
    e, v = torch.linalg.eigh(cov)
    idx = torch.tensor([2, 1, 0], device=x.device)
    e = e[:, :, idx]
    v = v[:, :, idx]                                    # ← 错排在分量维
    far = knn_pos[:, :, -1] - mean[:, :, 0]
    flip = (torch.einsum("ble,ble->bl", v[:, :, 0], far) < 0).float()
    v = v * (1.0 - 2.0 * flip[:, :, None, None]).to(x.dtype)
    if deterministic_e2:
        proj_e2 = torch.einsum("blke,ble->blk", x, v[:, :, 1]).mean(2)
        flip_e2 = (proj_e2 < 0).float()
        v[:, :, 1] = v[:, :, 1] * (1.0 - 2.0 * flip_e2[:, :, None]).to(x.dtype)
    v = unit_n(v)
    v3 = torch.linalg.cross(v[:, :, 0], v[:, :, 1])
    v = torch.stack([v[:, :, 0], v[:, :, 1], v3], dim=2)
    trace = e.sum(-1, keepdim=True) + 1e-8
    return v, e / trace


def local_frames(knn_pos, k, deterministic_e2=False, legacy=None):
    """每点 kNN 邻域 PCA: 返回 (e [B,L,3,3], 归一化特征值 [B,L,3] 降序).

    ⚠️ 默认帧策略: legacy=None (省略) 时跟随模块级默认 —— 初始为 legacy
    (八期冻结行为, 漂移闸门实测 FAIL: e3gt_v4 +6.86° / hybrid −0.86°),
    `ack_training_frame_policy('fixed')` 之后自动切换为修正版。
    冻结 checkpoint 复现路径不调用 ack, 行为不变; **显式传 legacy=True/False
    的调用点永远不受全局状态影响**。

    修正版输出布局: v[b,l,c,d] = 第 d 个标架向量的第 c 个分量 (列=标架向量),
    与下游 `einsum("blc,blcd->bld", x, v)` 投影约定一致; 符号约定: e1 钉到
    最远邻点方向; e3 = e1 × e2 保证右手系。

    Args:
        knn_pos: [B,L,k,3] 邻域点坐标 (未中心化).
        k: 邻域大小 (用于协方差归一化除数).
        deterministic_e2: 钉住 e2 符号 (邻域投影均值), 使带符号特征跨运行确定.
        legacy: None=跟随模块默认 (初始 legacy, ack 后 fixed);
            True=冻结行为 (_local_frames_legacy); False=T82 几何正确实现.
    """
    if legacy is None:
        legacy = _DEFAULT_LEGACY
    if legacy:
        return _local_frames_legacy(knn_pos, k, deterministic_e2)
    B, L, k, C = knn_pos.shape
    mean = knn_pos.mean(2, keepdim=True)
    x = knn_pos - mean
    cov = torch.einsum("blkc,blkd->blcd", x, x) / max(int(k), 2)
    e, v = torch.linalg.eigh(cov)                       # 升序, v[b,l,c,j] 列=特征向量
    idx = torch.tensor([2, 1, 0], device=x.device)
    e = e[:, :, idx]
    v = v[:, :, :, idx]                                 # 重排特征向量编号 (降序)
    far = knn_pos[:, :, -1] - mean[:, :, 0]
    # e1 = 最大主方向, 符号钉到最远邻点
    flip1 = (torch.einsum("ble,ble->bl", v[:, :, :, 0], far) < 0).float()
    v[:, :, :, 0] = v[:, :, :, 0] * (1.0 - 2.0 * flip1[:, :, None]).to(x.dtype)
    if deterministic_e2:
        proj_e2 = torch.einsum("blke,ble->blk", x, v[:, :, :, 1]).mean(2)
        flip2 = (proj_e2 < 0).float()
        v[:, :, :, 1] = v[:, :, :, 1] * (1.0 - 2.0 * flip2[:, :, None]).to(x.dtype)
    v = unit_n(v)
    e1, e2 = v[:, :, :, 0], v[:, :, :, 1]
    e3 = torch.linalg.cross(e1, e2)
    v = torch.stack([e1, e2, e3], dim=3)                # [B,L,comp,vec] 列=标架向量
    trace = e.sum(-1, keepdim=True) + 1e-8
    return v, e / trace


def build_graph_features(pos, nrm, lens, lith, mask, s1, s3, k, graph=None):
    """构造节点特征与边特征。

    输入:
      pos     [B,L,3]   已做平移/缩放归一化的坐标
      nrm     [B,L,3]   观测点法向 (未观测点已置零)
      lens    [B,L,1]   长度 (对数空间)
      lith    [B,L]     int 类别
      mask    [B,L]     float 0/1 观测掩码
      s1,s3   [B,3]     单位应力方向
      k       int       kNN 规模
      graph   可选, (idx, dist, knn_pos, evec, eval_ratio) 预计算图。
              提供时跳过内部 kNN/PCA, 避免 forward 与其重复计算 (约 2x 提速)。
    输出:
      fnode   [B,L,F=31] 节点特征 (布局见注释)
      fedge   [B,L,k,4]  边特征: 局部框架相对向量(3) + 归一化距离(1)
      idx     [B,L,k]    kNN 索引
      knn_pos [B,L,k,3]  邻域坐标
    F 布局: 0..2 pos_local, 3..5 nrm_local, 6 len, 7 mask, 8..10 eig, 11 aniso,
            12..14 s1, 15..17 s3, 18..20 nhb_nrm(全局), 21 obs占比, 22 最近观测距离,
            23..30 lith onehot
    """
    B, L, _ = pos.shape
    dev = pos.device

    if graph is None:
        idx, dist = knn_graph(pos, k)
        idx3 = idx.unsqueeze(-1).expand(B, L, k, 3)
        knn_pos = torch.gather(pos.unsqueeze(1).expand(B, L, L, 3), 2, idx3)
        evec, eval_ratio = local_frames(knn_pos, k)          # [B,L,3,3] [B,L,3]
    else:
        idx, dist, knn_pos, evec, eval_ratio = graph

    nrm_u = unit_n(torch.where(nrm.norm(dim=-1, keepdim=True) > 1e-6,
                               nrm, torch.zeros_like(nrm)))
    nrm_local = torch.einsum("blc,blcd->bld", nrm_u, evec)
    nrm_local = nrm_local * (nrm.norm(dim=-1, keepdim=True) > 1e-6).float()

    # 显式"观测传播"特征: kNN 内观测法向的距离衰减加权均值(全局系)
    # + 观测占比 + 最近观测距离; 快速抑制跨组混入。
    radius = pos.std(1).mean(-1, keepdim=True).clamp_min(1e-3)           # [B,1]
    n_obs = torch.gather(mask.unsqueeze(1).expand(B, L, L), 2, idx)      # [B,L,k]
    nrm_nb = torch.gather(nrm.unsqueeze(1).expand(B, L, L, 3), 2,
                          idx.unsqueeze(-1).expand(B, L, k, 3))
    w_exp = torch.exp(-(dist / radius.unsqueeze(-1)) * 3.0)              # 距离衰减
    wgt = n_obs.unsqueeze(-1) * w_exp.unsqueeze(-1)                      # [B,L,k,1]
    nrm_obs_sum = (nrm_nb * wgt).sum(2)                                  # [B,L,3]
    cnt = n_obs.sum(-1)                                                  # [B,L]
    nrm_obs_mean = unit_n(nrm_obs_sum)
    dist_obs_min = torch.min(dist + (1.0 - n_obs) * 1e4, dim=-1).values  # [B,L]
    aniso = (eval_ratio[:, :, 0] - eval_ratio[:, :, 2]).clamp(0.0, 1.0)
    lith_onehot = torch.nn.functional.one_hot(lith.clamp(0, 7), 8).float()

    centroid = pos.mean(1, keepdim=True)
    pos_local = torch.einsum("blc,blcd->bld", pos - centroid, evec)
    s1_local = torch.einsum("bc,blcd->bld", s1, evec)
    s3_local = torch.einsum("bc,blcd->bld", s3, evec)

    fnode = torch.cat([
        pos_local,                       # 3
        nrm_local,                       # 3
        lens.reshape(B, L, 1),           # 1
        mask.unsqueeze(-1),              # 1
        eval_ratio,                      # 3
        aniso.unsqueeze(-1),             # 1
        s1_local, s3_local,              # 6
        nrm_obs_mean,                    # 3
        (cnt.unsqueeze(-1) / k),         # 1
        (dist_obs_min / radius).reshape(B, L, 1),  # 1
        lith_onehot,                     # 8
    ], dim=-1)                           # 31

    rel = knn_pos - pos.unsqueeze(2)
    rel_local = torch.einsum("blke,blcd->blkd", rel, evec)   # [B,L,k,3]
    dist_n = dist / radius.unsqueeze(-1)
    fedge = torch.cat([rel_local, dist_n.unsqueeze(-1)], dim=-1)

    return fnode, fedge, idx, knn_pos


# ---------------------------------------------------------------------------
# 罗盘约定集中化 (B3-4): 唯一权威的产状 <-> 法向 <-> 走向 双向转换集
# ---------------------------------------------------------------------------
# 约定 (全库唯一标准, 其他地方手搓 atan2+sin/cos 产状组合将被 grep 门禁拦截):
#   坐标: x=东(E), y=北(N), z=上(Up)。
#   角度: dip 从水平面起算 0-90°; dip_dir 从北顺时针 0-360°。
#   单位方向向量为单位法向 n=(nx,ny,nz), 其中
#       nx = sin(dip)·sin(dip_dir)
#       ny = sin(dip)·cos(dip_dir)
#       nz = cos(dip)
#   倾向方向单位向量 (水平投影) = [sin(dip_dir), cos(dip_dir), 0]
#   走向 = 倾向 - 90°; 走向单位向量 = [sin(strike), cos(strike), 0]
#   无向法向铁律: n 与 -n 必须输出同一个 (dip, dip_dir) 对。
import numpy as _np  # noqa: E402  (本段为纯 numpy 工具, 与上方 torch 逻辑隔离)


def dip_dipdir_to_normal(dip_deg, dip_dir_deg):
    """倾角倾向 -> 单位法向 [nx, ny, nz] (numpy)。

    支持标量 / 数组; 返回与输入同形状 (...,3)。
    """
    dip = _np.radians(_np.asarray(dip_deg, float))
    dd = _np.radians(_np.asarray(dip_dir_deg, float))
    nx = _np.sin(dip) * _np.sin(dd)
    ny = _np.sin(dip) * _np.cos(dd)
    nz = _np.cos(dip)
    nrm = _np.stack([nx, ny, nz], axis=-1)
    nrm = nrm / (_np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12)
    return nrm


def normal_to_dip_dipdir(nrm):
    """单位法向 -> (dip_deg, dip_dir_deg) (numpy, 无向铁律安全版)。

    对 nz<0 的法向, 倾向角翻转 180°, 保证 n 与 -n 输出完全一致。
    """
    nrm = _np.asarray(nrm, float)
    nrm = nrm / (_np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12)
    nz = nrm[..., 2]
    dip = _np.degrees(_np.arccos(_np.clip(_np.abs(nz), 0.0, 1.0)))
    dd = _np.degrees(_np.arctan2(nrm[..., 0], nrm[..., 1])) % 360.0
    dd = _np.where(nz < 0, (dd + 180.0) % 360.0, dd)
    return dip, dd


def dip_dir_to_dip_vector(dip_dir_deg):
    """倾向方向单位向量 (水平投影) = [sin(dd), cos(dd), 0]。"""
    dd = _np.radians(_np.asarray(dip_dir_deg, float))
    return _np.stack([_np.sin(dd), _np.cos(dd), _np.zeros_like(dd, float)], axis=-1)


def dip_dir_to_strike_vector(dip_dir_deg):
    """走向单位向量 (走向 = 倾向 - 90°) = [sin(strike), cos(strike), 0]。

    BUG-A 修复锚点: 旧实现错用 [cos(strike), sin(strike)] 导致与倾向向量
    不正交 (点积 = -cos2α ≠ 0)。集中到此函数后由 B3-3 不变量锁死正交性。
    """
    dd = _np.radians(_np.asarray(dip_dir_deg, float))
    strike = dd - _np.pi / 2.0
    return _np.stack([_np.sin(strike), _np.cos(strike), _np.zeros_like(strike, float)], axis=-1)