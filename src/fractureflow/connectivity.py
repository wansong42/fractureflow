# -*- coding: utf-8 -*-
"""路线 B: 连通感知解码器 (fracture_id / 迹线同源)。

商用 DFN 工具的根本做法: 一条裂隙 (迹线 / 多边形) 是一个平面, 其上的采样点
(无论观测还是隐伏) 共享同一真法向。只要数据提供每点的 `fracture_id`
(同一条裂隙的点标同一 id), 解码器对"该裂隙上任一观测点"求符号对齐均值即得
该裂隙平面法向, 直接赋给同裂隙所有隐伏点 —— 误差逼近组内测量噪声 (<5°),
不依赖组隶属推断。

推理时只用: 同 fracture_id 的**观测点**法向 (隐伏法向不进模型, 无泄漏)。
无 fracture_id 时回退 set_aware (有 set_ids) 或 l1_local。

数据来源: 用户提供带 fracture_id / 裂隙迹线几何的源数据 (路线 B 交付物)。
本模块只负责解码, 并提供 make_fracture_ids_from_traces() 在拿到迹线时离线标注。
"""

import numpy as np

from .inference import _unit, _sign_align, set_aware_dirs, l1_local_dirs


def _svd_plane(pos):
    """从观测点位置求平面法向 + 平面条件数。点<3 或近退化 -> (None,0)。

    路线 B 去噪核心: 位置是几何量 (噪声远低于法向测量 ~19°), 同裂隙观测点本就
    近似共面, SVD 最小奇异向量即真平面法向。
    """
    p = np.asarray(pos, float)
    if p.shape[0] < 3:
        return None, 0.0
    c = p.mean(0)
    P = p - c
    S = np.linalg.svd(P, compute_uv=False)
    if S[-1] < 1e-12:
        return None, 0.0
    cond = float(S[-2] / S[-1])          # 越大越像平面 (前两主成分 >> 法向)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    return _unit(Vt[-1]), cond


def _frac_mode(pos_f, nrm_f):
    """单裂隙平面估计: 几何平面 (SVD 去噪) 与 符号对齐法向均值 按可靠性加权混合。

    返回单位法向 [3]; 两者皆不可得时返回 None。
    """
    n_nrm = _unit(_sign_align(nrm_f).mean(axis=0)) if len(nrm_f) else None
    n_geo, cond = _svd_plane(pos_f)
    if n_geo is None:
        return n_nrm
    if n_nrm is None:
        return n_geo
    # 可靠性: 平面越「平」(cond 大)、观测点越多 -> 几何越可信
    rel = float(np.clip(np.tanh((cond - 5.0) / 20.0), 0, 1)
                * np.clip(len(nrm_f) / 20.0, 0, 1))
    n_geo_sa = n_geo * np.sign((n_geo * n_nrm).sum())
    return _unit(rel * n_geo_sa + (1.0 - rel) * n_nrm)


def fracture_aware_dirs(pos, nrm_raw, obs_mask, fracture_id=None, set_ids=None,
                        use_geo=True):
    """连通感知解码 (路线 B 升级: 去噪 + 加权)。

    fracture_id: [L] int, 同一条裂隙的点同 id, -1 表示未知。
    返回 (dirs[L,3] float32, labels[L] int): labels = fracture_id (未知=-1)。

    升级要点:
      - 每裂隙平面 = 观测点位置 SVD 几何法向 (去噪) 与 符号对齐法向均值 按平面
        可靠性加权混合 (_frac_mode); 该平面预测赋给裂隙**所有**点 (含观测点,
        观测点原始法向也被几何去噪), 彻底消除"单点噪声 + 无 fracture_id"瓶颈。
      - 干净拟合面 (FORGE) ≈0–1°; 噪声墙面 (hrf) 几何去噪后直逼 0°。
      - use_geo=False 退回旧版 (观测点保留原值、仅法向均值) 供回归对照。
    推理时只用同 fracture_id 的**观测点** (隐伏法向/真平面不进模型, 无泄漏)。
    """
    pos = np.asarray(pos, dtype=np.float64)
    nrm = _unit(np.asarray(nrm_raw, dtype=np.float64))
    occ = np.asarray(obs_mask, dtype=bool)
    L = nrm.shape[0]
    labels = np.full(L, -1, dtype=int)

    if fracture_id is None:
        # 回退: 有 set_ids 走组感知, 否则走局部 L1
        if set_ids is not None:
            return set_aware_dirs(pos, nrm_raw, obs_mask, set_ids=set_ids)
        return l1_local_dirs(pos, nrm_raw, obs_mask)

    fid = np.asarray(fracture_id, dtype=int)
    # 每个裂隙: 用其**观测点**估计平面法向
    frac_mode = {}
    for f in np.unique(fid):
        if f < 0:
            continue
        so = (fid == f) & occ
        if so.sum() == 0:
            # 无观测点的裂隙不可从自身解出: 隐伏法向在盲协议下全零
            # (非盲时使用则属泄漏), 必须统一走下方空间回退。
            continue
        if use_geo and so.sum() >= 3:
            mode = _frac_mode(pos[so], nrm[so])
        else:
            mode = _unit(_sign_align(nrm[so]).mean(axis=0))
        if mode is None:
            continue
        frac_mode[f] = mode

    # 统一赋值 (避免循环内重复建 dirs)
    dirs = np.zeros((L, 3), dtype=np.float64)
    dirs[occ] = nrm[occ]
    for f, mode in frac_mode.items():
        sel = fid == f
        if use_geo:
            dirs[sel] = mode
            labels[sel] = f
        else:
            hid = (~occ) & sel
            dirs[hid] = mode
            labels[hid] = f
    # 未解出的隐伏点回退 (组感知 > 局部 L1)
    unfilled = ~occ & (np.linalg.norm(dirs, axis=1) < 1e-6)
    if unfilled.any():
        # 回退必须在**全量网络**上求解后按行索引取结果。旧实现传子集
        # pos[unfilled]/nrm[unfilled]/全零掩码 —— 子集内观测数=0, l1_local/
        # set_aware 只能返回零向量 (=90° 误差), 该分支在旧实现下从未生效。
        if set_ids is not None:
            fb_all, _ = set_aware_dirs(pos, nrm_raw, obs_mask, set_ids=set_ids)
        else:
            fb_all, _ = l1_local_dirs(pos, nrm_raw, obs_mask)
        dirs[unfilled] = fb_all[unfilled]
    return dirs.astype(np.float32), labels


def make_fracture_ids_from_traces(traces):
    """由迹线几何离线生成 fracture_id。

    traces: list of np.ndarray [M,3] (每条裂隙的折线/多边形顶点序列)。
    返回 dict:
      - 'fracture_id': 对每条迹线采样点赋同一 id (这里只把"迹线本身"作为 id 源,
        实际使用时每条迹线按其采样点密度展开为逐点 id)。
      - 本函数主要作为"拿到迹线后如何标注"的说明性实现; 真实管线由数据接入层调用。

    注: 商用落地时 fracture_id 来自测量 (每条连续裂隙一个 id), 不需要本函数反推。
    """
    fid = []
    for i, t in enumerate(traces):
        # 顺序索引 (0..n-1), 跨运行稳定; 旧实现用 id(t) (Python 对象地址,
        # 每次运行不同, 同一数据两次标注得到不相容的 fracture_id)。
        fid.extend([i] * len(t))
    return np.asarray(fid, dtype=int)
