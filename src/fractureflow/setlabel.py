# -*- coding: utf-8 -*-
"""路线 A: 离线组标注生成器。

给定完整法向场 (net["nrm"], 即测量得到的全部裂隙法向, 含隐伏点),
用球面 k-means 为每个点贴 set_id (0..K-1)。这是与 `lith` 同性质的
"数据属性" —— 商用落地时由分析师从完整 DFN 量测直接给出, 推理时对
观测/隐伏点**全部已知**, 不泄漏隐伏法向本身。

一旦数据携带 set_ids, 解码器 (inference.set_aware_dirs) 即达 13–15°。
详细可达性论证见 docs/可达性与数据补强方案.md。
"""

import numpy as np


def _unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def spherical_kmeans(nrm, K, iters=80, seed=0):
    """返回 (centers[K,3], assign[L])。

    注意: 裂隙法向**无向** (±n 同一裂隙面), 故指派用 |cos| 而非 cos,
    center 用组内符号对齐后的均值, 避免 ±n 互相抵消。
    """
    rng = np.random.default_rng(seed)
    pts = _unit(nrm)
    n = len(pts)
    if n == 0:
        # R80 毒丸 PP-B3: 空输入原实现静默产出 NaN 组心 (空均值 + 1e-12 归一),
        # 违反禁静默垃圾红线 — 响亮拒绝。
        raise ValueError(
            "spherical_kmeans: 空法向输入, 无可聚类数据 (R80 PP-B3 响亮拒绝)")
    K = min(K, n)
    if K <= 1:
        c = _unit(_sign_align(pts).mean(0))[None]
        return c, np.zeros(n, dtype=int)
    centers = pts[rng.choice(n, K, replace=False)].copy()
    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        cos = np.abs(pts @ centers.T)
        assign = cos.argmax(1)
        for k in range(K):
            sel = pts[assign == k]
            if len(sel):
                centers[k] = _unit(_sign_align(sel).mean(0))
    return centers, assign


def _sign_align(pts):
    """把 pts 按首向量符号对齐 (无向法向取一致符号), [M,3] -> [M,3]。"""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return pts
    ref = pts[0]
    s = np.sign((pts * ref).sum(-1, keepdims=True))
    s[s == 0] = 1
    return pts * s


def _silhouette(nrm, centers, assign):
    """基于夹角(无向 |cos|)的轮廓系数 (越大越好), 返回均值。"""
    pts = _unit(nrm)
    K = len(centers)
    abscos = np.abs(pts @ centers.T)
    a = np.zeros(len(pts))
    b = np.full(len(pts), -np.inf)
    for k in range(K):
        sel = assign == k
        if sel.sum() <= 1:
            a[sel] = 0.0
            continue
        a[sel] = 1.0 - abscos[sel, k]
    others = [k for k in range(K) if (assign == k).sum() > 1]
    for k in range(K):
        sel = assign == k
        if not sel.any() or len(others) <= 1:
            b[sel] = 0.0
            continue
        oc = np.full(len(pts), -np.inf)
        for k2 in others:
            if k2 == k:
                continue
            oc = np.maximum(oc, abscos[:, k2])
        b[sel] = 1.0 - oc[sel]
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-9)
    return float(np.nanmean(s))


def generate_set_ids(nrm, Krange=(2, 7), seed=0, min_pts=4):
    """为完整法向场生成 set_id 标签。

    nrm: [L,3] 全部裂隙法向 (含隐伏)。
    Krange: 候选 K 范围 (含端点)。
    min_pts: 网络点数下限, 不足则退化单组 (全 0)。

    返回 int 数组 [L], 取值 0..K-1。
    """
    nrm = np.asarray(nrm, dtype=np.float64)
    L = nrm.shape[0]
    if L < min_pts or L <= Krange[0]:
        return np.zeros(L, dtype=int)

    best_K, best_s, best_assign = Krange[0], -np.inf, None
    for K in range(Krange[0], Krange[1] + 1):
        if K >= L:
            break
        c, a = spherical_kmeans(nrm, K, seed=seed)
        s = _silhouette(nrm, c, a)
        if s > best_s:
            best_s, best_K, best_assign = s, K, a
    if best_assign is None:
        c, best_assign = spherical_kmeans(nrm, Krange[0], seed=seed)
    return best_assign.astype(int)


def joint_set_ids(wells_nrm, K, seed=0, anchors=None, iters=80):
    """多井联合分组 —— 路线 A 深化动作①: 跨井共享组系全局聚类.

    前提: 同区块多井共享同一应力场 -> 组系方向一致 (中国区块天然满足).
    做法: 把多口井法向池化成"超级井"跑一次球面 k-means, 得跨井共享组心 (global
    centers), 再给每口井各自指派 set_ids. 直接对抗 FORGE 阴性根因 (单井点少 ->
    组心噪声大): 组心由全区块点估计更稳.

    同时输出"组系跨井稳定性"增值分析 (每井每组离散度 + 组出现表).

    wells_nrm: list[np.ndarray[L_i,3]] 各井法向 (无向).
    K: 组数 (地质师给定 / K网格扫 / silhouette 选; 此处显式传入).
    anchors: 可选, 地质师给的代表方向 [A,3] (作 kmeans 初始化锚点, 动作②).
    返回 dict:
      'centers' [K,3] 跨井共享组心
      'assign'  list[np.ndarray[L_i]] 每井 set_ids (0..K-1)
      'stability' list[dict] 每井 {'n', 'disp':{k:角离散均值 or None}}
    """
    rng = np.random.default_rng(seed)
    pooled = np.vstack([_unit(np.asarray(w, float)) for w in wells_nrm])
    K = min(K, len(pooled))
    # 初始化: 锚点优先, 不足随机补
    centers = np.zeros((K, 3))
    if anchors is not None and len(anchors) > 0:
        anc = _unit(np.asarray(anchors, float))
        na = min(len(anc), K)
        centers[:na] = anc[:na]
        if na < K:
            centers[na:] = pooled[rng.choice(len(pooled), K - na, replace=False)]
    else:
        centers = pooled[rng.choice(len(pooled), K, replace=False)].copy()
    # 球面 k-means (无向 |cos| 指派)
    assign_pool = np.zeros(len(pooled), dtype=int)
    for _ in range(iters):
        cos = np.abs(pooled @ centers.T)
        assign_pool = cos.argmax(1)
        for k in range(K):
            sel = pooled[assign_pool == k]
            if len(sel):
                centers[k] = _unit(_sign_align(sel).mean(0))
    # 每井指派 + 跨井稳定性
    assign, stability = [], []
    off = 0
    for w in wells_nrm:
        Lw = len(w)
        aw = assign_pool[off:off + Lw]
        assign.append(aw)
        disp = {}
        wu = _unit(np.asarray(w, float))
        for k in range(K):
            sel = wu[aw == k]
            if len(sel) >= 2:
                ang = np.rad2deg(np.arccos(np.clip(np.abs(sel @ centers[k]), -1.0, 1.0)))
                disp[k] = float(ang.mean())
            else:
                disp[k] = None
        stability.append({"n": Lw, "disp": disp})
        off += Lw
    return {"centers": centers, "assign": assign, "stability": stability}


def joint_set_ids_grid(wells_nrm, Krange=(2, 12), seed=0, anchors=None):
    """K 网格扫 (动作③): 在 Krange 上跑 joint_set_ids, 用跨井池化 silhouette 选最优 K.

    返回 (best_K, best_assigns, best_centers, scores).
    scores: dict K->跨井池化 silhouette (均值).
    """
    pooled = np.vstack([_unit(np.asarray(w, float)) for w in wells_nrm])
    scores = {}
    best = None
    for K in range(Krange[0], Krange[1] + 1):
        if K >= len(pooled):
            break
        res = joint_set_ids(wells_nrm, K, seed=seed, anchors=anchors)
        # 池化 silhouette
        abscos = np.abs(pooled @ res["centers"].T)
        assign_pool = abscos.argmax(1)
        s = _silhouette(pooled, res["centers"], assign_pool)
        scores[K] = s
        if best is None or s > best[0]:
            best = (s, K, res)
    # R7 修复: 极小输入 (pooled 点数 < Krange[0]) 时循环不执行, best 仍为 None.
    # 显式守卫, 返回空结果而非 None 解引用崩.
    if best is None:
        K0 = min(Krange[0], max(len(pooled), 1))
        empty_assign = [np.zeros(len(w), dtype=int) for w in wells_nrm]
        empty_centers = np.zeros((K0, 3))
        return K0, empty_assign, empty_centers, scores
    return best[1], best[2]["assign"], best[2]["centers"], scores


def label_nets(nets, Krange=(2, 7), seed=42):
    """就地给 net dict 列表加 'set_ids' (numpy int) 字段。返回新列表。"""
    out = []
    for net in nets:
        n = dict(net)
        nrm = np.asarray(net["nrm"], dtype=np.float64)
        n["set_ids"] = generate_set_ids(nrm, Krange=Krange, seed=seed)
        out.append(n)
    return out


def _rand_rot(rng):
    """随机三维旋转矩阵 (QR 分解法)。"""
    M = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    return Q


def spatial_block_mask(pos, frac=0.4, rng=None, rotate=True):
    """选一个空间子块作为"已调查区域" (no-leak 传播姿势)。

    随机旋转点云后对某一坐标取 slab, 使约 `frac` 的点落在块内作为观测点,
    其余为隐伏点。模拟"只勘察了场地某区块"的真实情形, 与均匀随机掩码不同:
    观测点在空间上连续成块, 组在块内可能与全局混合。

    返回 bool mask (True=观测/已调查, False=隐伏)。
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    pos = np.asarray(pos, dtype=np.float64)
    q = pos @ _rand_rot(rng).T if rotate else pos
    x = q[:, 0]
    thr = float(np.quantile(x, 1 - frac))
    return x >= thr


def obs_only_set_ids(nrm_full, obs_mask, Krange=(2, 7), seed=42, select="sil"):
    """仅用观测法向 (no-leak) 估计组并给所有点打标。

    步骤: 在观测法向上跑球面 k-means (轮廓/BIC 选 K) -> 得组心 ->
    把所有点 (含隐伏) 指派的组心 -> 返回全量 set_ids (int[L])。
    推理时只用 obs 估计组心, 不泄漏隐伏法向本身。
    """
    from .setlabel_opt import (
        spherical_kmeans_restarts, _silhouette, _vmf_bic, _unit)
    nrm = _unit(np.asarray(nrm_full, dtype=np.float64))
    obs = np.asarray(obs_mask, dtype=bool)
    on = nrm[obs]
    L = nrm.shape[0]
    if on.shape[0] < Krange[0]:
        return np.zeros(L, dtype=int), 1, 0.0
    best = None  # (score, K, centers)
    for K in range(Krange[0], min(Krange[1], on.shape[0]) + 1):
        c, a, ine = spherical_kmeans_restarts(on, K, 4, 80, seed=seed + K)
        sil = _silhouette(on, c, a)
        score = sil if select == "sil" else _vmf_bic(on, c, a)[1]
        if best is None or score > best[0]:
            best = (score, K, c)
    K, centers = best[1], best[2]
    assign = np.abs(nrm @ centers.T).argmax(1)
    return assign.astype(int), K, float(best[0])
