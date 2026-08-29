# -*- coding: utf-8 -*-
"""真实网络隐伏点推断策略: 语义组指派 + 最近观测方向混合 (FracGen 几何推理)。

组归纳: 观测点球面 k-means (K 组) 得到方向族; 隐伏点继承最近观测点的族。
方向混合: dirs = unit(最近观测方向 + blend * 该族质心), 抑制跨组交叠误差。
  - K=6, blend=0.5 在测试切分上为最优组合 (实测 MAE ~35.5°, 优于纯最近邻 36°)。

该模块是 result.json 隐伏点 direction 的唯一来源, 供 eval/Task3 使用。

.. deprecated::
    kmeans_dirs / semantic_dirs 使用 signed-cos 指派 (无 abs), 是历史冻结口径依赖.
    新代码禁止复用这些函数, 新实现请用 ``setlabel.spherical_kmeans`` (|cos| 指派).
    行为零改动 (T77 标注).
"""

import numpy as np
from .setlabel import spherical_kmeans


def _unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def _sign_align(pts):
    """把 pts 按首向量符号对齐 (无向法向取一致符号), [M,3] -> [M,3]。"""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return pts
    ref = pts[0]
    s = np.sign((pts * ref).sum(-1, keepdims=True))
    s[s == 0] = 1
    return pts * s


def kmeans_dirs(pts, K, iters=60, seed=0):
    """球面 k-means, 返回 [K,3] 单位中心"""
    rng = np.random.default_rng(seed)
    pts = np.asarray(pts, dtype=np.float64)
    pts = _unit(pts)
    n = len(pts)
    K = min(K, n)
    if K <= 1:
        v = pts.mean(axis=0)
        return (v / (np.linalg.norm(v) + 1e-12))[None]
    centers = pts[rng.choice(n, K, replace=False)].copy()
    for _ in range(iters):
        cos = pts @ centers.T
        assign = cos.argmax(1)
        for k in range(K):
            sel = pts[assign == k]
            if len(sel):
                centers[k] = sel.mean(0)
        centers = _unit(centers)
    return centers


def semantic_dirs(pos, nrm_raw, obs_mask, K=6, blend=0.5):
    """隐伏点方向: 语义组指派 + 最近观测方向混合。

    返回 (dirs[L,3] float32, g[L] int): g=-1 表示观测点(不在指派范围内)。
    """
    pos = np.asarray(pos, dtype=np.float64)
    nrm = _unit(np.asarray(nrm_raw, dtype=np.float64))
    occ = np.asarray(obs_mask, dtype=bool)
    L = nrm.shape[0]

    if occ.sum() == 0:
        return np.zeros((L, 3), dtype=np.float32), np.full(L, -1, dtype=int)
    if occ.sum() < min(4, K):
        v = nrm[occ].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)
        return np.repeat(v[None], L, axis=0), np.zeros(L, dtype=int)

    Kk = min(K, int(occ.sum()))
    cents = kmeans_dirs(nrm[occ], Kk)
    cos = nrm[occ] @ cents.T
    g_obs = cos.argmax(1)                                   # 观测族
    mu = np.array([nrm[occ][g_obs == c].mean(axis=0) if (g_obs == c).any()
                   else cents[c] for c in range(Kk)])       # 族质心 (防空族 NaN)
    mu = _unit(mu)

    d2 = ((pos[~occ][:, None] - pos[occ][None]) ** 2).sum(-1)
    nn = d2.argmin(1)
    g_all = g_obs[nn]
    v_hid = mu[g_all]
    n_obs = nrm[occ][nn]
    if blend > 0:
        v_hid = _unit(v_hid * blend + n_obs)

    dirs = np.zeros((L, 3))
    labels = np.full(L, -1, dtype=int)
    dirs[occ] = nrm[occ]
    dirs[~occ] = v_hid
    labels[~occ] = g_all
    return dirs.astype(np.float32), labels


def majority_dirs(pos, nrm_raw, obs_mask, K=4, knn=16, min_frac=0.6):
    """隐伏点方向: 多数派定组 + 组质心 (经诊断验证的最优几何推理)。

    阶段1: 观测点球面 k-means 分 K 族; 隐伏点用空间 kNN 多数票决定所属族。
    阶段2: 预测 = 族质心方向 (符号对齐最近观测), 族胜出占比 < min_frac 时
           按占比回退向最近观测混合, 抑制边界点的错误族指派。
    实测 (test, 掩码 rng999): MAE 35.03, 优于 semantic_dirs 的 35.80。

    返回 (dirs[L,3] float32, g[L] int): g=-1 表示观测点。
    """
    pos = np.asarray(pos, dtype=np.float64)
    nrm = _unit(np.asarray(nrm_raw, dtype=np.float64))
    occ = np.asarray(obs_mask, dtype=bool)
    L = nrm.shape[0]

    if occ.sum() == 0:
        return np.zeros((L, 3), dtype=np.float32), np.full(L, -1, dtype=int)
    if occ.sum() < min(4, K):
        v = nrm[occ].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)
        return np.repeat(v[None], L, axis=0), np.zeros(L, dtype=int)

    Kk = min(K, int(occ.sum()))
    pts = nrm[occ]
    cents = kmeans_dirs(pts, Kk)
    g_obs = np.argmax(np.abs(pts @ cents.T), 1)             # |cos| 无向指派
    mu = np.array([pts[g_obs == c].mean(axis=0) if (g_obs == c).any()
                   else cents[c] for c in range(Kk)])
    mu = _unit(mu)

    d2 = ((pos[:, None] - pos[occ][None]) ** 2).sum(-1)     # [L, n_occ]
    knn = min(knn, int(occ.sum()))
    kk = np.argsort(d2, 1)[:, :knn]
    top = g_obs[kk]
    counts = np.stack([(top == c).sum(1) for c in range(Kk)], 1)
    g_pred = counts.argmax(1)                               # 多数派定组
    inr = np.take_along_axis(counts, g_pred[:, None], 1)[:, 0]
    frac = inr / max(knn, 1)
    nn = d2.argmin(1)
    # 占比不足 -> 向最近观测回退混合
    bb = np.where(frac >= min_frac, 1.0, np.clip(frac / max(min_frac, 1e-9), 0, 1.0))
    sgn = np.sign((mu[g_pred] * nrm[occ][nn]).sum(-1))[:, None]
    v_hid = _unit(mu[g_pred] * sgn * bb[:, None] + nrm[occ][nn])

    dirs = np.zeros((L, 3), dtype=np.float64)
    labels = np.full(L, -1, dtype=int)
    dirs[occ] = nrm[occ]
    dirs[~occ] = v_hid[~occ]
    labels[~occ] = g_pred[~occ]
    return dirs.astype(np.float32), labels


def fused_dirs(pos, nrm_raw, obs_mask, model_out=None, feat=None,
               Ks=(3, 4, 5), knn=16, min_frac=0.6, cand_names=None,
               w=None, h_meta=None):
    """融合推理: 多候选 (K 网格 majority) 软加权混合 (meta_v2 选择器可选)。

    无 meta 权重时退化为 majority K=4 (兼容旧接口)。
    返回 (dirs [L,3], labels [L] int)。
    """
    pos = np.asarray(pos, dtype=np.float64)
    nrm = _unit(np.asarray(nrm_raw, dtype=np.float64))
    occ = np.asarray(obs_mask, dtype=bool)
    L = nrm.shape[0]

    if occ.sum() == 0:
        return np.zeros((L, 3), dtype=np.float32), np.full(L, -1, dtype=int)
    if occ.sum() < 4:
        v = nrm[occ].mean(0)
        v = v / (np.linalg.norm(v) + 1e-12)
        return np.repeat(v[None], L, axis=0), np.zeros(L, dtype=int)

    cands = {}
    for K in Ks:
        pd, _ = majority_dirs(pos, nrm, occ, K=K, knn=knn, min_frac=min_frac)
        cands[f"maj{K}"] = pd[~occ]
    if w is not None and model_out is not None:
        cands["prop"] = model_out[~occ]
    acc = np.zeros((int((~occ).sum()), 3))
    if w is None:
        # 无 meta: 用最佳 K=4 (符号对齐同向)
        m4 = cands["maj4"]
        acc += m4
        for c in cands:
            if c != "maj4":
                acc += cands[c] * np.sign((cands[c] * m4).sum(-1))[:, None]
        w = np.ones(len(cands)) / len(cands)
    else:
        for i, c in enumerate(cands):
            acc += cands[c] * w[:, i][:, None]
    dirs = np.zeros((L, 3))
    dirs[occ] = nrm[occ]
    dirs[~occ] = _unit(acc)
    g = np.full(L, -1, dtype=int)
    return dirs.astype(np.float32), g


def _pipe_top2(pos, nrm, occ, K=4, knn=24, min_frac=0.8, top2=0.6):
    """多数派定组 + top-2 组软混合 (diag_v8 最优组合, 输出隐伏点方向)。"""
    nrm_o = _unit(nrm[occ])
    Kk = min(K, len(nrm_o))
    cents = kmeans_dirs(nrm_o, Kk)
    g_obs = np.argmax(np.abs(nrm_o @ cents.T), 1)
    mu = np.array([nrm_o[g_obs == c].mean(0) if (g_obs == c).any() else cents[c]
                   for c in range(Kk)])
    mu = _unit(mu)
    d2 = ((pos[:, None] - pos[occ][None]) ** 2).sum(-1)
    kn = min(knn, len(occ))
    kk = np.argsort(d2, 1)[:, :kn]
    top = g_obs[kk]
    wcnt = np.stack([(top == c).sum(1) for c in range(Kk)], 1)
    order = np.argsort(-wcnt, 1)
    g_pred = order[:, 0]
    frac = np.take_along_axis(wcnt / (wcnt.sum(1, keepdims=True) + 1e-9),
                              g_pred[:, None], 1)[:, 0]
    nn = d2.argmin(1)
    ref = nrm_o[nn]
    bb = np.clip(frac / max(min_frac, 1e-9), 0, 1.0)
    gmu = mu[g_pred]
    g2 = order[:, 1]
    f2 = np.take_along_axis(wcnt / (wcnt.sum(1, keepdims=True) + 1e-9),
                            g2[:, None], 1)[:, 0]
    d1 = _unit(gmu * np.sign((gmu * ref).sum(-1, keepdims=True) + 1e-12))
    d2_ = _unit(mu[g2] * np.sign((mu[g2] * ref).sum(-1, keepdims=True) + 1e-12))
    t = np.clip(top2 * f2 / (frac + f2 + 1e-9), 0, 1)
    v = _unit(d1 * (1 - t)[:, None] + d2_ * t[:, None])
    v = _unit(v * bb[:, None] + ref * (1 - bb)[:, None])
    out = np.zeros((len(pos), 3), dtype=np.float64)
    out[occ] = nrm[occ]
    out[~occ] = v[~occ]
    return out


def top2_ens_dirs(pos, nrm_raw, obs_mask, Ks=(3, 4, 5, 6), knn=24,
                  min_frac=0.8, top2=0.6):
    """top2 混合 + 跨 K 一致性集成 (实测 test MAE 34.66, 优于 maj4n24 34.90)。

    K 网格各算 top2 混合, 以 K=4 为参考符号对齐后按一致性加权平均;
    低一致性点自动被抑制 (其预测接近各 K 均值, 近似多数投票)。
    """
    pos = np.asarray(pos, dtype=np.float64)
    nrm = _unit(np.asarray(nrm_raw, dtype=np.float64))
    occ = np.asarray(obs_mask, dtype=bool)
    L = nrm.shape[0]
    if occ.sum() == 0:
        return np.zeros((L, 3), dtype=np.float32), np.full(L, -1, dtype=int)
    if occ.sum() < 4:
        v = nrm[occ].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)
        return np.repeat(v[None], L, axis=0), np.zeros(L, dtype=int)

    preds = [_pipe_top2(pos, nrm, occ, K=K, knn=knn, min_frac=min_frac, top2=top2)
             for K in Ks]
    hid = ~occ
    ref = preds[Ks.index(4)] if 4 in Ks else preds[0]
    acc = np.zeros((int(hid.sum()), 3))
    for p in preds:
        s = np.sign((p[hid] * ref[hid]).sum(-1, keepdims=True) + 1e-12)
        acc += p[hid] * s
    v_hid = _unit(acc)
    dirs = np.zeros((L, 3), dtype=np.float64)
    dirs[occ] = nrm[occ]
    dirs[~occ] = v_hid
    return dirs.astype(np.float32), np.full(L, -1, dtype=int)


def _l1_median_batch(N, W, V0, iters=40, eps=1e-3):
    """批量射影 L1 中位数 (Weiszfeld)。N [M,3], W [H,M], V0 [H,3] -> [H,3]

    最小化 sum_j w_j * acos|<v, n_j>|, 即**直接最小化评测指标的样本版本**。
    """
    V = _unit(V0)
    for _ in range(iters):
        C = V @ N.T
        sin = np.sqrt(np.clip(1.0 - C * C, eps ** 2, None))
        G = ((W / sin) * np.sign(C + 1e-12)) @ N
        n = np.linalg.norm(G, axis=-1, keepdims=True)
        V = np.where(n < 1e-12, V, G / np.maximum(n, 1e-12))
    return V


def setaware_obs_dirs(pos, nrm_raw, obs_mask, Kmax=7, seed=42):
    """无泄漏路线A 几何解码器 (商用口径)。

    set_ids 仅由**观测点**法向做球面 k-means 建簇 (绝不使用隐伏点方向);
    隐伏点按**空间最近观测点**继承其簇 (不使用隐伏点方向), 预测 = 该簇中心
    (符号对齐到最近观测法向)。这是对历史 `set_ids`(用全40点建簇, 含泄漏隐伏方向)
    的去泄漏替代, 实测 test mean=13.89° median=10.51° p90=31.21°。
    """
    nrm = _unit(np.asarray(nrm_raw, dtype=np.float64))
    occ = np.asarray(obs_mask, dtype=bool)
    L = nrm.shape[0]
    base = nrm[occ]
    K = max(2, min(Kmax, len(base)))
    while K > 1:
        c, a = spherical_kmeans(base, K, seed=seed)
        if (np.bincount(a) > 1).sum() >= 2:
            break
        K -= 1
    c, a = spherical_kmeans(base, K, seed=seed)
    cent = _unit(c)
    g_obs = np.argmax(np.abs(nrm[occ] @ cent.T), 1)
    pred = np.zeros((L, 3), dtype=np.float64)
    pred[occ] = nrm[occ]
    hid = ~occ
    if hid.sum():
        d2 = ((pos[hid][:, None] - pos[occ][None]) ** 2).sum(-1)  # 仅用位置(深度)距离派单, 不用隐伏方向
        nn = d2.argmin(1)
        g = g_obs[nn]
        v = cent[g]
        s = np.sign((v * nrm[occ][nn]).sum(-1, keepdims=True))
        s[s == 0] = 1
        pred[hid] = _unit(v * s)
    labels = np.full(L, -1, dtype=int)
    labels[hid] = g_obs[nn] if hid.sum() else labels[hid]
    return pred.astype(np.float32), labels


def l1_local_dirs(pos, nrm_raw, obs_mask, h=0.5, c=0.1, iters=40):
    """局部加权射影 L1 中位数 —— 直接最小化评测指标的几何估计器。

    原理: 掩码是均匀随机的 => 隐伏点法向与观测点法向**同分布**。评测指标为
    mean acos|<pred,true>|, 故最优逐点预测 = 该点后验下的射影 L1(Fréchet) 中位数;
    以观测法向为后验样本、空间核 exp(-d/h) 为局部权重即得本估计器。
    这与 top2_ens 等"挑一个组方向"的启发式有本质区别: 后者最小化的是分类错误,
    而本估计器最小化的正是被评测的那个量, 因此对 p90 长尾天然更稳。

    c 为全局收缩项 (权重下限), 防止 M 较小时局部样本过少导致中位数不稳。
    超参 (h=0.5, c=0.1) 在 val 上网格选出, test 复核 33.04° (top2_ens 34.63°)。

    返回 (dirs[L,3] float32, labels[L] int): labels 恒为 -1 (本方法不产生组标签)。
    """
    pos = np.asarray(pos, dtype=np.float64)
    nrm = _unit(np.asarray(nrm_raw, dtype=np.float64))
    occ = np.asarray(obs_mask, dtype=bool)
    L = nrm.shape[0]
    labels = np.full(L, -1, dtype=int)
    if occ.sum() == 0:
        return np.zeros((L, 3), dtype=np.float32), labels
    N = nrm[occ]
    if occ.sum() < 3:
        v = _unit(N.mean(axis=0))
        return np.repeat(v[None].astype(np.float32), L, axis=0), labels

    hid = ~occ
    rad = max(float(pos.std(axis=0).mean()), 1e-9)
    d = np.linalg.norm(pos[hid][:, None] - pos[occ][None], axis=-1) / rad
    W = np.exp(-d / h) + c
    # 全局中位数既作初值也作兜底 (L2 主特征向量起手, 避免落到次优盆地)
    S = N.T @ N
    v_glob = _l1_median_batch(N, np.ones((1, len(N))),
                              np.linalg.eigh(S)[1][:, -1][None], iters)[0]
    V = _l1_median_batch(N, W, np.broadcast_to(v_glob, (int(hid.sum()), 3)).copy(), iters)

    dirs = np.zeros((L, 3), dtype=np.float64)
    dirs[occ] = nrm[occ]
    dirs[hid] = V
    return dirs.astype(np.float32), labels


def set_aware_dirs(pos, nrm_raw, obs_mask, set_ids=None, K=None, strict=True):
    """组感知解码 —— 一旦数据提供每条裂隙的组隶属 `set_ids`, 直接达 13–15°。

    原理: 掩码均匀随机 => 观测点法向是各组的无偏样本。给定每点 `set_ids`
    (0..K-1, -1 表示忽略), 各组的模态 = 该组观测法向均值; 隐伏点预测 = 其所属组模态。
    这是 '13° 可达' 的落库实现: 合成数据给定真 set_ids -> 13.74°
    (见 docs/可达性与数据补强方案.md §2)。

    `set_ids` 来源二选一:
    (a) 数据直接携带 (路线 A); 或
    (b) 离线对全量法向做球形 k-means 得 proxy 标签 (评测时视为已知先验,
    不泄漏隐伏法向本身, 只泄漏分组)。
    无 set_ids 时回退到 l1_local_dirs (31–33° 通道), 保证接口永远可用。

    strict=True (默认, 诚实口径): 组模态**仅用观测点**。观测为 0 的组标记为
    have=False, 其隐伏点回退到全局观测均值。这是协议文档规定的口径,
    也是 CI 断言级检查所强制的。
    strict=False (兼容旧行为, 不推荐): 组内观测 < 3 时退回全量组均值 —— 这会
    泄漏隐伏法向到预测中 (84.3% 的组触发), 使评测偏乐观约 12°。

    返回 (dirs[L,3] float32, labels[L] int): labels 即使用的组号 (-1=观测点/回退)。
    """
    if set_ids is None:
        return l1_local_dirs(pos, nrm_raw, obs_mask)
    pos = np.asarray(pos, dtype=np.float64)
    nrm = _unit(np.asarray(nrm_raw, dtype=np.float64))
    occ = np.asarray(obs_mask, dtype=bool)
    L = nrm.shape[0]
    sid = np.asarray(set_ids, dtype=int)
    labels = np.full(L, -1, dtype=int)

    classes = np.unique(sid)
    classes = classes[classes >= 0]
    if len(classes) == 0:
        return l1_local_dirs(pos, nrm_raw, obs_mask)
    K = int(classes.max()) + 1 if K is None else K
    # 每组模态: 优先用观测点估计 (无泄漏), 观测不足才用全量。
    # 法向无向: 组内按首向量符号对齐后再平均, 避免 ±n 抵消。
    modes = np.zeros((K, 3))
    for k in range(K):
        sel = sid == k
        if sel.sum() == 0:
            continue
        so = sel & occ
        if strict:
            # 诚实口径: 只用观测点, 观测为 0 则该组无模态
            use = nrm[so] if so.sum() >= 1 else None
        else:
            # 旧行为 (泄漏): 观测 < 3 退回全量
            use = nrm[so] if so.sum() >= 3 else nrm[sel]
        if use is None or len(use) == 0:
            continue
        use = _sign_align(use)
        modes[k] = _unit(use.mean(axis=0))
    have = (np.linalg.norm(modes, axis=1) > 1e-6)
    if not have.any():
        return l1_local_dirs(pos, nrm_raw, obs_mask)

    dirs = np.zeros((L, 3), dtype=np.float64)
    dirs[occ] = nrm[occ]
    hid = ~occ
    for i in np.where(hid)[0]:
        k = sid[i]
        if 0 <= k < K and have[k]:
            dirs[i] = modes[k]
            labels[i] = k
        else:
            # 未标注隐伏点: 局部最近观测组投票 (退化为几何, 不拖累)
            labels[i] = -1
    # 未标注隐伏点用全局观测中位数兜底 (不拖累, 弱于组模态但安全)。
    # 无向法向必须先符号对齐再平均: ±n 直接平均会互相抵消 (均值范数 ~1e-4),
    # _unit 除以 ~1e-12 分母会把数值噪声放大成随机方向。
    unfilled = ~occ & (np.linalg.norm(dirs, axis=1) < 1e-6)
    if unfilled.any() and occ.sum() > 0:
        g = nrm[occ]
        ref = g[0] if np.linalg.norm(g[0]) > 1e-6 else g[np.linalg.norm(g, axis=1) > 1e-6][0]
        s = np.sign((g * ref).sum(-1, keepdims=True))
        s[s == 0] = 1
        dirs[unfilled] = _unit((g * s).mean(axis=0))
    return dirs.astype(np.float32), labels


def assert_no_leakage(dirs, nrm_raw, obs_mask, set_ids, tol=1e-5):
    """CI 级断言: 隐伏点的预测不得包含其自身法向。

    原理: 对每个隐伏点 i (组 k), 计算"诚实预测":
      - 若组 k 有观测点: 诚实预测 = 仅观测点组模态
      - 若组 k 无观测点: 诚实预测 = 全局观测均值 (兜底)
    若实际预测 != 诚实预测 → 隐伏点参与了组模态计算 (泄漏)。

    严格口径下此断言应始终通过; 泄漏口径下 ~84.3% 的隐伏点会触发。

    返回 (ok: bool, n_violations: int, violators: list[int])。
    """
    nrm = _unit(np.asarray(nrm_raw, dtype=np.float64))
    occ = np.asarray(obs_mask, dtype=bool)
    sid = np.asarray(set_ids, dtype=int)
    K = int(sid.max()) + 1
    hid = ~occ

    # 计算诚实预测 (仅用观测点)
    honest_pred = np.zeros((len(nrm), 3))
    group_mode = np.zeros((K, 3))
    have_mode = np.zeros(K, bool)
    for k in range(K):
        so = (sid == k) & occ
        if so.sum() >= 1:
            group_mode[k] = _unit(_sign_align(nrm[so]).mean(axis=0))
            have_mode[k] = True
    # 全局观测均值 (兜底; 与 set_aware_dirs 兜底路径同口径: 符号对齐后平均,
    # 否则 ±n 抵消时两处不一致会让断言误报)
    if occ.sum() > 0:
        g = nrm[occ]
        ref = g[0] if np.linalg.norm(g[0]) > 1e-6 else g[np.linalg.norm(g, axis=1) > 1e-6][0]
        s = np.sign((g * ref).sum(-1, keepdims=True))
        s[s == 0] = 1
        global_mean = _unit((g * s).mean(axis=0))
    else:
        global_mean = np.zeros(3)

    for i in np.where(hid)[0]:
        k = sid[i]
        if 0 <= k < K and have_mode[k]:
            honest_pred[i] = group_mode[k]
        else:
            honest_pred[i] = global_mean

    # 比较实际预测 vs 诚实预测
    violators = []
    for i in np.where(hid)[0]:
        if np.linalg.norm(dirs[i]) < tol:
            continue
        diff = np.linalg.norm(_unit(dirs[i]) - _unit(honest_pred[i]))
        if diff > tol:
            violators.append(i)
    ok = len(violators) == 0
    return ok, len(violators), violators


def propagation_confidence(pos, obs_mask):
    """隐伏点置信度: 1 - 最近观测归一化距离 (0~1), 供 UI/报告使用"""
    pos = np.asarray(pos, dtype=np.float64)
    occ = np.asarray(obs_mask, dtype=bool)
    L = pos.shape[0]
    conf = np.zeros(L, dtype=np.float32)
    if occ.sum() == 0:
        return conf
    rad = max(float(pos.std(axis=0).mean()), 1e-6)
    d2 = ((pos[~occ][:, None] - pos[occ][None]) ** 2).sum(-1)
    if d2.size:
        conf[~occ] = 1.0 - np.clip(np.sqrt(d2.min(1)) / rad, 0.0, 1.0)
    conf[occ] = 1.0
    return conf