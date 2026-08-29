# -*- coding: utf-8 -*-
"""路线 A 优化版组标注生成器 (setlabel 升级).

相对 setlabel.py 的改进 (全部为"让组标注更稳/更准", 而非换评测口径):
-----------------------------------------------------------------------
1. 多重启球面 k-means: 原版每个 K 只做 **1 次**随机初始化, 极易陷局部最优
   (球面 |cos| 距离下 k-means 对初值极敏感). 本版每个 K 跑 R 次重启
   (k-means++ 球面初值 + 随机初值混合), 保留惯性最小 (簇内最紧) 的解.
2. K 选择用 vMF 混合 BIC, 而非仅靠轮廓系数: 裂隙组本质是"角向簇",
   von Mises-Fisher 是其自然生成模型, 浓度 κ 直接对应组内角离散度
   (即最终误差上界). BIC = 2·logL − ν·log n 自动在"组数 vs 拟合度"间权衡,
   避免轮廓系数在高维球面倾向过选/漏选 K 的缺陷. 轮廓系数仍作并列报告.
3. 符号对齐更稳: 簇内均值用"对当前中心对齐"而非固定 pts[0],  outlier 更鲁棒.

泄漏属性: set_ids 是分析师从**全量 DFN 量测**给出的"数据属性"; 本模块只用
net["nrm"] (全量法向场) 拟合, 与隐伏/观测掩码无关, 推理时观测/隐伏点全已知,
不泄漏隐伏法向本身. (K 选择用全量场是合法的, 等同分析师从完整 DFN 定组数.)

训练时组方向 set_dirs 仍由 data.py 仅用观测点构造 (已修零向量污染), 与本模块解耦.
"""

import numpy as np

EPS = 1e-12
KAPPA_CAP = 50.0  # 防 sinh/κ 溢出 (坑 #4)


def _unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + EPS)


def _sign_align(pts, ref=None):
    """无向法向符号对齐: 以 ref (默认首向量) 为基准, 把每条法向翻到同号半空间."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return pts
    if ref is None:
        ref = pts[0]
    s = np.sign((pts * ref).sum(-1, keepdims=True))
    s[s == 0] = 1
    return pts * s


# ----------------------- 球面 k-means (多重启) -----------------------
def _spherical_kmeans_once(pts, K, centers0, iters=80):
    """单次 Lloyd. 返回 (centers[K,3], assign[L], inertia). 用 |cos| 指派."""
    centers = _unit(centers0.copy())
    assign = np.zeros(len(pts), dtype=int)
    for _ in range(iters):
        cos = np.abs(pts @ centers.T)            # [L,K]
        assign = cos.argmax(1)
        new_c = centers.copy()
        for k in range(K):
            sel = pts[assign == k]
            if len(sel):
                # 符号对齐到当前中心再平均, 比固定首向量更稳
                new_c[k] = _unit(_sign_align(sel, ref=centers[k]).mean(0))
        if np.allclose(new_c, centers, atol=1e-9):
            centers = new_c
            break
        centers = new_c
    inertia = float((1.0 - np.abs(pts @ centers.T).max(1)).sum())
    return centers, assign, inertia


def _kpp_init(pts, K, rng):
    """球面 k-means++: 按 |cos| 距离的最远点采样选初值."""
    n = len(pts)
    first = rng.integers(n)
    centers = [pts[first]]
    d2 = 1.0 - np.abs(pts @ pts[first])   # 到最近已选中心的距离(角距 proxy)
    for _ in range(1, K):
        s = d2.sum()
        # 数值安全: 退化 (全零/非有限) 时回退均匀
        if not np.isfinite(s) or s <= EPS:
            idx = rng.integers(n)
        else:
            probs = d2 / s
            probs = probs / probs.sum()   # 精确归一, 防 rng.choice 浮点误差
            idx = rng.choice(n, p=probs)
        centers.append(pts[idx])
        d2 = np.minimum(d2, 1.0 - np.abs(pts @ pts[idx]))
    return np.array(centers, dtype=np.float64)


def spherical_kmeans_restarts(pts, K, n_restarts=20, iters=80, seed=0):
    """多重启球面 k-means. 返回 (centers, assign, inertia). 保留惯性最小解."""
    pts = _unit(np.asarray(pts, dtype=np.float64))
    n = len(pts)
    K = min(K, n)
    if K <= 1:
        c = _unit(_sign_align(pts).mean(0))[None]
        return c, np.zeros(n, dtype=int), float((1.0 - np.abs(pts @ c.T).max(1)).sum())
    rng = np.random.default_rng(seed)
    best = None
    for r in range(n_restarts):
        if r == 0:
            init = _kpp_init(pts, K, rng)
        else:
            init = pts[rng.choice(n, K, replace=False)]
        c, a, ine = _spherical_kmeans_once(pts, K, init, iters)
        if best is None or ine < best[2] - 1e-12:
            best = (c, a, ine)
    return best


# ----------------------- 轮廓系数 (并列报告) -----------------------
def _silhouette(nrm, centers, assign):
    pts = _unit(nrm)
    K = len(centers)
    abscos = np.abs(pts @ centers.T)
    a = np.zeros(len(pts))
    for k in range(K):
        sel = assign == k
        if sel.sum() <= 1:
            a[sel] = 0.0
            continue
        a[sel] = 1.0 - abscos[sel, k]
    others = [k for k in range(K) if (assign == k).sum() > 1]
    b = np.full(len(pts), 0.0)  # 单组或不足时 b=0 (轮廓=0, 不影响 BIC 决策)
    if len(others) > 1:
        oc = np.full(len(pts), -np.inf)
        for k2 in others:
            oc = np.maximum(oc, abscos[:, k2])
        for k in range(K):
            sel = assign == k
            if not sel.any():
                continue
            b[sel] = 1.0 - oc[sel]
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-9)
    return float(np.nanmean(s))


# ----------------------- vMF 混合 BIC (K 选择) -----------------------
def _vmf_logC3(kappa):
    """log C_3(κ) = 0.5 log κ − 1.5 log(2π) − log I_{0.5}(κ),
    I_{0.5}(κ)=sqrt(2/(π κ))·sinh κ. κ 截断防溢出."""
    k = min(float(kappa), KAPPA_CAP)
    log_I = 0.5 * np.log(2.0 / (np.pi * k) + EPS) + np.log(np.sinh(k) + EPS)
    return 0.5 * np.log(k + EPS) - 1.5 * np.log(2.0 * np.pi) - log_I


def _vmf_bic(pts, centers, assign):
    """对给定聚类拟合 vMF 混合, 返回 (logL, bic, kappas[K]). 3 自由参数/组."""
    pts = _unit(pts)
    K = len(centers)
    n = len(pts)
    logL = 0.0
    kappas = np.zeros(K)
    for k in range(K):
        sel = pts[assign == k]
        nk = len(sel)
        if nk == 0:
            kappas[k] = 0.0
            continue
        R = float(np.linalg.norm(_sign_align(sel, ref=centers[k]).sum(0)) / nk)  # 合成长度
        R = min(max(R, 1e-6), 1.0 - 1e-9)
        # p=3 的 κ 近似: κ ≈ R(3−R²)/(1−R²)
        kappa = R * (3.0 - R * R) / (1.0 - R * R + EPS)
        kappa = min(kappa, KAPPA_CAP)
        kappas[k] = kappa
        # log f_i = log C_3(κ) + κ · (x_i·μ); 簇内用中心对齐后合成长度 R 期望
        logL += nk * (_vmf_logC3(kappa) + kappa * R)
    nu = 3 * K  # 每组 μ(2 自由)+κ(1)
    bic = 2.0 * logL - nu * np.log(max(n, 2))
    return float(logL), float(bic), kappas


# ----------------------- 顶层: 选 K + 标注 -----------------------
def generate_set_ids_opt(nrm, Kmin=1, Kmax=8, n_restarts=20, iters=80,
                         seed=42, min_pts=4, select="bic", verbose=False):
    """为完整法向场生成 set_id. 返回 int 数组 [L].

    select: "bic" (默认, 推荐) 或 "sil" (轮廓). 两种都算, 仅决策依据不同.
    """
    nrm = np.asarray(nrm, dtype=np.float64)
    L = nrm.shape[0]
    if L < min_pts or L <= Kmin:
        return np.zeros(L, dtype=int)

    best = None  # (score, K, centers, assign, sil, bic, kappas)
    for K in range(Kmin, min(Kmax, L) + 1):
        c, a, ine = spherical_kmeans_restarts(nrm, K, n_restarts, iters, seed=seed + K)
        sil = _silhouette(nrm, c, a)
        logL, bic, kappas = _vmf_bic(nrm, c, a)
        score = bic if select == "bic" else sil
        if verbose:
            print(f"  K={K} inertia={ine:.3f} sil={sil:+.3f} BIC={bic:.1f} "
                  f"κmean={float(np.mean(kappas[kappas>0])):.2f}")
        if best is None or score > best[0]:
            best = (score, K, c, a, sil, bic, kappas)

    if best is None:
        c, a, _ = spherical_kmeans_restarts(nrm, Kmin, n_restarts, iters, seed=seed)
        return a.astype(int)
    return best[3].astype(int)


def label_nets_opt(nets, Kmin=1, Kmax=8, n_restarts=20, iters=80,
                   seed=42, select="bic", verbose=False):
    """给 net dict 列表就地加 'set_ids'. 返回 (新列表, 统计 dict)."""
    out, stats = [], {"K_hist": {}, "sil_before": [], "sil_after": [], "n": 0}
    for net in nets:
        n = dict(net)
        nrm = np.asarray(net["nrm"], dtype=np.float64)
        sid = generate_set_ids_opt(nrm, Kmin, Kmax, n_restarts, iters,
                                    seed=seed, select=select, verbose=verbose)
        n["set_ids"] = sid
        K = int(sid.max()) + 1 if (len(sid) and sid.max() >= 0) else 0
        stats["K_hist"][K] = stats["K_hist"].get(K, 0) + 1
        stats["n"] += 1
        out.append(n)
    return out, stats
