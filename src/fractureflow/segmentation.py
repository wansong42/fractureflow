# -*- coding: utf-8 -*-
"""无标签平面分割管线 (Label-Free Route B)。

命题: 稠密裂隙点云里, **平面结构全在位置里**。不需要 fracture_id, 也不需要
任何法向标签, 只用 xyz 就能把点云切成"裂隙平面段", 每段 SVD 即得平面法向。

实测 (hrf 1M 子采样, 真值 = frac_normals[fid], 无泄漏):
  RANSAC 分割 + 分段 SVD = 0.78° (p50 0.39 / p90 2.22)
  对照: 局部 PCA K=64 = 5.83°; KNN K=16 无标签基线 = 9.87°
  诊断上限 (完美分割 = 用 fid 分段) = 0.0057°

数值卫生 (红线):
  - 全程 float64 + 质心居中 (hrf 坐标 ~4e6 量级, float32 不居中会出 3.82° 假象);
  - 禁止 [N,N] 距离矩阵, 全部分块 / 子采样;
  - cKDTree 邻域一律用 `tree.data[nn]` 取点 (tree.data 与输入顺序不保证同一对象)。

无泄漏: 分割与法向估计只用位置 (pos), 观测法向仅在回退链里可选使用,
隐伏点真值/frac_normals 绝不进入预测路径。

入口:
  segment_planes_ransac(pos, ...) -> (seg_id, planes)
  segment_svd_dirs(pos, seg_id)   -> dirs
  label_free_dirs(pos, nrm_raw, obs_mask) -> (dirs, labels)   # 与 fracture_aware_dirs 同风格
"""

import numpy as np
from scipy.spatial import cKDTree

from .inference import _unit, _sign_align

__all__ = [
    "median_knn_dist",
    "segment_planes_ransac",
    "segment_svd_dirs",
    "label_free_dirs",
]


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _center(pos):
    """float64 + 居中 (红线 2)。返回 (X, c)。"""
    p = np.asarray(pos, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"pos 必须是 [L,3], 得到 {p.shape}")
    c = p.mean(0)
    return p - c, c


def _svd_normal(P):
    """点集最小主成分方向 (= 最佳拟合平面法向)。P[m,3] -> unit[3] 或 None。

    用 3x3 gram 矩阵 eigh (float64), 比 svd 快且等价。
    退化 (共线 / 点太少) 返回 None。
    """
    if P.shape[0] < 3:
        return None
    X = P - P.mean(0)
    S = X.T @ X
    w, v = np.linalg.eigh(S)            # 升序
    if w[2] <= 0:
        return None
    if w[1] <= 1e-12 * w[2]:            # 共线: 平面不确定
        return None
    return _unit(v[:, 0])


def median_knn_dist(X, k=8, sub=200_000, tree=None):
    """中位 k-NN 距离 (尺度估计, 用于自动 tol)。X 应已居中。

    子采样用等距抽样 (stride) 而非随机, 保证同输入可复现。
    """
    N = X.shape[0]
    if tree is None:
        tree = cKDTree(X)
    step = max(1, N // max(1, sub))
    Q = X[::step]
    kk = int(min(k, max(1, tree.n - 1)))
    dist, _ = tree.query(Q, k=kk + 1)   # 第 0 列是自身
    d = dist[:, -1] if dist.ndim == 2 else dist
    return float(np.median(d))


def _local_pca_normals(tree, q, k=64, mem_pts=3_000_000):
    """局部 PCA 法向 (回退链末端)。分块防 OOM (陷阱: 1M x 1024 x 3 f64 = 24GB)。"""
    q = np.asarray(q, dtype=np.float64)
    out = np.zeros((q.shape[0], 3), dtype=np.float64)
    if q.shape[0] == 0:
        return out
    k = int(max(3, min(k, tree.n)))
    chunk = int(max(1_000, mem_pts // k))
    for s in range(0, q.shape[0], chunk):
        qq = q[s:s + chunk]
        _, nn = tree.query(qq, k=k)
        if nn.ndim == 1:
            nn = nn[:, None]
        P = tree.data[nn]                                  # [m,k,3] (陷阱 2)
        X = P - P.mean(1, keepdims=True)
        S = np.einsum("nki,nkj->nij", X, X)
        _, v = np.linalg.eigh(S)
        out[s:s + chunk] = v[:, :, 0]
    return _unit(out)


def _auto_min_inliers(N, min_inliers):
    """小网络自适应: N 太小时不该要求 30 个 inlier (FORGE 窗口只有 ~47 点)。"""
    if N >= 50 * min_inliers:
        return int(min_inliers)
    return int(max(3, N // 10))


# --------------------------------------------------------------------------- #
# A1: RANSAC 贪心平面分割 (只用位置)
# --------------------------------------------------------------------------- #
def segment_planes_ransac(pos, seed=42, tol=None, pool_size=300_000,
                          iters_per_plane=1500, min_inliers=30, max_planes=250,
                          fallback_mult=5.0, tol_scale=2.5,
                          refine_rounds=0, adaptive_assign=False,
                          verbose=False, return_p0=False):
    """贪心 RANSAC 平面分割 (零标签, 只用 pos)。

    参数
    ----
    tol : None 时自动 = tol_scale x 中位 8-NN 距离 (子采样 200k)。
    pool_size : 提案池大小 (子采样), 每轮在池内做 iters_per_plane 次 3 点采样。
    min_inliers : 池内最佳 inlier 少于此值即停 (小网络自动放宽)。
    fallback_mult : 未归点在 fallback_mult x tol 内可挂到最近平面, 否则 -1。
    refine_rounds : >0 时做 EM 式 refit&reassign (每点改归残差最小平面 + 段内 SVD)。
    adaptive_assign : True 时分配阶段用逐点局部尺度阈值 (每点 8-NN 距离 x tol_scale)。

    返回
    ----
    (seg_id[L] int, planes[n,3] float64)  ; return_p0=True 时追加 p0s[n,3]
    seg_id = -1 表示未归段。planes 是各段提案法向 (最终法向请用 segment_svd_dirs)。
    """
    X, _ = _center(pos)
    N = X.shape[0]
    seg = np.full(N, -1, dtype=int)
    if N < 3:
        empty = np.zeros((0, 3))
        return (seg, empty, empty) if return_p0 else (seg, empty)

    tree = cKDTree(X)
    if tol is None:
        d8 = median_knn_dist(X, k=8, tree=tree)
        tol = tol_scale * d8
        if verbose:
            print(f"[seg] d8={d8:.4f} tol={tol:.4f}")
    tol = float(tol)
    if tol <= 0:
        tol = 1e-9

    pt_tol = None
    if adaptive_assign:
        kk = int(min(8, max(1, tree.n - 1)))
        dist, _ = tree.query(X, k=kk + 1)
        pt_tol = tol_scale * dist[:, -1]
        pt_tol = np.maximum(pt_tol, 1e-12)

    mi = _auto_min_inliers(N, min_inliers)
    rng = np.random.default_rng(seed)
    npool = int(min(pool_size, N))
    pool_idx = rng.choice(N, npool, replace=False) if npool < N else np.arange(N)
    pool = X[pool_idx]
    remain = np.ones(npool, bool)

    planes, p0s = [], []
    un_idx = np.arange(N)               # 尚未归段的全量索引 (逐轮收缩, 避免重复扫描)

    while remain.sum() > max(50, mi) and len(planes) < max_planes:
        pr = pool[remain]
        best = None
        for _ in range(iters_per_plane):
            s = rng.integers(0, pr.shape[0], 3)
            a, b, p0 = pr[s]
            n = np.cross(a - p0, b - p0)
            nl = np.linalg.norm(n)
            if nl < 1e-12:
                continue
            n = n / nl
            cnt = int((np.abs((pr - p0) @ n) < tol).sum())
            if best is None or cnt > best[0]:
                best = (cnt, n, p0)
        if best is None or best[0] < mi:
            break
        _, n, p0 = best

        # 池内 inlier -> SVD 精化
        inl = np.abs((pr - p0) @ n) < tol
        nref = _svd_normal(pr[inl])
        if nref is not None:
            n = nref if float(n @ nref) >= 0 else -nref

        # 全量分配 (只看未归点)
        if un_idx.size:
            Xu = X[un_idx]
            d_all = np.abs((Xu - p0) @ n)
            thr = pt_tol[un_idx] if pt_tol is not None else tol
            hit = d_all < thr
            seg[un_idx[hit]] = len(planes)
            un_idx = un_idx[~hit]
        planes.append(n)
        p0s.append(p0)

        rm = remain.copy()
        rm[rm] &= ~inl
        remain = rm
        if verbose and (len(planes) % 20 == 0 or len(planes) < 4):
            print(f"[seg] plane#{len(planes)} pool_inl={int(inl.sum()):7d} "
                  f"unassigned={un_idx.size:8d} remain_pool={int(remain.sum()):7d}")

    planes_a = np.asarray(planes, dtype=np.float64).reshape(-1, 3)
    p0s_a = np.asarray(p0s, dtype=np.float64).reshape(-1, 3)

    # 未归点: 最近平面 (fallback_mult x tol 内) -> 段号; 否则 -1
    if un_idx.size and planes_a.shape[0]:
        Xu = X[un_idx]
        best_d = np.full(un_idx.size, np.inf)
        best_p = np.full(un_idx.size, -1, dtype=int)
        for pi in range(planes_a.shape[0]):
            dd = np.abs((Xu - p0s_a[pi]) @ planes_a[pi])
            m = dd < best_d
            best_d[m] = dd[m]
            best_p[m] = pi
        thr = pt_tol[un_idx] if pt_tol is not None else tol
        ok = (best_d < fallback_mult * thr) & (best_p >= 0)
        take = un_idx[ok]                                   # 陷阱 4: 先索引再合并
        seg[take] = best_p[ok]

    if refine_rounds > 0 and planes_a.shape[0]:
        seg = _em_refit(X, seg, planes_a, p0s_a, tol, fallback_mult,
                        rounds=refine_rounds, pt_tol=pt_tol, verbose=verbose)

    if verbose:
        print(f"[seg] {planes_a.shape[0]} planes, assigned={int((seg >= 0).sum())}/{N}")
    if return_p0:
        return seg, planes_a, p0s_a
    return seg, planes_a


def _em_refit(X, seg, planes, p0s, tol, fallback_mult, rounds=1, pt_tol=None,
              verbose=False):
    """EM 式: 段内 SVD refit -> 每点改归"残差最小平面" -> 重复。修正交点归属偏差。"""
    N = X.shape[0]
    n_pl = planes.shape[0]
    cur_n = planes.copy()
    cur_p = p0s.copy()
    for r in range(rounds):
        # M 步: 段内 SVD
        for pi in range(n_pl):
            m = seg == pi
            if int(m.sum()) >= 3:
                nn = _svd_normal(X[m])
                if nn is not None:
                    cur_n[pi] = nn
                    cur_p[pi] = X[m].mean(0)
        # E 步: 残差最小平面 (分块, 禁 [N,n] 大矩阵一次性)
        best_d = np.full(N, np.inf)
        best_p = np.full(N, -1, dtype=int)
        chunk = 1_000_000
        for s in range(0, N, chunk):
            Xc = X[s:s + chunk]
            bd = np.full(Xc.shape[0], np.inf)
            bp = np.full(Xc.shape[0], -1, dtype=int)
            for pi in range(n_pl):
                dd = np.abs((Xc - cur_p[pi]) @ cur_n[pi])
                m = dd < bd
                bd[m] = dd[m]
                bp[m] = pi
            best_d[s:s + chunk] = bd
            best_p[s:s + chunk] = bp
        thr = pt_tol if pt_tol is not None else np.full(N, tol)
        ok = best_d < fallback_mult * thr
        new_seg = np.full(N, -1, dtype=int)
        new_seg[ok] = best_p[ok]
        moved = int((new_seg != seg).sum())
        seg = new_seg
        if verbose:
            print(f"[seg] EM round {r + 1}: moved={moved}")
        if moved == 0:
            break
    return seg


# --------------------------------------------------------------------------- #
# A2: 分段 SVD 法向 + 库内推理入口
# --------------------------------------------------------------------------- #
def segment_svd_dirs(pos, seg_id, planes=None, pca_k=64, tree=None):
    """每段 SVD 平面法向; 段内 <3 点 / 退化 / seg=-1 -> 局部 PCA K=pca_k 回退。

    返回 dirs[L,3] float64 (单位向量)。
    """
    X, _ = _center(pos)
    N = X.shape[0]
    seg = np.asarray(seg_id, dtype=int)
    dirs = np.zeros((N, 3), dtype=np.float64)
    need_pca = seg < 0

    if planes is not None:
        planes = np.asarray(planes, dtype=np.float64).reshape(-1, 3)
    for pi in np.unique(seg[seg >= 0]):
        m = seg == pi
        nn = _svd_normal(X[m]) if int(m.sum()) >= 3 else None
        if nn is None and planes is not None and pi < planes.shape[0]:
            nn = planes[pi]                        # 段太小 -> 用提案法向
        if nn is None:
            need_pca |= m
        else:
            dirs[m] = nn

    if need_pca.any():
        if tree is None:
            tree = cKDTree(X)
        dirs[need_pca] = _local_pca_normals(tree, X[need_pca], k=pca_k)
    return _unit(dirs)


def label_free_dirs(pos, nrm_raw=None, obs_mask=None, set_ids=None,
                    seed=42, tol=None, pca_k=64, min_pts=12, **seg_kw):
    """无标签平面分割解码 (签名/返回对齐 connectivity.fracture_aware_dirs)。

    只用 pos: RANSAC 分割 -> 分段 SVD 法向 -> 观测点与隐伏点**同样**赋分段法向
    (观测点也被几何去噪, 这是 0.78° 的关键之一)。

    nrm_raw / obs_mask 仅用于: (a) 点数过少时回退到有法向的局部策略;
    (b) 输出符号对齐 (评测口径 |cos| 无关符号, 仅为可读性)。
    真值 / 隐伏法向绝不进入预测路径 (红线 1)。

    返回 (dirs[L,3] float32, labels[L] int = seg_id, -1 = 局部 PCA 回退)。
    """
    p = np.asarray(pos, dtype=np.float64)
    N = p.shape[0]
    if N == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=int)

    if N < max(6, min_pts):
        # 太小: 分割无意义, 回退到既有策略 (需要法向)
        if nrm_raw is not None and obs_mask is not None:
            from .inference import l1_local_dirs
            d, _ = l1_local_dirs(p, nrm_raw, obs_mask)
            return np.asarray(d, dtype=np.float32), np.full(N, -1, dtype=int)
        X, _ = _center(p)
        tree = cKDTree(X)
        d = _local_pca_normals(tree, X, k=min(pca_k, N))
        return d.astype(np.float32), np.full(N, -1, dtype=int)

    seg, planes = segment_planes_ransac(p, seed=seed, tol=tol, **seg_kw)
    dirs = segment_svd_dirs(p, seg, planes=planes, pca_k=pca_k)

    if nrm_raw is not None and obs_mask is not None:
        occ = np.asarray(obs_mask, dtype=bool)
        if occ.any():
            nrm = _unit(np.asarray(nrm_raw, dtype=np.float64))
            s = np.sign((dirs[occ] * nrm[occ]).sum(-1, keepdims=True))
            s[s == 0] = 1
            dirs[occ] = dirs[occ] * s
    return dirs.astype(np.float32), seg


def segment_quality(pos, seg_id):
    """诊断: 每段点数 / 段内平面残差 (无真值, 可上线自检)。"""
    X, _ = _center(pos)
    seg = np.asarray(seg_id, dtype=int)
    rows = []
    for pi in np.unique(seg):
        m = seg == pi
        if pi < 0:
            rows.append((int(pi), int(m.sum()), float("nan")))
            continue
        nn = _svd_normal(X[m])
        if nn is None:
            rows.append((int(pi), int(m.sum()), float("nan")))
            continue
        r = np.abs((X[m] - X[m].mean(0)) @ nn)
        rows.append((int(pi), int(m.sum()), float(r.mean())))
    return rows


# --------------------------------------------------------------------------- #
# R105 产品化单入口 (L3 无标签点云管线收口): python -m fractureflow.segmentation
#
# 最小 diff 留痕 (2026-08-27, R105): 本文件只新增 CLI 入口, 不引入任何新算法;
# 六段流水 (加载→降采样→RANSAC→组系表→DFN 筛查→报告) 的唯一实现在
# scripts/v_r105_productize.py (R105 收口脚本), 此处延迟 import 复用, 避免重复实现.
# 能力声明与预注册见 results/v_r105_capability.md / results/v_r105_preregister.json.
# --------------------------------------------------------------------------- #
def _entrypoint(argv=None):
    """CLI 入口 → scripts/v_r105_productize.py::main (延迟导入, 只组装不发明)."""
    import os
    import sys

    _src_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _scripts = os.path.join(_src_root, "scripts")
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    try:
        import v_r105_productize as prod
    except ImportError as e:  # pragma: no cover - 环境损坏才会走到
        raise SystemExit(f"[fractureflow.segmentation] R105 收口脚本不可用 (import 失败): {e}")
    return prod.main(argv)


if __name__ == "__main__":
    _entrypoint()
