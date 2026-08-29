# -*- coding: utf-8 -*-
"""随机 DFN 生成器 —— Baecher 盘模型。

输入: SetTable (组系: 方向模态 + 浓度) + 强度 P32 + 假设参数 (尺寸幂律 β, 域尺寸)。
输出: DFNRealization (中心 / 法向 / 半径 / 组来源)。

物理模型:
  - 中心: 域内均匀撒点 (泊松过程, 数密度 n = N/V)
  - 方向: 按组内 vMF 分布 (浓度 κ = 组内离散度的逆)
  - 半径: 幂律 p(r) ∝ r^{-β} 截断于 [r_min, r_max]
  - 数量: 由 P32 = Σ π r² / V 反解 N

参考: Balberg et al. (1984) 排除体积理论, n_c·⟨V_ex⟩ ≈ 2.7。

防坑:
  - H2: 尺寸用井壁迹长直接当直径 → 本模块只做统计实现, 不声称是真实裂隙
  - H4: β=4 时幂律采样数值下溢 → 对数空间采样 + r_min 截断
  - H6: 渗流曲线用 3 个实现就出结论 → 由 percolation.py 控制, 本模块只生成

用法:
    from fractureflow.dfn import SetTable, generate_dfn, build_connectivity_graph
    st = SetTable(centers=..., concentrations=..., proportions=...)
    dfn = generate_dfn(st, p32=1.0, beta=3.5, domain=(100,100,100), seed=42)
    G = build_connectivity_graph(dfn)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
from scipy.spatial import cKDTree
from scipy import sparse


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class SetTable:
    """组系表 —— DFN 生成的方向输入。

    centers:        (K, 3) 各组模态方向 (单位向量, 无向)
    concentrations: (K,) vMF 浓度参数 κ (越大越集中; κ→0 即均匀)
    proportions:    (K,) 各组占比 (和为 1)
    """
    centers: np.ndarray
    concentrations: np.ndarray
    proportions: np.ndarray

    def __post_init__(self):
        self.centers = np.asarray(self.centers, dtype=np.float64)
        self.concentrations = np.asarray(self.concentrations, dtype=np.float64)
        self.proportions = np.asarray(self.proportions, dtype=np.float64)
        # 单位化
        norms = np.linalg.norm(self.centers, axis=1, keepdims=True)
        self.centers = self.centers / np.clip(norms, 1e-12, None)
        # 归一化比例
        self.proportions = self.proportions / self.proportions.sum()
        self.K = self.centers.shape[0]

    @staticmethod
    def from_centers(centers: np.ndarray, kappa: float = 20.0):
        """从方向中心快速构建 (等比例, 统一浓度)。"""
        centers = np.asarray(centers, dtype=np.float64)
        K = centers.shape[0]
        return SetTable(
            centers=centers,
            concentrations=np.full(K, kappa),
            proportions=np.ones(K) / K,
        )


@dataclass
class DFNRealization:
    """单次 DFN 实现。"""
    centers: np.ndarray   # (M, 3)
    normals: np.ndarray   # (M, 3) 单位法向
    radii: np.ndarray     # (M,) 半径
    sets: np.ndarray      # (M,) 组来源标记 (0..K-1)
    # 生成所用域尺寸 (Lx, Ly, Lz)。PBC 周期必须 = 真实 domain, 而非数据极值
    # (八期 B7 教训: 用数据包围盒做周期会在稀疏 DFN 上过度连通)。
    domain: Optional[Tuple[float, float, float]] = None

    @property
    def n_fractures(self) -> int:
        return self.centers.shape[0]


# ---------------------------------------------------------------------------
# 采样工具
# ---------------------------------------------------------------------------

def _sample_vmf(mu: np.ndarray, kappa: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """从 vMF(μ, κ) 在 S² 上采样 n 个点 (Ulrich 1984, 逆 CDF 法)。

    μ: (3,) 单位均值方向
    κ: 浓度参数 (κ < 1e-8 退化为均匀)
    """
    mu = np.asarray(mu, dtype=float)
    mu = mu / np.linalg.norm(mu)

    if kappa < 1e-8 or n == 0:
        if n == 0:
            return np.empty((0, 3))
        xyz = rng.standard_normal((n, 3))
        return xyz / np.linalg.norm(xyz, axis=1, keepdims=True)

    # w = cos(θ) 服从截断指数分布 on [-1, 1], 率参数 κ
    # 逆 CDF: w = log(exp(-κ) + u*(exp(κ) - exp(-κ))) / κ
    u = rng.random(n)
    e_pos = np.exp(kappa)
    e_neg = np.exp(-kappa)
    w = np.log(e_neg + u * (e_pos - e_neg)) / kappa
    w = np.clip(w, -1.0, 1.0)

    # 方位角均匀
    phi = rng.uniform(0, 2 * np.pi, size=n)
    r = np.sqrt(np.clip(1.0 - w ** 2, 0.0, None))

    # 构造 μ 的正交基
    if abs(mu[0]) < 0.9:
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(mu, v)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(mu, e1)
    e2 = e2 / np.linalg.norm(e2)

    samples = (w[:, None] * mu[None, :]
               + (r * np.cos(phi))[:, None] * e1[None, :]
               + (r * np.sin(phi))[:, None] * e2[None, :])

    return samples / np.linalg.norm(samples, axis=1, keepdims=True)


def _sample_powerlaw(beta: float, r_min: float, r_max: float, n: int,
                      rng: np.random.Generator) -> np.ndarray:
    """从幂律 p(r) ∝ r^{-β} on [r_min, r_max] 采样 n 个半径 (逆 CDF)。

    β < 3: 大半径主导, 逆 CDF 直接可用
    β = 1: 对数均匀
    β > 3: 小半径主导
    """
    # R4 守卫: 非有限 β (inf/nan) 会让逆 CDF 退化成常量 1.0 (静默错误), 显式拒绝.
    if not np.isfinite(beta):
        raise ValueError(f"beta 必须有限, 收到 {beta!r} (非有限 β 会产出退化半径)")
    if n == 0:
        return np.empty(0)
    u = rng.random(n)
    if abs(beta - 1.0) < 1e-6:
        # β=1 (对数均匀): r = r_min * (r_max/r_min)^u
        radii = r_min * (r_max / r_min) ** u
    else:
        # 一般 β ≠ 1: r = [r_min^{1-β} + u*(r_max^{1-β} - r_min^{1-β})]^{1/(1-β)}
        exp = 1.0 - beta
        base_min = r_min ** exp
        base_max = r_max ** exp
        radii = (base_min + u * (base_max - base_min)) ** (1.0 / exp)
    return radii


# ---------------------------------------------------------------------------
# 主生成器
# ---------------------------------------------------------------------------

def generate_dfn(set_table: SetTable, p32: float, beta: float,
                 domain: Tuple[float, float, float], seed: int,
                 n_min: int = 200, r_min: Optional[float] = None,
                 r_max: Optional[float] = None) -> DFNRealization:
    """生成一个 Baecher 盘 DFN 实现。

    参数:
        set_table: 组系表 (方向 + 浓度 + 比例)
        p32: 目标体强度 P32 (m²/m³)
        beta: 幂律指数 (通常 3–4)
        domain: (Lx, Ly, Lz) 域尺寸 (m)
        seed: 随机种子
        n_min: 最少裂隙数 (防退化)
        r_min, r_max: 半径截断 (默认: 域最短边的 1e-3 和 5e-2)

    返回:
        DFNRealization
    """
    rng = np.random.default_rng(seed)
    Lx, Ly, Lz = domain
    # R4 守卫: 非有限 β / p32 会让生成器退化或产出坏数, 显式拒绝 (不静默).
    if not np.isfinite(beta):
        raise ValueError(f"generate_dfn: beta 必须有限, 收到 {beta!r}")
    if not np.isfinite(p32) or p32 < 0:
        raise ValueError(f"generate_dfn: p32 必须 >=0 且有限, 收到 {p32!r}")
    V = Lx * Ly * Lz

    # 默认半径截断
    # r_min: 默认 0.5m (地质合理最小值), 避免 β>3 时产生过多微裂隙
    # r_max: 域最短边的 5% (避免单裂隙占域过大)
    shortest = min(Lx, Ly, Lz)
    if r_min is None:
        r_min = max(0.5, shortest * 1e-2)  # 至少 0.5m
    if r_max is None:
        r_max = shortest * 5e-2

    # 由 P32 反解 N: P32 = N * E[πr²] / V → N = P32 * V / (π * E[r²])
    # 自适应 r_min: 如果 N 过大, 迭代提高 r_min 防内存爆炸
    N_max = 20000
    n_pilot = 10000
    for _ in range(10):
        pilot_radii = _sample_powerlaw(beta, r_min, r_max, n_pilot, rng)
        mean_r2 = np.mean(pilot_radii ** 2)
        mean_area = np.pi * mean_r2
        N = max(n_min, int(round(p32 * V / mean_area))) if mean_area > 1e-15 else n_min
        if N <= N_max or r_min >= r_max * 0.5:
            break
        # 提高 r_min 来减少 N (近似: N ∝ 1/E[r²] ∝ r_min^{-(β-2)} for β > 2)
        r_min = min(r_min * 1.5, r_max * 0.5)

    if mean_area < 1e-15 or p32 <= 0:
        return DFNRealization(
            centers=np.empty((0, 3)), normals=np.empty((0, 3)),
            radii=np.empty(0), sets=np.empty(0, dtype=int))

    N = max(n_min, int(round(p32 * V / mean_area)))

    # 各组裂隙数 (按 proportions 分配)
    n_per_set = np.round(set_table.proportions * N).astype(int)
    # 修正舍入误差
    while n_per_set.sum() < N:
        n_per_set[rng.integers(0, set_table.K)] += 1
    while n_per_set.sum() > N:
        idx = np.argmax(n_per_set)
        n_per_set[idx] -= 1

    all_centers = []
    all_normals = []
    all_radii = []
    all_sets = []

    for k in range(set_table.K):
        nk = n_per_set[k]
        if nk <= 0:
            continue

        # 中心: 域内均匀撒点 (中心对称)
        ctrs = rng.uniform(-0.5, 0.5, size=(nk, 3)) * np.array(domain)

        # 方向: vMF(μ_k, κ_k)
        nrms = _sample_vmf(set_table.centers[k], set_table.concentrations[k], nk, rng)

        # 半径: 幂律
        rads = _sample_powerlaw(beta, r_min, r_max, nk, rng)

        all_centers.append(ctrs)
        all_normals.append(nrms)
        all_radii.append(rads)
        all_sets.append(np.full(nk, k, dtype=int))

    return DFNRealization(
        centers=np.vstack(all_centers) if all_centers else np.empty((0, 3)),
        normals=np.vstack(all_normals) if all_normals else np.empty((0, 3)),
        radii=np.concatenate(all_radii) if all_radii else np.empty(0),
        sets=np.concatenate(all_sets) if all_sets else np.empty(0, dtype=int),
        domain=domain,
    )


# ---------------------------------------------------------------------------
# 连通图构建
# ---------------------------------------------------------------------------

def _distance_vec(a: np.ndarray, b: np.ndarray, pbc: bool, box: np.ndarray) -> np.ndarray:
    """计算点对距离 (支持 PBC).

    pbc=True 时使用最小镜像约定: d_axis = min(|Δ|, box-|Δ|).
    a: (N, 3) 或 (3,), b: (N, 3) 或 (3,) (广播).
    """
    diff = np.abs(a - b)
    if pbc:
        diff = np.minimum(diff, box - diff)
    return np.linalg.norm(diff, axis=-1)


def build_connectivity_graph(dfn: DFNRealization, pbc: bool = False,
                              batch_size: int = 5000,
                              domain: Optional[Tuple[float, float, float]] = None) -> sparse.csr_matrix:
    """构建裂隙连通图 (两裂隙连通 = 圆盘相交)。

    判交准则 (第一版): 中心距 d(c_i, c_j) < r_i + r_j。
    这等价于把圆盘当成球体的排除体积近似, 与 Balberg 理论锚点一致。

    加速: cKDTree query_pairs (向量化, 比逐批 query_ball_point 快 5-10x)。
    对超大网络 (pair 数 > 20M), 回退到分批 query_ball_point 防内存爆炸。

    PBC 修复 (T68-A): cKDTree(boxsize=...) 要求所有坐标 ∈ [0, box)。
    generate_dfn 生成中心 ∈ [-L/2, L/2], pbc=True 时本函数内部自动平移
    中心到 [0, L) (不改变 dfn 原始数据), 平移仅用于 k-d 树建树 + 距离计算。

    参数:
        dfn: DFN 实现
        pbc: 是否周期性边界 (默认 False)
        batch_size: 每批处理的裂隙数 (仅超大网络回退时使用)

    返回:
        scipy.sparse.csr_matrix (M, M) 对称邻接矩阵
    """
    M = dfn.n_fractures
    if M == 0:
        return sparse.csr_matrix((0, 0))

    r_max = dfn.radii.max() if len(dfn.radii) > 0 else 1.0
    search_radius = 2 * r_max  # 最大可能的 r_i + r_j

    # PBC 周期 = 真实 domain, 而非数据包围盒 (八期 B7 教训: 稀疏 DFN 上
    # 数据极值 < domain 会导致幻影跨边界边/过度连通)。generate_dfn 生成中心
    # ∈ [-L/2, L/2], 平移 +L/2 即对齐到 [0, L), 与 domain 周期一致。
    centers = dfn.centers
    center_shift = np.zeros(3)
    if pbc:
        if domain is not None:
            box = np.asarray(domain, dtype=float)
        elif dfn.domain is not None:
            box = np.asarray(dfn.domain, dtype=float)
        else:
            raise ValueError(
                "build_connectivity_graph(pbc=True) 需要真实 domain 作为周期, "
                "但 domain 参数与 dfn.domain 均为空。数据极值不是周期 (八期 B7 教训)! "
                "请在 generate_dfn 中携带 domain, 或显式传入 domain=。")
        if not np.all(box > 0):
            raise ValueError(f"domain 必须为正: {box}")
        center_shift = box / 2.0   # [-L/2, L/2] -> [0, L)
        centers = centers + center_shift
        # 严格落入基本域 [0, box) (含恰落在域壁的点, 否则 cKDTree 抛
        # "Some input data are greater than the size of the periodic box")
        centers = np.mod(centers, box)
        tree = cKDTree(centers, boxsize=box)
    else:
        box = np.zeros(3)  # pbc=False 时 _distance_vec 不使用 box
        tree = cKDTree(centers)

    # 快速路径: query_pairs 向量化 (避免 Python 层级循环)
    # 对 M=15k, search_radius=10m, domain=50m → ~3-5M pairs → 内存 ~60MB, 可接受
    # 仅当预估 pair 数 > 20M 时回退分批 (防内存爆炸)
    # 真实 domain 已知时优先用其体积估计 pair 密度 (pbc 时 dfn.domain 必存在);
    # 否则退化为数据极值 (仅影响 fast/slow 路径选择, 不改变图正确性)。
    if pbc and dfn.domain is not None:
        domain_vol = float(np.prod(np.asarray(dfn.domain)))
    else:
        domain_vol = np.prod(dfn.centers.max(0) - dfn.centers.min(0) + 1e-6)
    pair_density = min(1.0, (4.0 / 3.0) * np.pi * search_radius ** 3 / (domain_vol + 1e-12))
    est_pairs = int(M * (M - 1) / 2 * pair_density)

    if est_pairs <= 20_000_000:
        # 快速路径: 向量化 query_pairs
        pairs = tree.query_pairs(search_radius, output_type='ndarray')
        if len(pairs) == 0:
            return sparse.csr_matrix((M, M))
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        d = _distance_vec(centers[i_idx], centers[j_idx], pbc, box)
        mask = d < dfn.radii[i_idx] + dfn.radii[j_idx]
        i_idx, j_idx = i_idx[mask], j_idx[mask]
        rows = np.concatenate([i_idx, j_idx])
        cols = np.concatenate([j_idx, i_idx])
        data = np.ones(len(rows), dtype=np.float32)
        return sparse.csr_matrix((data, (rows, cols)), shape=(M, M))

    # 超大网络回退: 逐批 query_ball_point (内存友好)
    from scipy.sparse import lil_matrix
    G = lil_matrix((M, M), dtype=np.float32)

    for i_start in range(0, M, batch_size):
        i_end = min(i_start + batch_size, M)
        batch_centers = centers[i_start:i_end]
        neighbor_lists = tree.query_ball_point(batch_centers, search_radius)
        for local_i, neighbors in enumerate(neighbor_lists):
            i = i_start + local_i
            if not neighbors:
                continue
            neighbors = np.array(neighbors, dtype=np.intp)
            neighbors = neighbors[neighbors > i]
            if len(neighbors) == 0:
                continue
            d = _distance_vec(centers[neighbors], centers[i], pbc, box)
            r_sum = dfn.radii[i] + dfn.radii[neighbors]
            mask = d < r_sum
            hit = neighbors[mask]
            if len(hit) > 0:
                G[i, hit] = 1.0
                G[hit, i] = 1.0

    return G.tocsr()


# ---------------------------------------------------------------------------
# 验证工具
# ---------------------------------------------------------------------------

def compute_p32(dfn: DFNRealization, domain: Tuple[float, float, float]) -> float:
    """计算实现的 P32 = Σ π r² / V。"""
    V = domain[0] * domain[1] * domain[2]
    if V <= 0 or dfn.n_fractures == 0:
        return 0.0
    return float(np.sum(np.pi * dfn.radii ** 2) / V)


def set_table_from_net(net: dict) -> SetTable:
    """从带 set_ids 的 net dict 提取 SetTable。

    net 需含:
      - 'nrm_full' 或 'nrm': (N, 3) 法向
      - 'set_ids': (N,) 组标签 (int, 0..K-1)

    返回 SetTable (centers = 符号对齐组心, concentrations = 组内离散度的逆,
    proportions = 各组占比)。
    """
    nrm = np.asarray(net.get("nrm_full", net.get("nrm")), dtype=float)
    set_ids = np.asarray(net["set_ids"], dtype=int)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)

    K = int(set_ids.max()) + 1
    centers = np.zeros((K, 3))
    concentrations = np.zeros(K)
    proportions = np.zeros(K)

    for k in range(K):
        mask = set_ids == k
        proportions[k] = mask.sum() / len(set_ids)
        if mask.sum() == 0:
            centers[k] = np.array([1, 0, 0], dtype=float)
            concentrations[k] = 1.0
            continue
        cluster_nrms = nrm[mask]
        # 符号对齐均值
        ref = cluster_nrms[0]
        sgn = np.sign((cluster_nrms * ref).sum(-1, keepdims=True))
        sgn[sgn == 0] = 1
        cluster_nrms = cluster_nrms * sgn
        mean_dir = cluster_nrms.mean(0)
        mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-12)
        centers[k] = mean_dir
        # 浓度: κ ≈ 1/σ² (σ in radians, 用组内平均角距)
        cos_angles = np.clip(np.abs(cluster_nrms @ mean_dir), 0, 1)
        angles = np.degrees(np.arccos(cos_angles))
        sigma_rad = np.radians(max(np.mean(angles), 1.0))
        concentrations[k] = min(1.0 / (sigma_rad ** 2 + 0.01), 100.0)

    return SetTable(centers=centers, concentrations=concentrations,
                    proportions=proportions)


def set_table_from_normals(nrm: np.ndarray, K: int = 4, seed: int = 42) -> tuple:
    """从法向数组直接球面 k-means 聚类产生 SetTable。

    返回 (SetTable, set_ids)。
    """
    nrm = np.asarray(nrm, dtype=float)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    N = len(nrm)
    rng = np.random.default_rng(seed)
    # 初始化: 随机选 K 个方向
    idx = rng.choice(N, min(K, N), replace=False)
    centers = nrm[idx].copy()

    set_ids = np.zeros(N, dtype=int)
    for _ in range(30):
        cos_sim = np.abs(nrm @ centers.T)
        new_ids = cos_sim.argmax(1)
        if np.array_equal(new_ids, set_ids) and _ > 0:
            break
        set_ids = new_ids
        for k in range(K):
            mask = set_ids == k
            if mask.sum() > 0:
                cluster_nrms = nrm[mask]
                ref = cluster_nrms[0]
                sgn = np.sign((cluster_nrms * ref).sum(-1, keepdims=True))
                sgn[sgn == 0] = 1
                mean_dir = (cluster_nrms * sgn).mean(0)
                centers[k] = mean_dir / (np.linalg.norm(mean_dir) + 1e-12)

    # 提取 SetTable
    net_dummy = {"nrm_full": nrm, "set_ids": set_ids}
    return set_table_from_net(net_dummy), set_ids


def auto_select_K(nrm: np.ndarray, Krange: tuple = (2, 8),
                   seed: int = 42, sample_size: int = 2000,
                   apply_data_volume_cap: bool = True) -> tuple:
    """用 spherical silhouette score 自动选 K, 可选融合数据量安全上限.

    融合规则 (T54):
        K_silhouette = silhouette 选出的最优 K
        K_data = clip(floor(N / C_RECOMMENDED), 2, 12)  (数据安全上限)
        K_final = min(K_silhouette, K_data)  (取小)

    参数:
        nrm: 法向数组 (N, 3)
        Krange: 搜索范围 (Kmin, Kmax)
        seed: 随机种子
        sample_size: 用于计算 silhouette 的最大样本数 (大数据集子采样)
        apply_data_volume_cap: 是否融合数据量安全上限 (默认 True)

    返回:
        (best_K, scores_dict)
        scores_dict 新增 'K_data_cap' / 'K_fusion' 键 (当 apply_data_volume_cap=True)
    """
    from sklearn.metrics import silhouette_score
    N = len(nrm)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)

    # 子采样加速 silhouette 计算
    if N > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, sample_size, replace=False)
        nrm_sample = nrm[idx]
    else:
        nrm_sample = nrm

    scores = {}
    best_K = Krange[0]
    best_score = -1

    for K in range(Krange[0], Krange[1] + 1):
        # 球面 K-means
        rng_k = np.random.default_rng(seed)
        centers = nrm_sample[rng_k.choice(len(nrm_sample), K, replace=False)]
        set_ids = np.zeros(len(nrm_sample), dtype=int)
        for _ in range(30):
            cos_sim = np.abs(nrm_sample @ centers.T)
            new_ids = cos_sim.argmax(1)
            if np.array_equal(new_ids, set_ids) and _ > 0:
                break
            set_ids = new_ids
            for k in range(K):
                mask = set_ids == k
                if mask.sum() > 0:
                    cluster = nrm_sample[mask]
                    ref = cluster[0]
                    sgn = np.sign((cluster * ref).sum(-1, keepdims=True))
                    sgn[sgn == 0] = 1
                    mean_dir = (cluster * sgn).mean(0)
                    centers[k] = mean_dir / (np.linalg.norm(mean_dir) + 1e-12)

        # Spherical distance: 1 - |cos|
        if len(set(set_ids)) < 2:
            continue
        # 用余弦距离矩阵 (预计算)
        cos_mat = np.abs(nrm_sample @ nrm_sample.T)
        dist_mat = np.clip(1 - cos_mat, 0, None)  # clip 防浮点负值
        np.fill_diagonal(dist_mat, 0)
        try:
            # 大数据集用子采样加速 (precomputed 不支持 sample_size)
            n_sil = len(nrm_sample)
            if n_sil > 500:
                rng_sil = np.random.default_rng(seed)
                idx_sil = rng_sil.choice(n_sil, 500, replace=False)
                sil = silhouette_score(dist_mat[np.ix_(idx_sil, idx_sil)],
                                       set_ids[idx_sil], metric='precomputed')
            else:
                sil = silhouette_score(dist_mat, set_ids, metric='precomputed')
        except Exception:
            sil = -1
        scores[K] = float(sil)
        if sil > best_score:
            best_score = sil
            best_K = K

    # --- T54: 数据量安全上限融合 ---
    if apply_data_volume_cap:
        from .adaptive_k import recommend_K
        K_data = recommend_K(N)
        K_fusion = int(np.clip(min(best_K, K_data), 2, 12))
        scores['K_silhouette'] = int(best_K)
        scores['K_data_cap'] = int(K_data)
        scores['K_fusion'] = K_fusion
        scores['fusion_rule'] = f'min(K_sil={best_K}, K_data={K_data})={K_fusion}'
        if K_fusion != best_K:
            scores['fusion_note'] = (
                f'silhouette 推荐 K={best_K} 但数据量安全上限 K_data={K_data} 更保守, '
                f'取 K={K_fusion} (n_obs={N})'
            )
        return K_fusion, scores
    else:
        return best_K, scores


def fit_beta_from_tracelength(trace_lengths: np.ndarray,
                               r_min: float = 0.5,
                               r_max: float = 5.0) -> dict:
    """从迹长数据拟合幂律指数 β (Phase D 数据升档).

    原理: 裂隙半径服从幂律 p(r) ∝ r^(-β), 迹长 l 与半径的关系为
    l ≈ 2r·sin(θ) (θ 为裂隙面与井壁夹角)。在随机方位假设下,
    迹长的分布也是幂律, 指数与半径相同。

    拟合方法: 最大似然估计 (MLE) 对截断幂律。
    Clauset et al. (2009) 方法: β_hat = 1 + n / Σ ln(x_i / x_min)

    参数:
        trace_lengths: 迹长数组 (m)
        r_min, r_max: 截断范围

    返回:
        dict: {beta, beta_std, r_min, r_max, n_samples, method, assumptions}
    """
    tl = np.asarray(trace_lengths, float)
    tl = tl[(tl >= r_min * 0.1) & (tl <= r_max * 2)]  # 过滤异常值
    if len(tl) < 10:
        return {'beta': 3.5, 'beta_std': 0.5, 'n_samples': len(tl),
                'method': 'default (insufficient data)',
                'assumptions': '迹长数据不足, 使用默认 β=3.5'}

    # MLE for power-law exponent (continuous approximation)
    x_min = max(r_min * 0.5, tl.min())
    tl_filtered = tl[tl >= x_min]
    n = len(tl_filtered)
    if n < 5:
        return {'beta': 3.5, 'beta_std': 0.5, 'n_samples': n,
                'method': 'default (insufficient data)',
                'assumptions': '过滤后迹长数据不足, 使用默认 β=3.5'}

    # MLE: β = 1 + n / Σ ln(x_i / x_min)
    log_ratios = np.log(tl_filtered / x_min)
    beta_raw = 1.0 + n / log_ratios.sum()
    # Standard error: SE = (β - 1) / sqrt(n) — 对应未截断的原始 MLE
    beta_std = (beta_raw - 1.0) / np.sqrt(n)

    # Clip to geologically reasonable range
    beta_hat = float(np.clip(beta_raw, 1.5, 5.0))
    clipped = abs(beta_hat - beta_raw) > 1e-9

    return {
        'beta': beta_hat,
        'beta_std': float(beta_std),
        'beta_raw_mle': float(beta_raw),
        'clipped': bool(clipped),
        'r_min': r_min,
        'r_max': r_max,
        'n_samples': n,
        'method': 'MLE (Clauset 2009)' + (' [clipped]' if clipped else ''),
        'assumptions': f'基于 {n} 条迹长 MLE 拟合, 截断 [{r_min:.1f}, {r_max:.1f}]m. '
                      f'迹长≈2r·sin(θ) 在随机方位假设下指数相同.'
                      + (f' ⚠ 原始 MLE={beta_raw:.2f} 超出地质合理域被截断到 {beta_hat:.2f},'
                         f' std 对应原始 MLE.' if clipped else ''),
    }


def estimate_p32_from_spacing(net: dict, domain: tuple) -> dict:
    """从间距数据直接算 P32 (Phase D 数据升档).

    原理: 如果已知裂隙间距 s (同组内沿井轴的平均间距),
    则面密度 P10 = 1/s, P32 ≈ P10 * π * r_mean (Baecher 盘模型).

    参数:
        net: 需含 'spacing' 键 (同组裂隙沿井轴间距, m)
        domain: 域尺寸

    返回:
        dict: {p32, method, assumptions}
    """
    spacing = net.get('spacing')
    if spacing is None:
        return {'p32': None, 'method': 'no spacing data',
                'assumptions': '无间距数据, 回退到计数估计'}

    spacing = np.asarray(spacing, float)
    spacing = spacing[spacing > 0]
    if len(spacing) == 0:
        return {'p32': None, 'method': 'invalid spacing',
                'assumptions': '间距数据无效'}

    s_mean = float(np.mean(spacing))
    p10 = 1.0 / s_mean  # 线密度 (fractures/m)
    # P32 ≈ P10 * π * r_mean (假设 r_mean ≈ 1m 量级)
    r_mean = 1.0  # 默认
    p32 = p10 * np.pi * r_mean * r_mean

    return {
        'p32': p32,
        'p10': p10,
        'spacing_mean': s_mean,
        'method': 'spacing → P10 → P32',
        'assumptions': f'基于间距均值 {s_mean:.2f}m, P10=1/s, P32=P10·π·r² (r≈{r_mean}m)',
    }


def estimate_p32_interval(net: dict, domain: tuple,
                          lith: np.ndarray = None) -> dict:
    """从观测裂隙计数估计 P32 区间。

    编录表场景: 只有井周 dm 级的裂隙交切数, P32 不能直接算。
    返回区间形式 (P10, P50, P90) —— 基于:
      - 观测裂隙数 N_obs
      - 域体积 V
      - 采样体积修正 (井周 dm 级 vs 域 m 级)

    诚实注记: 这是量级估计, 不是精确值。
    """
    nrm = np.asarray(net.get("nrm_full", net.get("nrm")), dtype=float)
    N_total = len(nrm)
    V = domain[0] * domain[1] * domain[2]

    # 如果 net 有 obs_mask, 用观测数反推总数
    obs_mask = net.get("obs_mask")
    if obs_mask is not None:
        obs_mask = np.asarray(obs_mask, bool)
        N_obs = obs_mask.sum()
        obs_frac = N_obs / N_total
    else:
        N_obs = N_total
        obs_frac = 1.0

    # 平均单裂隙面积 (用默认半径的中位数)
    # 对于编录表, 半径未知 → 用经验范围 r ∈ [0.5, 5.0] m
    r_min, r_max = 0.5, 5.0
    # β=3.5 时的平均面积 (近似)
    mean_area = np.pi * (r_min * r_max)  # 几何平均近似

    # P32 区间: 基于 N_obs / obs_frac 反推总裂隙数
    N_est = N_obs / max(obs_frac, 0.1)
    p32_median = N_est * mean_area / V

    # 区间 (考虑到 obs_frac 的不确定性 + 尺寸不确定性)
    p32_p10 = p32_median * 0.2
    p32_p90 = p32_median * 5.0

    return {
        'p32_p10': float(p32_p10),
        'p32_p50': float(p32_median),
        'p32_p90': float(p32_p90),
        'N_obs': int(N_obs),
        'obs_frac': float(obs_frac),
        'domain': list(domain),
        'assumptions': (
            f"半径范围 [{r_min}, {r_max}]m (编录表缺尺寸数据), "
            f"观测比例 {obs_frac:.1%}, 幂律 β≈3.5。 "
            f"P32 为量级估计, 需露头/岩心标定。"
        ),
    }


def compute_set_dispersion(dfn: DFNRealization) -> dict:
    """计算各组内法向角离散度 (度), 用于验证 vMF 采样正确性。"""
    dispersions = {}
    for k in np.unique(dfn.sets):
        sel = dfn.sets == k
        if sel.sum() < 2:
            dispersions[int(k)] = 0.0
            continue
        nrm_k = dfn.normals[sel]
        nrm_k = nrm_k / np.linalg.norm(nrm_k, axis=1, keepdims=True)
        # 符号对齐
        ref = nrm_k[0]
        sgn = np.sign((nrm_k * ref).sum(-1, keepdims=True))
        sgn[sgn == 0] = 1
        nrm_k = nrm_k * sgn
        mean_dir = nrm_k.mean(0)
        mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-12)
        cos_angles = np.clip(np.abs(nrm_k @ mean_dir), 0, 1)
        angles = np.degrees(np.arccos(cos_angles))
        dispersions[int(k)] = float(np.mean(angles))
    return dispersions
