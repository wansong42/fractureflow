# -*- coding: utf-8 -*-
"""T54: 自适应 K 梯度规则 —— 数据量驱动 + 结构探测融合选 K.

核心规则 (数据量梯度):
    K_rec = clip(floor(n_obs / C_RECOMMENDED), 2, 12)

其中 C_RECOMMENDED = 11.0 由 loaded_real 30 井拟合 (拟合/验证数据分离),
在 beishan/FORGE 上验证不劣于全局固定最优 K 超 0.5°.

融合规则 (与 dfn.auto_select_K 协同):
    K_silhouette = silhouette 定结构上限 (已有)
    K_data = floor(n_obs / C_RECOMMENDED) 数据安全上限 (本模块)
    K_final = clip(min(K_silhouette, K_data), 2, 12)

类型感知 (与 T51 一致):
    若有 ftype 列 (natural/induced):
        natural 子集 K = min(K_data, 12)
        induced 子集 K = min(K_data, 4)
    T51 实证: induced K=4 显著优于 K=12 (过拟合抑制).

拟合数据来源 (与验证数据严格分离):
    拟合: loaded_real 30 井 (results/p0_full_audit.json K 网格)
    验证: beishan 22 井, FORGE 双井 FMI
    结果: C=11.0 使 auto-K 在 beishan/FORGE 上 modal_err 不劣于固定最优 K 超 0.5°

参照: docs/技术路线_七期_类型感知泛化与国际对标.md (T54).

用法:
    from fractureflow.adaptive_k import recommend_K, auto_K_with_fusion
    K = recommend_K(n_obs=40)  # → clip(floor(40/11), 2, 12) = 3
    K_final, info = auto_K_with_fusion(nrm, occ=None)
"""

import numpy as np

# ---------------------------------------------------------------------------
# 核心参数 (由 loaded_real 拟合, 不改)
# ---------------------------------------------------------------------------
C_RECOMMENDED = 11.0  # n_obs _per_group 安全阈值 (拟合得到, 不自调)
K_MIN = 2
K_MAX = 12  # 铁律: K>12 进文档需架构师书面批准


def recommend_K(n_obs: int, c: float = C_RECOMMENDED,
                 k_min: int = K_MIN, k_max: int = K_MAX) -> int:
    """数据量梯度规则: K_rec = clip(floor(n_obs / c), k_min, k_max).

    参数:
        n_obs: 观测点数 (观测裂隙数)
        c: 每组期望观测数阈值 (默认 11.0)
        k_min: 下限 (默认 2)
        k_max: 上限 (默认 12)

    返回:
        推荐 K 值
    """
    if n_obs < k_min:
        return k_min
    k_rec = int(np.floor(n_obs / c))
    return int(np.clip(k_rec, k_min, k_max))


def auto_K_with_fusion(nrm: np.ndarray, occ: np.ndarray = None,
                       Krange: tuple = (2, 8),
                       seed: int = 42,
                       sample_size: int = 2000,
                       type_labels: np.ndarray = None) -> tuple:
    """融合选 K: silhouette 结构上限 + 数据安全上限, 取小.

    与 dfn.auto_select_K 融合: silhouette 定结构上限, 数据量定安全上限, 取小.
    一致性: 与 T51 induced K=4 逻辑保持一致 (type_labels 可用时).

    参数:
        nrm: (N, 3) 法向数组
        occ: (N,) bool 观测掩码 (None = 全量)
        Krange: silhouette 搜索范围
        seed: 随机种子
        sample_size: silhouette 子采样数
        type_labels: (N,) 类型标签 (可选, "natural"/"induced")

    返回:
        (K_final, info_dict)
        info_dict = {
            'K_silhouette': int,
            'K_data': int,
            'K_final': int,
            'n_obs': int,
            'n_per_group_expected': float,
            'type_aware': bool,
            'per_type': dict or None,  # 类型感知时各类型的 K
        }
    """
    from .dfn import auto_select_K

    nrm = np.asarray(nrm, dtype=np.float64)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)

    if occ is not None:
        occ = np.asarray(occ, dtype=bool)
        n_obs = int(occ.sum())
        nrm_obs = nrm[occ]
    else:
        n_obs = len(nrm)
        nrm_obs = nrm

    # 1. Silhouette 结构上限
    K_sil, sil_scores = auto_select_K(nrm_obs, Krange=Krange,
                                       seed=seed, sample_size=sample_size)

    # 2. 数据安全上限
    K_data = recommend_K(n_obs)

    # 3. 融合: 取小
    K_final = min(K_sil, K_data)
    K_final = int(np.clip(K_final, K_MIN, K_MAX))

    # 4. 类型感知 (T51 一致性)
    per_type = None
    type_aware = False
    if type_labels is not None:
        type_labels = np.asarray(type_labels)
        per_type = {}
        for tname in ["natural", "induced"]:
            mask = type_labels == tname if occ is None else (type_labels == tname) & occ
            n_t = int(mask.sum()) if mask.dtype == bool else 0
            if n_t > 0:
                if tname == "induced":
                    # T51: induced K=4 上限 (过细分导致过拟合)
                    per_type[tname] = min(recommend_K(n_t), 4)
                else:
                    per_type[tname] = recommend_K(n_t)
                type_aware = True

    info = {
        'K_silhouette': int(K_sil),
        'K_data': int(K_data),
        'K_final': int(K_final),
        'n_obs': n_obs,
        'n_per_group_expected': round(n_obs / max(K_final, 1), 1),
        'silhouette_scores': sil_scores if isinstance(sil_scores, dict) else {},
        'type_aware': type_aware,
        'per_type': per_type,
        'rule': f'K_rec = clip(floor(n_obs / {C_RECOMMENDED}), {K_MIN}, {K_MAX})',
        'fusion': 'min(K_silhouette, K_data)',
    }
    return K_final, info


def auto_K_simple(n_obs: int, structure_K: int = None) -> int:
    """简化版 auto-K: 纯数据量规则, 可选与结构 K 取小.

    用于 CLI --auto-K 快速路径 (不需 silhouette 计算时).

    参数:
        n_obs: 观测点数
        structure_K: 结构探测推荐 K (可选)

    返回:
        推荐 K
    """
    k_data = recommend_K(n_obs)
    if structure_K is not None:
        return int(np.clip(min(k_data, structure_K), K_MIN, K_MAX))
    return k_data
