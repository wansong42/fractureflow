# -*- coding: utf-8 -*-
"""可微 Terzaghi 采样偏差校正层 (R185 T2)。

物理与口径
----------
钻孔与裂隙面相交的概率 ∝ sin α = |n·a|, 其中 α = 裂隙面与井轴夹角,
θ = 法向与井轴夹角, |n·a| = cos θ = sin α。因此观测到的法向密度是被**乘性
偏置**过的:  p_obs(n) ∝ p_true(n) · |n·a|。经典 Terzaghi (1965) 校正就是给
每条观测加权重 w_i = 1/|n·a_i|。

一个容易被忽略的关键点 (本模块明确写死)
--------------------------------------
「看不见的裂隙」数量是未知的 —— 我们不能凭空补样本。所以权重必须
**自归一化**:  ṽ_i = w_i / mean(w)。这在统计学上不是 hack, 而是
self-normalized importance sampling: 它正确处理了缺失质量未知的情形,
代价是引入 O(1/N) 的偏差与方差放大。现役 `terzaghi.terzaghi_weights` 里的
`w / w.mean()` 正是这一步, **方向与做法是统计上正确的, 本层不改动它**。

方差放大的可观测后果: 有效样本量

    ESS = (Σw)² / Σw²      (∈ [1, N])

ESS/N 越小, 校正后的估计越不稳。这直接解释了本项目实测「Terzaghi 只有
~0.2–0.3° 增益」—— 权重一旦长尾 (近垂直面权重 5.0 封顶仍是个离群点),
ESS 掉得比样本量快, 校正带来的去偏被方差吃掉。本层把 ESS 作为一等公民输出,
让「这次校正值不值得信」变成可回答的问题。

可微性设计
----------
- 硬裁剪 [0.1, 5.0] 在边界处梯度为 0 (且 np 版是 `np.clip`, 不可导), 这里用
  **soft_clamp**: lo + τ[softplus((x-lo)/τ) - softplus((x-hi)/τ)],
  区间内 ≈ 恒等, 两端渐近到 lo/hi, 全程 C^∞ 可导。
- |n·a| 用 sqrt((n·a)² + ε²) 平滑掉 0 点不可导 (ε 默认 1e-9, 数值上等价)。
- 分布参数走 `bingham_params_from_scatter_t(..., create_graph=True)`:
  加权二阶矩 → `linalg.eigh` (可微) → 浓度牛顿迭代 (torch.where 回溯, 保图)。
  于是梯度可以 **从钻孔轴 a 一路反传到 Bingham 的 (M, λ)**, 这意味着井轴本身
  可以作为待估/待校准参数。

与现役 numpy 实现的关系 (退路③: 冲突只钉不擅改)
----------------------------------------------
本层不修改 `fractureflow/terzaghi.py`。两者差异由 `tests/test_v_r185_.py`
用数值断言钉住:
  * `mode="legacy"` 分支**逐位复现** numpy 的 clip 语义 (不可导, 供对照);
  * `mode="soft"` 是默认 (可微), 与 legacy 的差异只在裁剪邻域, 且差异上界
    被单测量化。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ._sphere import HAS_TORCH
from .bingham import bingham_params_from_scatter_t, LAMBDA_CAP

if HAS_TORCH:
    import torch
    import torch.nn as nn


__all__ = [
    "soft_clamp",
    "terzaghi_weights_t",
    "terzaghi_scatter",
    "terzaghi_ess",
    "censoring_mask",
    "TerzaghiBinghamLayer",
    "W_LO_DEFAULT",
    "W_HI_DEFAULT",
]


W_LO_DEFAULT = 0.1
W_HI_DEFAULT = 5.0


def _smooth_abs(x, eps: float = 1e-9):
    """|x| 的 C^∞ 替代: sqrt(x² + ε²)。ε=1e-9 时数值上与 |x| 无差别。"""
    if HAS_TORCH and torch.is_tensor(x):
        return torch.sqrt(x * x + eps * eps)
    return np.sqrt(x * x + eps * eps)


def soft_clamp(x, lo: float, hi: float, tau: float = 0.25):
    """平滑裁剪: 区间内 ≈ 恒等, 两端渐近到 lo / hi, 全程可微。

        soft_clamp(x) = lo + τ·[softplus((x-lo)/τ) - softplus((x-hi)/τ)]

    x ≫ hi 时 → lo + (x-lo) - (x-hi) = hi;  x ≪ lo 时 → lo。
    τ 越小越贴近硬裁剪 (但梯度越陡); 默认 0.25 在 [0.1,5] 区间上梯度良态。
    """
    if HAS_TORCH and torch.is_tensor(x):
        sp = torch.nn.functional.softplus
        return lo + tau * (sp((x - lo) / tau) - sp((x - hi) / tau))
    sp = lambda z: np.log1p(np.exp(np.clip(z, -60, 60)))  # noqa: E731
    return lo + tau * (sp((x - lo) / tau) - sp((x - hi) / tau))


def terzaghi_weights_t(normals, axis=None, mode: str = "soft",
                       w_lo: float = W_LO_DEFAULT, w_hi: float = W_HI_DEFAULT,
                       tau: float = 0.25, eps: float = 1e-9):
    """可微 Terzaghi 权重 (自归一化, 均值 = 1)。

    normals : (N, 3) torch/numpy 单位法向
    axis    : (3,) 井轴, 默认 [0,0,1] (垂直井);
              **或 (N,3) 逐点井轴** (R211 追加: 弯曲井段的局部切向 —— 某深度的裂隙
              是被该处的井轴钻到的, 故 w_i = 1/|n_i·a_i| 比整段单一轴更贴合物理)。
              一维分支的代码路径一字未动 ⇒ 常向量结果逐位不变 (R185 单测守卫)。
    mode    : "soft"   平滑裁剪 (默认, 可微)
              "legacy" torch.clamp —— 逐位复现 numpy 现役语义 (不可导, 仅对照)
    返回 (N,) 权重。
    """
    if mode not in ("soft", "legacy"):
        raise ValueError(f"terzaghi_weights_t: mode 必须是 soft/legacy, got {mode!r}")
    is_t = HAS_TORCH and torch.is_tensor(normals)
    if is_t:
        n = normals.to(torch.float64)
        if n.ndim != 2 or n.shape[1] != 3:
            raise ValueError(f"terzaghi_weights_t: normals 需 (N,3), got {tuple(n.shape)}")
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        a = (torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) if axis is None
             else torch.as_tensor(axis, dtype=torch.float64))
        if a.ndim == 2:
            if tuple(a.shape) != tuple(n.shape):
                raise ValueError(f"terzaghi_weights_t: 逐点 axis 需与 normals 同形, "
                                 f"got {tuple(a.shape)} vs {tuple(n.shape)}")
            a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cos_wa = _smooth_abs((n * a).sum(dim=-1), eps=eps)
        else:
            a = a / a.norm().clamp_min(1e-12)
            cos_wa = _smooth_abs(n @ a, eps=eps)         # |n·a| = sin α
        w = 1.0 / cos_wa.clamp_min(1e-6)
        w = w / w.mean().clamp_min(1e-12)                # 自归一化 IS
        return w.clamp(w_lo, w_hi) if mode == "legacy" else soft_clamp(w, w_lo, w_hi, tau=tau)

    n = np.asarray(normals, dtype=np.float64)
    if n.ndim != 2 or n.shape[1] != 3:
        raise ValueError(f"terzaghi_weights_t: normals 需 (N,3), got {n.shape}")
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    a = np.array([0.0, 0.0, 1.0]) if axis is None else np.asarray(axis, dtype=np.float64)
    if a.ndim == 2:
        if a.shape != n.shape:
            raise ValueError(f"terzaghi_weights_t: 逐点 axis 需与 normals 同形, "
                             f"got {a.shape} vs {n.shape}")
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        cos_wa = _smooth_abs((n * a).sum(axis=1), eps=eps)
    else:
        a = a / (np.linalg.norm(a) + 1e-12)
        cos_wa = _smooth_abs(n @ a, eps=eps)
    w = 1.0 / np.clip(cos_wa, 1e-6, None)
    w = w / max(float(w.mean()), 1e-12)
    return np.clip(w, w_lo, w_hi) if mode == "legacy" else soft_clamp(w, w_lo, w_hi, tau=tau)


def terzaghi_ess(weights) -> float:
    """有效样本量 ESS = (Σw)²/Σw²。返回 float。"""
    if HAS_TORCH and torch.is_tensor(weights):
        w = weights.detach()
        return float((w.sum() ** 2) / (w * w).sum().clamp_min(1e-300))
    w = np.asarray(weights, dtype=np.float64)
    return float((w.sum() ** 2) / max(float((w * w).sum()), 1e-300))


def terzaghi_scatter(normals, axis=None, weights=None, mode: str = "soft",
                     **kw) -> Tuple:
    """加权二阶矩矩阵 S = Σw·n nᵀ / Σw (迹归一) —— Bingham 的充分统计量。

    **整条 Terzaghi 校正对 Bingham 拟合的影响, 全部浓缩在「把 S 换成加权版」**
    这一件事上 (因 ℓ 只通过 S 依赖数据)。这是本层最干净的结构性结论。

    返回 (S (3,3), w (N,), ess float)。
    """
    is_t = HAS_TORCH and torch.is_tensor(normals)
    if not is_t:
        n = np.asarray(normals, dtype=np.float64)
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        w = terzaghi_weights_t(n, axis=axis, mode=mode, **kw) if weights is None else np.asarray(weights, np.float64)
        S = (w[:, None, None] * n[:, :, None] * n[:, None, :]).sum(0) / w.sum()
        S = 0.5 * (S + S.T)
        return S / np.trace(S), w, terzaghi_ess(w)

    n = torch.as_tensor(normals, dtype=torch.float64)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w = terzaghi_weights_t(n, axis=axis, mode=mode, **kw) if weights is None else weights
    S = torch.einsum("n,ni,nj->ij", w, n, n) / w.sum().clamp_min(1e-12)
    S = 0.5 * (S + S.T)
    S = S / S.diagonal().sum().clamp_min(1e-12)
    return S, w, terzaghi_ess(w)


def censoring_mask(normals, axis=None, alpha_min_deg: float = 10.0):
    """Terzaghi 经典截断判据: 面与井轴夹角 α < α_min 的裂隙几乎钻不到。

    α = 90° - θ, cos θ = |n·a| = sin α  ⇒  可观测 ⟺ |n·a| ≥ sin(α_min)
    """
    is_t = HAS_TORCH and torch.is_tensor(normals)
    n = np.asarray(normals, dtype=np.float64) if not is_t else None
    if not is_t:
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        a = np.array([0.0, 0.0, 1.0]) if axis is None else np.asarray(axis, float)
        a = a / np.linalg.norm(a)
        return np.abs(n @ a) >= np.sin(np.deg2rad(alpha_min_deg))
    nn_ = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    a = torch.as_tensor([0.0, 0.0, 1.0], dtype=normals.dtype,
                        device=normals.device) if axis is None else torch.as_tensor(axis, dtype=normals.dtype)
    a = a / a.norm()
    return torch.abs(nn_ @ a) >= np.sin(torch.deg2rad(torch.tensor(alpha_min_deg, dtype=normals.dtype)))


class TerzaghiBinghamLayer(nn.Module if HAS_TORCH else object):
    """可微 Terzaghi → Bingham 参数层。

    forward(normals) 返回 dict:
        weights  (N,)    自归一化 Terzaghi 权重
        ess      float   有效样本量
        ess_frac float   ESS/N  (< 0.5 时该校正基本不值得信)
        S        (3,3)   加权二阶矩 (充分统计量)
        axes     (3,3)   Bingham 主轴 M (列)
        lmbda    (3,)    Bingham 浓度 (升序, min=0)
    所有张量在 create_graph=True 时对 `normals` 与 `self.axis` 可导。
    """

    def __init__(self, axis=None, learnable_axis: bool = False,
                 mode: str = "soft", w_lo: float = W_LO_DEFAULT,
                 w_hi: float = W_HI_DEFAULT, tau: float = 0.25,
                 alpha_min_deg: Optional[float] = None):
        if not HAS_TORCH:
            raise RuntimeError("TerzaghiBinghamLayer 需要 torch")
        super().__init__()
        a = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) if axis is None else torch.as_tensor(axis, dtype=torch.float64)
        a = a / a.norm()
        if learnable_axis:
            self.axis = nn.Parameter(a.clone())
        else:
            self.register_buffer("axis", a.clone())
        self.mode = mode
        self.w_lo = w_lo
        self.w_hi = w_hi
        self.tau = tau
        self.alpha_min_deg = alpha_min_deg

    def forward(self, normals, weights=None, create_graph: bool = True,
                axis=None):
        """axis 显式传入时覆盖 self.axis —— 便于把井轴当自由变量做梯度检验
        (不要靠重赋 nn.Parameter, 那样容易把计算图弄断)。"""
        a = self.axis if axis is None else axis
        n = torch.as_tensor(normals, dtype=torch.float64)
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        if self.alpha_min_deg is not None:
            keep = censoring_mask(n, a, self.alpha_min_deg)
            n_eff = n[keep]
        else:
            n_eff = n
        w = terzaghi_weights_t(n_eff, a, mode=self.mode,
                               w_lo=self.w_lo, w_hi=self.w_hi, tau=self.tau) \
            if weights is None else weights
        S = torch.einsum("n,ni,nj->ij", w, n_eff, n_eff) / w.sum().clamp_min(1e-12)
        S = 0.5 * (S + S.T)
        S = S / S.diagonal().sum().clamp_min(1e-12)
        lam, M = bingham_params_from_scatter_t(
            S.unsqueeze(0), u_init=None, create_graph=create_graph)
        ess = terzaghi_ess(w)
        return {
            "weights": w,
            "ess": ess,
            "ess_frac": ess / max(int(w.shape[0]), 1),
            "S": S,
            "axes": M[0],
            "lmbda": lam[0],
        }
