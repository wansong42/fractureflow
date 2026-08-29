# -*- coding: utf-8 -*-
"""T66-B8: κ 公式分位单测.

防"记错公式": 验证 vMF (d=3) 的 CDF 闭式 + 截断指数精确采样 + 理论 95 分位
三者自洽. 如果任何一个公式记错 (CDF 形式 / 采样器 / 角度定义),
本测试会暴露.

核心发现 (B8 验证):
  bootstrap_confidence_cone 度量的是「中心的角度不确定度」(center uncertainty),
  而非「包含 95% 数据的锥半角」. 后者由 vMF CDF 直接给出 (θ95).
  本测试验证 CDF 公式与实测经验分位一致 (<0.5°).

vMF (d=3) 精确数学:
  - z = cos(θ) 边际 = 截断指数: f(z) ∝ exp(κz), z ∈ [-1, 1]
  - 精确采样: z = (1/κ) ln(exp(-κ) + u·(exp(κ) - exp(-κ))), u ~ Uniform(0,1)
  - CDF(θ) = (exp(κ) - exp(κ·cos θ)) / (exp(κ) - exp(-κ))

参考: Mardia & Jupp 2000, §9.3.2 (d=3 截断指数精确采样).
"""

import os
import sys
import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fractureflow.l4.fuse import bootstrap_confidence_cone


def _sample_vmf(kappa: float, n: int, seed: int = 42) -> np.ndarray:
    """vMF (d=3) 精确采样, 中心 = z 轴."""
    rng = np.random.default_rng(seed)
    if kappa < 1e-8:
        xyz = rng.normal(size=(n, 3))
        return xyz / np.linalg.norm(xyz, axis=1, keepdims=True)

    u = rng.uniform(size=n)
    exp_k = np.exp(kappa)
    exp_neg_k = np.exp(-kappa)
    z = (1.0 / kappa) * np.log(exp_neg_k + u * (exp_k - exp_neg_k))

    phi = rng.uniform(0, 2 * np.pi, size=n)
    r = np.sqrt(np.maximum(0, 1 - z ** 2))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _vmf_cdf(kappa: float, theta: float) -> float:
    """vMF CDF (d=3): P(angle <= theta)."""
    if kappa < 1e-8:
        return (1 - np.cos(theta)) / 2
    num = np.exp(kappa) - np.exp(kappa * np.cos(theta))
    den = np.exp(kappa) - np.exp(-kappa)
    return num / den


def _vmf_theta95(kappa: float) -> float:
    """求 vMF 理论 95 分位半角 θ95 (度). 二分查找."""
    lo, hi = 0.0, np.pi
    for _ in range(80):
        mid = (lo + hi) / 2
        if _vmf_cdf(kappa, mid) < 0.95:
            lo = mid
        else:
            hi = mid
    return np.degrees((lo + hi) / 2)


def test_vmf_cdf_self_consistent():
    """B8 核心: vMF CDF 在端点取值正确."""
    assert abs(_vmf_cdf(30.0, np.pi) - 1.0) < 1e-10
    assert abs(_vmf_cdf(30.0, 0.0)) < 1e-10
    # 关键分位点交叉验证 (θ 增大 CDF 单调增, 用宽松容差避免浮点抖动)
    prev = -1.0
    for deg in [5, 15, 25, 35, 45, 55, 65, 75, 85]:
        cur = _vmf_cdf(30.0, np.radians(deg))
        assert cur > prev, f"CDF 在 {deg}° 处非单调: {cur} vs prev {prev}"
        prev = cur


def test_empirical_quantile_matches_formula():
    """B8 核心: 经验 95 分位 vs vMF 理论 θ95, 差 < 0.5°.

    防"记错公式": 采样器 + CDF 有一处错就不一致.
    n=8000 足以把采样噪声压到 <0.5°.
    """
    for kappa in [10.0, 30.0, 80.0, 150.0]:
        s = _sample_vmf(kappa, n=8000, seed=42)
        angles = np.degrees(np.arccos(np.clip(s[:, 2], -1, 1)))
        emp95 = np.percentile(angles, 95)
        theta95 = _vmf_theta95(kappa)
        diff = abs(emp95 - theta95)
        assert diff < 0.5, (f"κ={kappa}: emp95={emp95:.3f}°, θ95={theta95:.3f}°, "
                            f"diff={diff:.3f}°")


def test_vmf_theta95_monotone():
    """κ 越大 → 数据越集中 → θ95 越小."""
    t95s = [_vmf_theta95(k) for k in [10, 30, 80, 150]]
    assert all(t95s[i] > t95s[i + 1] for i in range(len(t95s) - 1))


def test_vmf_theta95_reasonable_range():
    """θ95 值在合理范围."""
    assert 35.0 < _vmf_theta95(10.0) < 55.0
    assert 20.0 < _vmf_theta95(30.0) < 35.0
    assert 12.0 < _vmf_theta95(80.0) < 20.0
    assert 8.0 < _vmf_theta95(150.0) < 15.0


def test_vmf_sampler_R_matches_theory():
    """vMF 采样器验证: R (平均向量长度) ≈ coth(κ) - 1/κ."""
    for kappa in [10, 30, 80]:
        s = _sample_vmf(kappa, n=10000, seed=99)
        R_emp = np.linalg.norm(np.mean(s, axis=0))
        R_theory = 1.0 / np.tanh(kappa) - 1.0 / kappa
        assert abs(R_emp - R_theory) < 0.03, (f"κ={kappa}: "
                                             f"R_emp={R_emp:.4f}, R_theory={R_theory:.4f}")


def test_bootstrap_cone_monotone_kappa():
    """bootstrap_confidence_cone (center uncertainty): κ 越大 cone 越小."""
    cones = []
    for kappa in [10, 50, 150]:
        samples = _sample_vmf(kappa, n=2000, seed=11)
        cone = bootstrap_confidence_cone(samples, n_bootstrap=1000, seed=3)
        cones.append(cone)
    assert cones[0] > cones[1] > cones[2], (f"κ 增大 cone 应递减: "
                                            f"κ=10→{cones[0]:.3f}°, "
                                            f"κ=50→{cones[1]:.3f}°, "
                                            f"κ=150→{cones[2]:.3f}°")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
