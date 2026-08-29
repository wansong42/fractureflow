# -*- coding: utf-8 -*-
"""T72 锚点 + 不变性测试组.

不依赖任何真实数据, 用"数学真相锚"与对称性验证整条评测链路的正确性。
任何管线喂全随机输入却给出显著偏离锚点的结果 = 链路 bug。
"""
import numpy as np
import pytest


def _angular_mae_deg(pred, true):
    cos = np.clip(np.abs(np.sum(pred * true, axis=1, dtype=float)), 0.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def _rand_unit(n, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _rand_rot(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    return Q


# 数学真实锚: 两独立均匀随机单位向量, |cos θ| ~ Uniform[0,1],
# E[acos(|cos θ|)] = ∫₀¹ arccos(u) du = 1 rad = 57.2958°.
RANDOM_ANCHOR_DEG = 57.29577951308232


def test_random_anchor_matches_truth():
    """全随机 pred vs 全随机 true → 误差必须 ≈ 57.2958°."""
    pred = _rand_unit(50000, 1)
    true = _rand_unit(50000, 2)
    err = _angular_mae_deg(pred, true)
    assert abs(err - RANDOM_ANCHOR_DEG) < 1.5, f"随机锚点 {err:.3f}° 偏离 57.296°"


def test_identity_is_zero():
    true = _rand_unit(1000, 5)
    assert _angular_mae_deg(true, true) < 1e-6


def test_rotation_invariance():
    """同旋转 R 作用于 pred 与 true → 误差不变 (旋转等变)."""
    pred = _rand_unit(5000, 1)
    true = _rand_unit(5000, 2)
    base = _angular_mae_deg(pred, true)
    R = _rand_rot(7)
    rot = _angular_mae_deg(pred @ R.T, true @ R.T)
    assert abs(rot - base) < 1e-6


def test_permutation_invariance():
    """打乱点序 → 误差不变 (置换不变)."""
    pred = _rand_unit(5000, 1)
    true = _rand_unit(5000, 2)
    base = _angular_mae_deg(pred, true)
    perm = np.random.default_rng(3).permutation(5000)
    perm_err = _angular_mae_deg(pred[perm], true[perm])
    assert abs(perm_err - base) < 1e-6


def test_degenerate_k1_modal_err_zero():
    """K=1 时所有点同组, modal_err 必须为 0 (退化用例)."""
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (50, 1))
    # 全局 mode = z 轴; 每点偏差 0
    from fractureflow.l4.evidence import dip_dipdir_to_normal
    # 构造全部同一个走向的平面
    dd = np.full(50, 30.0)
    dip = np.full(50, 45.0)
    normals = dip_dipdir_to_normal(dip, dd)
    # 单组: 符号对齐后的 mode 与自身误差应为 0
    # 用 inference 的 _sign_align / 组 mode 估计
    from fractureflow.inference import _sign_align
    aligned = _sign_align(normals)
    mode = np.mean(aligned, axis=0)
    mode /= np.linalg.norm(mode) + 1e-12
    cos = np.abs(aligned @ mode)
    err = np.degrees(np.arccos(np.clip(cos, 0, 1))).mean()
    # 数值舍入下应≈0 (单组无离散); 阈值 1e-2° 足以捕获"符号未对齐→~90°"类灾难 bug
    assert err < 1e-2, f"单组 modal_err 应为≈0, 实际 {err}"


def test_single_orientation_pointcloud_near_zero():
    """单组近水平: 标签无关 RANSAC/SVD 应给出近 0 隐伏误差 (T30 退化基线).

    不依赖 fractureflow.segmentation, 仅验证几何约定自洽:
    给定全部来自同一平面的点, 拟合法向与真法向偏差 → 0。
    """
    rng = np.random.default_rng(11)
    n = 2000
    # 平面 z = 0, 法向 (0,0,1); 加点噪声
    xy = rng.uniform(-1, 1, size=(n, 2))
    z = rng.normal(0, 1e-3, size=(n, 1))
    pts = np.hstack([xy, z])
    # SVD 拟合法向
    centroid = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vt[2]
    normal = normal * np.sign(normal[2])  # 对齐 +z
    angle = np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1)))
    assert angle < 1.0, f"SVD 拟合单平面法向偏差 {angle:.3f}°"


if __name__ == "__main__":
    test_random_anchor_matches_truth()
    test_identity_is_zero()
    test_rotation_invariance()
    test_permutation_invariance()
    test_degenerate_k1_modal_err_zero()
    test_single_orientation_pointcloud_near_zero()
    print("ALL T72 ANCHOR TESTS PASS")
