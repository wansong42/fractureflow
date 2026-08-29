# -*- coding: utf-8 -*-
"""Terzaghi 权重方向单测 — 防权重方向反向 bug 复发 (八期 T59).

验证核心物理: 垂直面 (几乎钻不到) 应升权, 水平面 (容易钻遇) 应降权.
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fractureflow.terzaghi import terzaghi_weights


def test_vertical_gt_horizontal():
    """核心物理: 垂直面权重应 > 水平面权重."""
    n = np.array([
        [0.0, 0.0, 1.0],   # 水平面 (容易钻遇)
        [1.0, 0.0, 0.0],   # 垂直面 (几乎钻不到)
    ])
    w = terzaghi_weights(n)
    assert w[1] > w[0], f"垂直面 ({w[1]:.3f}) 应 > 水平面 ({w[0]:.3f})"


def test_horizontal_below_mean():
    """水平面应低于均值 (w < 1)."""
    n = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    w = terzaghi_weights(n)
    assert w[0] < 1.0, f"水平面应低于均值, got {w[0]:.3f}"


def test_vertical_above_mean():
    """垂直面应高于均值 (w > 1)."""
    n = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    w = terzaghi_weights(n)
    assert w[1] > 1.0, f"垂直面应高于均值, got {w[1]:.3f}"


def test_mean_normalized():
    """无裁剪时权重均值应归一化到 1.0."""
    rng = np.random.RandomState(42)
    n = rng.randn(100, 3)
    n[:, 2] = np.abs(n[:, 2]) + 0.5  # 确保 |n·a| 不太小, 避免裁剪
    w = terzaghi_weights(n)
    assert abs(w.mean() - 1.0) < 1e-6, f"均值应 = 1.0, got {w.mean()}"


def test_clipping_range():
    """权重应裁剪到 [0.1, 5.0]."""
    n = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.001],  # 几乎垂直面
        [0.0, 1.0, 0.001],
    ])
    w = terzaghi_weights(n)
    assert w.min() >= 0.1 - 1e-6, f"最小权重应 >= 0.1, got {w.min()}"
    assert w.max() <= 5.0 + 1e-6, f"最大权重应 <= 5.0, got {w.max()}"


def test_intermediate_angle():
    """无极端采样: 中间角度权重应在水平和陡倾之间."""
    # 只使用非极端方向 (避免 1e6 主导归一化)
    angles = np.array([20, 30, 45, 60, 70, 80])  # 度, 相对于井轴
    n = np.column_stack([
        np.sin(np.deg2rad(angles)),
        np.zeros_like(angles),
        np.cos(np.deg2rad(angles)),
    ])
    w = terzaghi_weights(n)
    # 角度越大 (越接近垂直面) 权重越高
    # 20° < 30° < 45° < 60° < 70° < 80°
    for i in range(len(w) - 1):
        assert w[i] < w[i+1], \
            f"角度 {angles[i]}° 权重 ({w[i]:.3f}) 应 < {angles[i+1]}° 权重 ({w[i+1]:.3f})"


if __name__ == "__main__":
    tests = [
        test_vertical_gt_horizontal,
        test_horizontal_below_mean,
        test_vertical_above_mean,
        test_mean_normalized,
        test_clipping_range,
        test_intermediate_angle,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
