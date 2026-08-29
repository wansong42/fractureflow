# -*- coding: utf-8 -*-
"""T68-A 单测: build_connectivity_graph(pbc=True) 不崩 + 返回有效边。

构造 2 组各 50 法向 → set_table → build_connectivity_graph(pbc=True) 不报错且返回边数 ≥0。
使用内置合成数据, 不依赖外部文件。
"""

import numpy as np
import sys
import os

# 确保 project_root 在路径中 (与 test_geometry_conventions.py 同约定)
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TEST_DIR)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SRC_DIR)

from fractureflow.dfn import (SetTable, generate_dfn,
                               build_connectivity_graph, set_table_from_normals)


def make_test_normals():
    """生成 2 组合成法向 (各 50 条), 方向分离 ~90°。"""
    rng = np.random.default_rng(42)
    # 组 1: 近垂直面 (法向近水平, 朝北)
    mu1 = np.array([1.0, 0.0, 0.0])
    # 组 2: 近垂直面 (法向近水平, 朝东)
    mu2 = np.array([0.0, 1.0, 0.0])

    def sample_vmf(mu, kappa, n):
        """简化 vMF 采样 (Wood 1994)。"""
        samples = []
        for _ in range(n):
            # 拒绝采样: 在 S² 上, 绕 mu 旋转
            w = np.log(np.exp(-kappa) + np.random.rand() * (np.exp(kappa) - np.exp(-kappa))) / kappa
            w = np.clip(w, -1, 1)
            phi = rng.uniform(0, 2 * np.pi)
            r2 = 1 - w ** 2
            if r2 < 0:
                r2 = 0
            r = np.sqrt(r2)
            # 构造正交基
            if abs(mu[0]) < 0.9:
                v = np.array([1.0, 0.0, 0.0])
            else:
                v = np.array([0.0, 1.0, 0.0])
            e1 = np.cross(mu, v)
            e1 = e1 / np.linalg.norm(e1)
            e2 = np.cross(mu, e1)
            e2 = e2 / np.linalg.norm(e2)
            s = w * mu + r * np.cos(phi) * e1 + r * np.sin(phi) * e2
            s = s / np.linalg.norm(s)
            samples.append(s)
        return np.array(samples)

    nrm1 = sample_vmf(mu1, 20.0, 50)
    nrm2 = sample_vmf(mu2, 20.0, 50)
    return np.vstack([nrm1, nrm2])


def test_pbc_no_crash():
    """测试 pbc=True 时 build_connectivity_graph 不崩溃。"""
    nrm = make_test_normals()
    st, _ = set_table_from_normals(nrm, K=2, seed=42)

    dfn = generate_dfn(st, p32=2.0, beta=3.5, domain=(50, 50, 50), seed=42)
    assert dfn.n_fractures > 0, "生成的 DFN 裂隙数应为正"
    # 关键: pbc=True 不应崩溃
    G = build_connectivity_graph(dfn, pbc=True)
    assert G.shape == (dfn.n_fractures, dfn.n_fractures), "邻接矩阵形状应匹配"
    # 边数 ≥ 0
    n_edges = G.nnz // 2  # 对称矩阵, 非对角线元素计一次
    assert n_edges >= 0, f"边数应 ≥ 0, 实际 {n_edges}"
    print(f"[PASS] test_pbc_no_crash: {dfn.n_fractures} 裂隙, {n_edges} 边")


def test_pbc_false_still_works():
    """确认 pbc=False (默认) 不受影响。"""
    nrm = make_test_normals()
    st, _ = set_table_from_normals(nrm, K=2, seed=42)
    dfn = generate_dfn(st, p32=2.0, beta=3.5, domain=(50, 50, 50), seed=42)
    G = build_connectivity_graph(dfn, pbc=False)
    assert G.shape == (dfn.n_fractures, dfn.n_fractures)
    print(f"[PASS] test_pbc_false_still_works: shape={G.shape}")


def test_pbc_cross_boundary_connectivity():
    """P1 修复: 验证 PBC 确实创建了跨边界连通 (不仅是"不崩").

    构造场景: 角点锚裂隙 (跨越完整 [-L/2,L/2]³ 以保证 boxsize=L) +
    2 个目标裂隙分别靠近 x 方向两边界 (x ≈ -L/2+ε 与 x ≈ L/2-ε).
    断言: pbc=True 时 G[0,1] = 1 (跨边界连通); pbc=False 时 G[0,1] = 0.
    """
    from fractureflow.dfn import DFNRealization
    L = 10.0
    eps = 0.3
    r = 1.0  # 圆盘半径
    # 锚裂隙: 8 个角点, 确保 coord_min=[-L/2,-L/2,-L/2], coord_max=[L/2,L/2,L/2]
    corners = np.array([
        [-L/2, -L/2, -L/2], [-L/2, -L/2, L/2], [-L/2, L/2, -L/2], [-L/2, L/2, L/2],
        [L/2, -L/2, -L/2], [L/2, -L/2, L/2], [L/2, L/2, -L/2], [L/2, L/2, L/2]
    ])
    # 目标裂隙: 分别靠近左右 x 边界, 相同 y,z 使跨边界距离最短
    c0 = np.array([-L/2 + eps, 0.0, 0.0])
    c1 = np.array([L/2 - eps, 0.0, 0.0])
    centers = np.vstack([c0, c1, corners])
    n = len(centers)
    normals = np.tile([1, 0, 0], (n, 1))
    radii = np.full(n, r)
    sets = np.arange(n) % 2
    dfn = DFNRealization(centers=centers, normals=normals, radii=radii, sets=sets,
                         domain=(L, L, L))
    # PBC 路径: 应连通 (跨边界距离 2*eps=0.6 < 2r=2.0)
    G_pbc = build_connectivity_graph(dfn, pbc=True)
    assert G_pbc[0, 1] == 1, (f"PBC 下跨边界裂隙应连通, 但 G[0,1]={G_pbc[0,1]}; "
                               f"跨边界距离={2*eps}, r={r}")
    # 非 PBC 路径: 不应连通 (非周期距离 L-2eps=9.4 > 2r=2.0)
    G_no = build_connectivity_graph(dfn, pbc=False)
    assert G_no[0, 1] == 0, (f"非 PBC 下跨边界裂隙不应连通, 但 G[0,1]={G_no[0,1]}")
    print(f"[PASS] test_pbc_cross_boundary_connectivity: "
          f"pbc=True→{G_pbc[0,1]}, pbc=False→{G_no[0,1]}")


if __name__ == "__main__":
    test_pbc_no_crash()
    test_pbc_false_still_works()
    test_pbc_cross_boundary_connectivity()
    print("ALL PASS")
