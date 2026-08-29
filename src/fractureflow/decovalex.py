# -*- coding: utf-8 -*-
"""DECOVALEX DFN 数据解析器。

解析 dfnWorks 输出的裂隙几何定义文件，得到每条裂隙的法向、中心点、
多边形顶点。用于 Route B 验证——同一裂隙的所有顶点共面且法向一致。

数据格式（dfnWorks v3.x）:
- normal_vectors.dat: 每行一个单位法向量 (nx, ny, nz)，每条裂隙一行
- translations.dat:   每行一个中心点 (x, y, z)，首行是格式说明
- radii_Final.dat:    每行 (xRadius, yRadius, Family#)，首行是说明
- polygons.dat:        首行 nPolygons，之后每行一个多边形:
                       n_verts {x1,y1,z1} {x2,y2,z2} ...
- aperture.dat:        首行标题，之后每行 (id, 0, 0, aperture)
- perm.dat:            首行标题，之后每行 (id, 0, 0, kx, ky, kz)

用法:
    from fractureflow.decovalex import load_dfnworks_dir
    nets = load_dfnworks_dir("data/external/decovalex_dfn/dfn_extra/4frac_benchmarks/4frac_example")
    # nets = [{"pos": ..., "nrm": ..., "fracture_id": ..., "src": "decovalex_4frac"}, ...]
"""

import os
import re
import numpy as np
from typing import List, Dict, Optional


def _parse_polygons(line: str) -> Optional[np.ndarray]:
    """解析多边形顶点行，返回 (n_verts, 3) 数组。"""
    # 格式: n_verts {x,y,z} {x,y,z} ...
    match = re.match(r'^\s*(\d+)\s+(.*)$', line)
    if not match:
        return None
    n_verts = int(match.group(1))
    coords_str = match.group(2)
    # 提取所有 {x,y,z}
    coord_matches = re.findall(r'\{([^}]+)\}', coords_str)
    if len(coord_matches) != n_verts:
        # 尝试备用解析
        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', coords_str)
        if len(nums) != n_verts * 3:
            return None
        verts = np.array([float(x) for x in nums]).reshape(n_verts, 3)
        return verts
    
    verts = []
    for cm in coord_matches:
        coords = [float(x) for x in cm.replace(',', ' ').split()]
        verts.append(coords[:3])
    return np.array(verts)


def load_dfnworks_dir(dir_path: str) -> List[Dict]:
    """加载 dfnWorks 输出目录，返回网络列表。

    每个网络对应一条裂隙，字段:
    - pos: (N, 3) 多边形顶点坐标
    - nrm: (3,) 法向量（所有顶点共享）
    - fracture_id: int (恒等于裂隙索引)
    - center: (3,) 裂隙中心
    - radius: (2,) 长/短半径
    - aperture: float
    - permeability: float
    - src: "decovalex"
    """
    # 读取法向量
    nrm_path = os.path.join(dir_path, 'normal_vectors.dat')
    nrms = []
    with open(nrm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    nrm = [float(x) for x in parts[:3]]
                    nrms.append(nrm)
                except ValueError:
                    continue
    nrms = np.array(nrms)
    n_fractures = len(nrms)

    # 读取中心点（首行是格式说明）
    ctr_path = os.path.join(dir_path, 'translations.dat')
    centers = []
    with open(ctr_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                ctr = [float(x) for x in parts[:3]]
                centers.append(ctr)
            except ValueError:
                continue
    centers = np.array(centers[:n_fractures])

    # 读取半径（首行是说明）
    rad_path = os.path.join(dir_path, 'radii_Final.dat')
    radii = []
    with open(rad_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                r = [float(x) for x in parts[:2]]
                radii.append(r)
            except ValueError:
                continue
    radii = np.array(radii[:n_fractures])

    # 读取多边形顶点
    poly_path = os.path.join(dir_path, 'polygons.dat')
    polygons = []
    with open(poly_path, 'r') as f:
        n_poly_line = f.readline().strip()
        for line in f:
            line = line.strip()
            if not line:
                continue
            verts = _parse_polygons(line)
            if verts is not None:
                polygons.append(verts)

    # 读取孔径（可选）
    ap_path = os.path.join(dir_path, 'aperture.dat')
    apertures = np.zeros(n_fractures)
    if os.path.exists(ap_path):
        with open(ap_path, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines[1:], 0):
            if i >= n_fractures:
                break
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    apertures[i] = float(parts[3])
                except ValueError:
                    pass

    # 构建网络列表
    nets = []
    for i in range(n_fractures):
        if i < len(polygons):
            pos = polygons[i]
        else:
            # 无多边形数据时用中心点生成一个小正方形
            c = centers[i] if i < len(centers) else np.zeros(3)
            r = radii[i] if i < len(radii) else np.array([100.0, 100.0])
            # 局部坐标
            u = np.array([1.0, 0.0, 0.0])
            v = np.array([0.0, 1.0, 0.0])
            # 确保 u, v 垂直于法向
            n = nrms[i]
            u = u - np.dot(u, n) * n
            v = v - np.dot(v, n) * n
            if np.linalg.norm(u) < 1e-6:
                u = np.array([0.0, 1.0, 0.0])
                u = u - np.dot(u, n) * n
            u = u / (np.linalg.norm(u) + 1e-12)
            v = v / (np.linalg.norm(v) + 1e-12)
            corners = [
                c + r[0] * u + r[1] * v,
                c - r[0] * u + r[1] * v,
                c - r[0] * u - r[1] * v,
                c + r[0] * u - r[1] * v,
            ]
            pos = np.array(corners)
        
        net = {
            'pos': pos.astype(np.float32),
            'nrm': nrms[i].astype(np.float32),
            'fracture_id': i,
            'center': centers[i].astype(np.float32) if i < len(centers) else np.zeros(3, dtype=np.float32),
            'radius': radii[i].astype(np.float32) if i < len(radii) else np.zeros(2, dtype=np.float32),
            'aperture': float(apertures[i]),
            'src': 'decovalex',
        }
        nets.append(net)

    return nets


def generate_survey_points(nets, rng, n_obs=20):
    """从每条裂隙的多边形顶点中随机采样观测点。

    返回 (all_pos, all_nrm, all_fid, obs_mask):
    - all_pos: (total_points, 3) 所有顶点坐标
    - all_nrm: (total_points, 3), 各点所属裂隙法向
    - all_fid: (total_points,) int, 各点所属 fracture_id
    - obs_mask: (total_points,) bool, True 表示观测点
    """
    all_pos = []
    all_nrm = []
    all_fid = []
    for net in nets:
        pos = net['pos']
        nrm = net['nrm']
        fid = net['fracture_id']
        for p in pos:
            all_pos.append(p)
            all_nrm.append(nrm)
            all_fid.append(fid)

    all_pos = np.array(all_pos)
    all_nrm = np.array(all_nrm)
    all_fid = np.array(all_fid)

    n_total = len(all_pos)
    n_obs = min(n_total, n_obs)  # 每裂隙最多采 n_obs 个点

    # 每裂隙独立采样 (保证每裂隙都有观测点)
    obs_mask = np.zeros(n_total, dtype=bool)
    idx = 0
    for net in nets:
        nf = len(net['pos'])
        ns = min(n_obs, nf)
        perm = rng.permutation(nf)[:ns]
        obs_mask[idx + perm] = True
        idx += nf

    return all_pos, all_nrm, all_fid, obs_mask


if __name__ == '__main__':
    # Quick test
    import json
    
    base = "data/external/decovalex_dfn/dfn_extra/4frac_benchmarks"
    
    print("=== 4frac_example ===")
    nets1 = load_dfnworks_dir(os.path.join(base, "4frac_example"))
    print(f"Loaded {len(nets1)} fractures")
    for i, net in enumerate(nets1):
        print(f"  F{i}: pos={net['pos'].shape}, nrm={net['nrm']}, center={net['center']}")
    
    print("\n=== 4fracplus_example ===")
    nets2 = load_dfnworks_dir(os.path.join(base, "4fracplus_example"))
    print(f"Loaded {len(nets2)} fractures")
    print(f"  F0: pos={nets2[0]['pos'].shape}, nrm={nets2[0]['nrm']}")
    print(f"  F1: pos={nets2[1]['pos'].shape}, nrm={nets2[1]['nrm']}")
    print(f"  ...")
    print(f"  F{len(nets2)-1}: pos={nets2[-1]['pos'].shape}")
