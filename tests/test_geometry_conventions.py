# -*- coding: utf-8 -*-
"""几何约定单测 — 防 dip_dipdir_to_normal 类 bug 复发 (八期 T58).

覆盖:
  1. dip=0 (水平面) -> 法向竖直 (nU=±1, 水平分量=0)
  2. dip=90 (垂直面) -> 法向水平 (nU=0, 水平分量=1)
  3. dip=45 -> |nz|=cos45≈0.707
  4. dip_dir=0 (北) -> nN>0 (法向水平投影指北)
  5. dip_dir=90 (东) -> nE>0
  6. dip_dir=180 (南) -> nN<0
  7. dip_dir=270 (西) -> nE<0
  8. 重建 .pt 的 implied dip 与 raw dip 一致 (血统锚点)
  9. local_frames 旋转等变性 (deterministic_e2=True, 含反射) — BUG-2 回归
 10. fracture_aware_dirs 含无观测裂隙不产出零向量 — BUG-1 回归
"""
import os
import sys
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from scripts.read_forge_las import dip_dipdir_to_normal


def test_horizontal_plane():
    """dip=0 -> 法向竖直."""
    n = dip_dipdir_to_normal(np.array([0.0]), np.array([0.0]))
    assert abs(abs(n[0, 2]) - 1.0) < 1e-6, f"水平面法向 nU 应为 ±1,  got {n[0, 2]}"
    assert abs(n[0, 0]) < 1e-6 and abs(n[0, 1]) < 1e-6, "水平面法向水平分量应为 0"


def test_vertical_plane():
    """dip=90 -> 法向水平 (nU=0)."""
    n = dip_dipdir_to_normal(np.array([0.0]), np.array([90.0]))
    assert abs(n[0, 2]) < 1e-6, f"垂直面法向 nU 应为 0, got {n[0, 2]}"
    assert abs(n[0, 0]) < 1e-6 and abs(n[0, 1] - 1.0) < 1e-6, "垂直面 dip_dir=0 法向应指北"


def test_45_degree():
    """dip=45 -> |nz|=cos45≈0.707."""
    n = dip_dipdir_to_normal(np.array([0.0]), np.array([45.0]))
    assert abs(abs(n[0, 2]) - np.cos(np.deg2rad(45))) < 1e-6, \
        f"dip=45 时 |nz| 应为 cos45, got {abs(n[0, 2])}"


def test_dip_direction_north():
    """dip_dir=0 -> nN>0."""
    n = dip_dipdir_to_normal(np.array([0.0]), np.array([60.0]))
    assert n[0, 1] > 0, f"dip_dir=0 法向应指北 (nN>0), got {n[0, 1]}"


def test_dip_direction_east():
    """dip_dir=90 -> nE>0."""
    n = dip_dipdir_to_normal(np.array([90.0]), np.array([60.0]))
    assert n[0, 0] > 0, f"dip_dir=90 法向应指东 (nE>0), got {n[0, 0]}"


def test_dip_direction_south():
    """dip_dir=180 -> nN<0."""
    n = dip_dipdir_to_normal(np.array([180.0]), np.array([60.0]))
    assert n[0, 1] < 0, f"dip_dir=180 法向应指南 (nN<0), got {n[0, 1]}"


def test_dip_direction_west():
    """dip_dir=270 -> nE<0."""
    n = dip_dipdir_to_normal(np.array([270.0]), np.array([60.0]))
    assert n[0, 0] < 0, f"dip_dir=270 法向应指西 (nE<0), got {n[0, 0]}"


def test_forge_16b_lineage_anchor():
    """重建后的 16B 数据: arccos|nz| 应等于 raw dip (血统锚点, 排除补角 bug)."""
    pt_path = os.path.join(_ROOT, "data/external/utah_forge_fmi/forge_fmi_2wells.pt")
    if not os.path.exists(pt_path):
        print("  [SKIP] forge_fmi_2wells.pt not found")
        return
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    for n in d["nets"]:
        if n["wid"] == "16B":
            nrm = n["nrm"]
            nz = np.abs(nrm[:, 2])
            implied = np.degrees(np.arccos(np.clip(nz, 0, 1)))
            raw = n["dip"]
            diff = np.median(np.abs(implied - raw))
            assert diff < 0.01, \
                f"16B 血统锚点失败: |arccos|nz| - raw_dip| 中位差 {diff:.4f}° (应 <0.01°)"
            print(f"  [PASS] 16B lineage anchor: median |implied - raw| = {diff:.6f}°")
            return
    print("  [SKIP] 16B not in dataset")


def test_local_frames_rotation_equivariance():
    """BUG-2 回归: local_frames 输出标架必须旋转等变 (含反射).

    T82 真因修复: 旧实现把 eigh 输出 ([分量,特征向量]) 当作 [向量,分量] 用,
    重排/读出全在分量轴上 —— 标架实为特征向量的任意置换混合, 旋转输入后
    逐点帧偏差可达 76°。修正后输出 [B,L,分量,向量] (列=标架向量),
    对任意正交 R 满足 |<R@col_j(V(x)), col_j(V(Rx))>| ≈ 1 (逐对).
    """
    from fractureflow.geometry import local_frames
    rng = np.random.default_rng(7)
    B, L, K = 2, 24, 8
    # 确定性各向异性邻域 (椭球轴点模式): 协方差特征值严格分离,
    # 排除近简并特征值下 eigh 基底任意性造成的假失败; 微噪声防完全共面。
    base_pts = np.array([
        [3.0, 0.0, 0.0], [-3.0, 0.0, 0.0],
        [0.0, 2.0, 0.0], [0.0, -2.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
        [0.7, 0.5, 0.2], [-0.6, 0.4, -0.3]], dtype=np.float64)
    centers = rng.normal(size=(B, L, 3)) * 5.0
    knn_pos = (centers[:, :, None, :]
               + base_pts[None, None]
               + rng.normal(size=(B, L, K, 3)) * 1e-3).astype(np.float32)

    for trial in range(4):
        A = rng.normal(size=(3, 3))
        Q, _ = np.linalg.qr(A)
        if trial % 2 == 1 and np.linalg.det(Q) > 0:
            Q[:, 0] *= -1.0
        R = torch.tensor(Q, dtype=torch.float32)
        kp1 = torch.tensor(knn_pos)
        kp2 = (kp1 @ R.T)
        # legacy=True 是默认 (冻结行为); 等变性只在修正版 (legacy=False) 上要求
        v1, _ = local_frames(kp1, K, deterministic_e2=True, legacy=False)
        v2, _ = local_frames(kp2, K, deterministic_e2=True, legacy=False)
        # 列约定: v[b,l,c,d] 第 d 列 = 标架向量. 旋转各列后逐对比较.
        rv1 = torch.einsum("ef,blfg->bleg", R, v1)
        cosv = (rv1 * v2).sum(-2).abs().clamp(0, 1)
        ang = torch.acos(cosv)
        max_deg = float(ang.max()) * 180.0 / np.pi
        assert max_deg < 5.0, \
            f"det={np.linalg.det(Q):.0f} 时旋转等变性失败: max pair angle {max_deg:.2f}°"
        # 默认 (legacy) 分支在同一数据上应可复现地非等变 (>5°),
        # 防止有人误把 legacy 行为当成等变实现 (或反向误删该分支)。
        w1, _ = local_frames(kp1, K)                      # 默认=legacy
        w2, _ = local_frames(kp2, K)
        rw1 = torch.einsum("ef,blfg->bleg", R, w1)
        ang_leg = torch.acos((rw1 * w2).sum(-2).abs().clamp(0, 1))
        max_leg = float(ang_leg.max()) * 180.0 / np.pi
        assert max_leg > 5.0, \
            f"legacy 帧意外等变 ({max_leg:.2f}°)? 冻结行为描述需更新"


def test_fracture_aware_dirs_unobserved_fractures():
    """BUG-1 回归: 无观测点的裂隙必须走空间回退, 不产出零向量 (=90°).

    构造: 60 点、两个正交组、40% 观测; 其中若干裂隙的全部点均未被观测。
    断言: 所有隐伏点输出非零方向 (旧实现 37/37 全零 -> acos(0)=90°)。
    """
    from fractureflow.connectivity import fracture_aware_dirs
    rng = np.random.default_rng(42)
    n1 = np.array([1.0, 0.0, 0.0])
    n2 = np.array([0.0, 1.0, 0.0])
    pos_list, nrm_list, fid_list = [], [], []
    fid = 0
    for base_n, c in ((n1, [0.0, 0.0, 0.0]), (n2, [10.0, 0.0, 0.0])):
        for _ in range(6):                      # 每组 6 条裂隙
            ctr = np.array(c) + rng.normal(scale=2.0, size=3)
            pts = ctr + rng.normal(scale=0.05, size=(5, 3))   # 5 点/裂隙
            noisy = base_n + rng.normal(scale=0.02, size=3)
            pos_list.append(pts)
            nrm_list.append(np.repeat(noisy[None], 5, axis=0))
            fid_list.append(np.full(5, fid))
            fid += 1
    pos = np.concatenate(pos_list)              # 60 点
    nrm_true = np.concatenate(nrm_list)
    fids = np.concatenate(fid_list)
    occ = rng.random(60) < 0.4
    # 保证至少一条裂隙完全无观测
    hidden_frac = fids[~occ]
    all_fids = np.unique(fids)
    no_obs_fids = np.setdiff1d(all_fids, hidden_frac)
    if len(no_obs_fids) == 0:
        f = all_fids[0]
        occ[fids == f] = False                  # 强制整条裂隙无观测
        no_obs_fids = np.array([f])
    assert len(no_obs_fids) >= 1

    # 盲协议: 隐伏点法向置零 (与 honest_eval.make_blind 一致)
    nrm_blind = nrm_true * occ[:, None]
    dirs, labels = fracture_aware_dirs(pos, nrm_blind, occ, fracture_id=fids)
    hid = ~occ
    norms = np.linalg.norm(dirs[hid], axis=1)
    assert (norms > 0.5).all(), \
        f"{int((norms <= 0.5).sum())}/{int(hid.sum())} 个隐伏点输出零向量 (BUG-1 复发)"
    # 有观测的裂隙: 平面解应贴近真法向 (<15°)
    solved = np.isin(fids[~occ], np.setdiff1d(all_fids, no_obs_fids))
    if solved.any():
        err = np.degrees(np.arccos(np.clip(
            np.abs((dirs[hid][solved] * nrm_true[hid][solved]).sum(-1)), 0, 1)))
        assert err.mean() < 15.0, f"有观测裂隙误差过大: {err.mean():.2f}°"


def test_geo_prior_batch_no_collapse():
    """BUG-6 回归: 批内低观测网不得把其他网的 k-means 组数拖垮.

    旧实现 Kk=min(K, max(全批最小观测,2)): 一个 2 观测的"穷"网会把同批
    富网的组数压到 2, 输出与单独成批时不一致。修复后按网 cap 屏蔽超限簇,
    富网输出必须与单独成批时逐点一致。
    """
    import sys as _sys
    _SRC = os.path.join(_ROOT, "src")
    if _SRC not in _sys.path:
        _sys.path.insert(0, _SRC)
    from fractureflow.geo_prior import geo_prior_dirs_gpu

    rng = np.random.default_rng(3)
    axes = np.eye(3)

    def make_net(center, scale):
        pts, nrms = [], []
        for k in range(3):
            c = center + rng.normal(size=3) * scale
            p = c + rng.normal(size=(13, 3)) * 0.05 + axes[k] * 0.1
            n = axes[k] + rng.normal(scale=0.02, size=3)
            pts.append(p)
            nrms.append(np.repeat(n[None], 13, axis=0))
        return np.concatenate(pts), np.concatenate(nrms)

    rich_pos, rich_nrm = make_net([0., 0., 0.], 5.0)          # 39 点富网
    L = len(rich_pos)
    occ_rich = rng.random(L) < 0.6                            # ~23 观测
    nrm_blind_rich = rich_nrm * occ_rich[:, None]

    # 穷网: 仅 2 个观测
    poor_pos = rng.normal(size=(L, 3)) * 5.0
    poor_nrm = np.zeros((L, 3))
    poor_idx = rng.choice(L, 2, replace=False)
    poor_nrm[poor_idx] = axes[2] + rng.normal(scale=0.02, size=(2, 3))
    occ_poor = np.zeros(L, bool)
    occ_poor[poor_idx] = True

    def to_t(pos, nrm, occ):
        return (torch.tensor(pos, dtype=torch.float32)[None],
                torch.tensor(nrm, dtype=torch.float32)[None],
                torch.tensor(occ.astype(np.float32))[None])

    pr, nr_, mr = to_t(rich_pos, nrm_blind_rich, occ_rich)
    pp, np_, mp = to_t(poor_pos, poor_nrm, occ_poor)
    pos_b = torch.cat([pr, pp]); nrm_b = torch.cat([nr_, np_]); mask_b = torch.cat([mr, mp])

    gp_alone = geo_prior_dirs_gpu(pr, nr_, mr)
    gp_mixed = geo_prior_dirs_gpu(pos_b, nrm_b, mask_b)[0:1]
    max_diff = float((gp_alone - gp_mixed).abs().max())
    assert max_diff < 1e-4, \
        f"富网输出因批内穷网改变 (BUG-6 复发): max|diff|={max_diff:.4f}"


if __name__ == "__main__":
    tests = [
        test_horizontal_plane,
        test_vertical_plane,
        test_45_degree,
        test_dip_direction_north,
        test_dip_direction_east,
        test_dip_direction_south,
        test_dip_direction_west,
        test_forge_16b_lineage_anchor,
        test_local_frames_rotation_equivariance,
        test_fracture_aware_dirs_unobserved_fractures,
        test_geo_prior_batch_no_collapse,
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
