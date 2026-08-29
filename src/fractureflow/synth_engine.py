# -*- coding: utf-8 -*-
"""M1 物理合成数据引擎：生成"物理正确的"条件训练对。

原理（对应 ARCHITECTURE.md M1）:
  应力 (s1,s3) -> Anderson 候选面 (张裂面法向 n_A、剪裂面法向 n_B)
  -> K 个裂隙组 (Fisher 噪声) -> 逐点法向 (Fisher 噪声) -> 点位/长度/岩性
  -> 观测掩码 (观测比例 obs_frac, 未观测点法向置零)。

输出 dict 与真实数据 (data/real/loaded_real_nets.pt) 完全对齐, 额外保存:
  obs_mask, nrm_full(用于监督), set_ids, set_dirs, 供评测组级质量。
"""

import os
import numpy as np
import torch

rng_default = None


def set_seed(seed):
    global rng_default
    rng_default = np.random.default_rng(seed)
    torch.manual_seed(seed)


def _rng():
    return rng_default if rng_default is not None else np.random.default_rng()


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _basis(mu):
    """以 mu 为法向的平面上两个正交基向量"""
    mu = unit(mu)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(mu, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    a = unit(np.cross(ref, mu))
    b = np.cross(mu, a)
    return a, b


def rot_z_axis_to(v, mu):
    """将 z 轴为参照的 Fisher 样本旋转到 mu: 反射 (vMF 关于中心对称故合法)"""
    z = np.array([0.0, 0.0, 1.0])
    if np.all(np.abs(mu - z) < 1e-6):
        return unit(v)
    if np.all(np.abs(mu + z) < 1e-6):
        return unit(v * np.array([1.0, 1.0, -1.0]))
    h = unit(mu - z)
    R = np.eye(3) - 2.0 * np.outer(h, h)
    return unit(v @ R.T)


def fisher_samples(center, kappa, n, rng=None):
    """center 附近按 Fisher(vMF, kappa) 采样 n 个方向 [n,3] (numpy)。

    kappa<=0 时退化为均匀球面采样。center 可为标量或向量, 输出 shape 对应。
    """
    rng = rng if rng is not None else _rng()
    c1 = np.asarray(center, dtype=float)
    M = c1.shape[0] if c1.ndim == 2 else 1
    if c1.ndim == 1:
        c1 = c1[None]
    kap = np.broadcast_to(np.asarray(kappa, dtype=float), (M,))
    out = np.zeros((M, n, 3))
    for m in range(M):
        if kap[m] <= 1e-8:
            v = rng.standard_normal((n, 3))
            out[m] = unit(v)
            continue
        u = rng.uniform(size=(n,))
        w = 1.0 + (1.0 / kap[m]) * np.log(u + (1.0 - u) * np.exp(-2.0 * kap[m]))
        w = np.clip(w, -1.0, 1.0)
        s = np.sqrt(np.maximum(1.0 - w * w, 0.0))
        v = rng.standard_normal((n, 2))
        vn = unit(v)
        z = np.concatenate((s[:, None] * vn, w[:, None]), axis=1)
        out[m] = rot_z_axis_to(z, c1[m])
    return out[0] if M == 1 else out


def fit_fisher(pts):
    """vMF 拟合返回 (均值方向, MLE kappa 小样本偏差校正)。kappa 解 coth(k)-1/k = Rbar"""
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return unit(np.asarray([0.0, 0.0, 1.0])), 0.0
    R = np.mean(pts, axis=0)
    Rbar = np.linalg.norm(R)
    mean = unit(R)
    n = pts.shape[0]
    if Rbar < 1e-6:
        return mean, 0.0
    x = Rbar
    if x < 0.53:
        kappa = 3.0 * x + x**3 * (2.0 / 5.0 + 8.0 * x**2 / 35.0 + 8.0 * x**4 / 105.0)
    elif x < 0.85:
        kappa = (3.0 * x - 12.0 * x**3 + 24.0 * x**5) / (1.0 - 2.0 * x**2)
    else:
        kappa = 1.0 / (1.0 - x) - 1.5
    if kappa <= 0:
        return mean, 0.0
    from scipy.optimize import brentq

    def f(k):
        return 1.0 / np.tanh(k) - 1.0 / k - Rbar

    try:
        kap = brentq(f, 1e-6, 1000.0)
    except Exception:
        kap = kappa
    # 小样本偏差校正: E[khat]/k ≈ 1 + 2.5/n (经验, 见文档注)
    return mean, float(kap * n / (n + 2.5))


def fracture_normal_prior(s1, s3):
    """Anderson 断裂预测: 返回候选法向 [n_A, n_B]。

    n_A: 张裂面法向 (n_A = s3)
    n_B: 剪裂面法向, 与 s1 成 15°~45° (s1-s3 平面内向 s3 偏转)
    """
    z = unit(np.asarray(s3, dtype=float))
    s1v = unit(np.asarray(s1, dtype=float))
    plane = s1v - np.dot(s1v, z) * z
    if np.linalg.norm(plane) < 1e-6:
        plane = np.array([1.0, 0.0, 0.0])
        for ax in [np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]:
            if np.linalg.norm(np.cross(z, ax)) > 1e-6:
                plane = np.cross(z, ax)
                break
    plane = unit(plane)
    theta = np.deg2rad(_rng().uniform(15.0, 45.0))
    nB = s1v * np.cos(theta) + plane * np.sin(theta)
    return unit(np.stack([z, nB]))


def random_stress():
    """模拟地带差异的应力方向: s3 部分偏离垂直, s1 正交"""
    rng = _rng()
    if rng.random() < 0.6:
        tilt = np.deg2rad(rng.uniform(0, 30))
        s3 = unit(np.array([np.sin(tilt) * rng.uniform(-1, 1),
                            np.cos(tilt) * np.sin(rng.uniform(-1, 1) * tilt * 0.5),
                            np.cos(tilt)]))
    else:
        s3 = np.array([0.0, 0.0, 1.0])
    while True:
        s1 = unit(rng.standard_normal(3))
        if abs(np.dot(s1, s3)) < 5e-2:
            break
    s1 = s1 - np.dot(s1, s3) * s3
    return unit(s1), unit(s3)


def generate_net(k_sets=None, L=None, stress=None, obs_frac=0.4, seed=None,
                 lith_range=(1, 3), len_range=(0.25, 6.0)):
    """生成一个合成网络 dict (keys 与真实数据对齐)。"""
    rng = np.random.default_rng(seed)
    if k_sets is None:
        k_sets = int(rng.integers(1, 5))
    if L is None:
        L = int(rng.choice([40, 57, 114], p=[0.8, 0.15, 0.05]))
    L = max(L, k_sets * 4)

    if stress is None:
        s1, s3 = random_stress()
    else:
        s1 = unit(stress[0])
        s3 = unit(stress[1] - np.dot(stress[1], s1) * s1)

    dirs_prior = fracture_normal_prior(s1, s3)

    points, normals, lens, set_ids, set_dirs, kappa_pt = [], [], [], [], [], []
    n_per = np.full(k_sets, L // k_sets, dtype=int)
    n_per[-1] = L - n_per[:-1].sum()
    center = rng.uniform(-2, 2, size=3)

# 每个组的空间子域 (真实 DFN 中组在空间上局部集聚, 位置是组归属的强证据)
    sub_regions = []
    for k in range(k_sets):
        R = rng.uniform(0.45, 1.1)                       # 子域半径
        axis = rng.normal(0, 1, 3)
        axis = unit(axis)
        # 子域中心从网络中心沿随机方向偏移 >= 量级 R, 保证组间空间分离
        off = axis * (rng.uniform(0.5, 2.2) + R)
        sub_regions.append((center + off, R, rng.uniform(0.15, 0.45)))

    for k in range(k_sets):
        mu_k = fisher_samples(dirs_prior[k % 2], rng.uniform(5.0, 30.0), 1, rng)[0]
        set_dirs.append(mu_k)
        a, b = _basis(mu_k)
        p0, Rk, ov = sub_regions[k]
        n_pts = int(n_per[k])
        for j in range(n_pts):
            r = Rk * (np.abs(rng.normal(0, 0.55)) if rng.random() < 0.7 else rng.uniform(0.0, 1.0))
            ang = rng.uniform(0, 2 * np.pi)
            other = rng.standard_normal(3) * ov * Rk
            pos = p0 + r * (a * np.cos(ang) + b * np.sin(ang)) + mu_k * rng.normal(0, 0.12) + other
            points.append(pos)
            kp = rng.uniform(10.0, 50.0)
            normals.append(fisher_samples(mu_k, kp, 1, rng)[0])
            kappa_pt.append(kp)
            set_ids.append(k)
    order = rng.permutation(L)
    points = np.asarray(points)[order]
    normals = np.asarray(normals)[order]
    set_ids = np.asarray(set_ids, dtype=np.int64)[order]
    set_dirs = np.asarray(set_dirs)
    lens = np.exp(rng.uniform(np.log(len_range[0]), np.log(len_range[1]), size=L))
    lith = np.asarray(rng.integers(lith_range[0], lith_range[1] + 1, size=L), dtype=np.int64)
    mask = rng.random(L) < obs_frac

    return {
        "pos": torch.as_tensor(points, dtype=torch.float32),
        "nrm": torch.as_tensor(normals * mask[:, None], dtype=torch.float32),
        "nrm_full": torch.as_tensor(normals, dtype=torch.float32),
        "len": torch.as_tensor(lens[:, None], dtype=torch.float32),
        "lith": torch.as_tensor(lith, dtype=torch.int64),
        "s1": torch.as_tensor(s1, dtype=torch.float32),
        "s3": torch.as_tensor(s3, dtype=torch.float32),
        "obs_mask": torch.as_tensor(mask.astype(np.float32)),
        "set_ids": torch.as_tensor(set_ids),
        "set_dirs": torch.as_tensor(set_dirs, dtype=torch.float32),
        "kappa_pt": torch.as_tensor(np.asarray(kappa_pt, dtype=np.float32)[order]),
        "src": "synth",
        "wid": f"synth_{seed}",
    }


def generate_batch(N, out_path, seed=0, chunk=500):
    """生成 N 个网络保存到 out_path (.pt, list of dict)。"""
    nets = []
    for c0 in range(0, N, chunk):
        for _ in range(min(chunk, N - c0)):
            nets.append(generate_net(seed=c0 * 1000 + _ + 1000 * seed))
        print(f"  generated {len(nets)}/{N}", flush=True)
    torch.save(nets, out_path)
    print(f"saved -> {out_path} ({len(nets)} nets)")
    return nets


def validate_synth(nets, preview_dir):
    """验收: (a) 法向单位化 (b) Fisher 拟合回 κ 与设定误差<30% (c) 3D 图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d  # noqa

    os.makedirs(preview_dir, exist_ok=True)
    n_unit = 0
    for net in nets:
        nrm = net["nrm_full"].numpy()
        if np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-3):
            n_unit += 1
    set_errs = []
    kappa_rels = []
    for net in nets:
        ids = net["set_ids"].numpy()
        dirs = net["set_dirs"].numpy()
        nrm = net["nrm_full"].numpy()
        kp = net.get("kappa_pt")
        for k in range(dirs.shape[0]):
            sel = ids == k
            mu, _ = fit_fisher(nrm[sel])
            cos = abs(np.dot(mu, dirs[k]))
            set_errs.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            if kp is not None:
                _, kap = fit_fisher(nrm[sel])
                krel = abs(kap - float(kp[sel].numpy().mean())) / float(kp[sel].numpy().mean())
                kappa_rels.append(krel)
    print(f"[validate] unit-normalized nets: {n_unit}/{len(nets)}")
    print(f"[validate] per-set center true-vs-avg error: mean {np.mean(set_errs):.1f}°  "
          f"(应接近 0, 证明 Fisher 采样与拟合自洽)")
    if kappa_rels:
        print(f"[validate] Fisher κ 拟合相对误差: mean {np.mean(kappa_rels)*100:.1f}%  "
              f"(要求 <30%)")

    for i in range(min(5, len(nets))):
        net = nets[i]
        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        pos = net["pos"].numpy()
        nrm = net["nrm_full"].numpy()
        ids = net["set_ids"].numpy()
        colors = plt.cm.tab10(np.clip(ids, 0, 9) / 9.0)
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=10)
        ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2], nrm[:, 0], nrm[:, 1], nrm[:, 2],
                  length=0.8, normalize=True, alpha=0.6, linewidth=0.8)
        for s1, s3, col in [(net["s1"].numpy(), None, "r"), (net["s3"].numpy(), None, "b")]:
            ax.quiver(0, 0, 0, s1[0], s1[1], s1[2], color=col, length=2.5, linewidth=3)
        ax.set_title(net["wid"])
        fig.savefig(os.path.join(preview_dir, f"preview_{i}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    return float(np.mean(set_errs))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    from config import FracGenConfig

    cfg = FracGenConfig()
    nets = generate_batch(args.n, cfg.synth_path if args.out is None else args.out)
    validate_synth(nets, cfg.synth_preview_dir)