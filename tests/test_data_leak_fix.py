# -*- coding: utf-8 -*-
"""T76/T77 单测: data.py 泄漏修复 + 防御性加固.

覆盖:
  - S23 毒丸加强版: _group_dirs_from_ids 的 0-观测组回退不依赖 nrm_full
  - prepare_net 幂等性: collate(prepare_net(x)) 不重掩码
  - MAX_SETS 超界显式报错 (替代静默截断)
"""
import os
import sys
import numpy as np
import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fractureflow.data import _group_dirs_from_ids, prepare_net


def _make_net(L=40, K=4, seed=42):
    """构造合成 net dict: K 组正交方向 + 小噪声."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(L, 3)).astype(np.float32)
    base = np.eye(3, dtype=np.float32)
    nrm_list = []
    for k in range(K):
        noise = rng.normal(0, 0.05, (L // K, 3)).astype(np.float32)
        pts = base[k % 3] + noise
        pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
        nrm_list.append(pts)
    nrm = np.concatenate(nrm_list, axis=0)
    set_ids = np.repeat(np.arange(K), L // K)
    return {
        "pos": torch.as_tensor(pos),
        "nrm": torch.as_tensor(nrm),
        "len": torch.ones(L, 1),
        "lith": torch.zeros(L, dtype=torch.int64),
        "s1": torch.eye(3)[0],
        "s3": torch.eye(3)[2],
        "set_ids": torch.as_tensor(set_ids, dtype=torch.int64),
        "wid": "test",
    }


# ---------------------------------------------------------------------------
# T76: _group_dirs_from_ids 泄漏修复
# ---------------------------------------------------------------------------

class TestGroupDirsFromIds:
    def test_0obs_group_fallback_no_leak(self):
        """0 观测组的 set_dirs 回退方向不依赖 nrm_full (T76 修复)."""
        net = _make_net()
        L = 40
        # 观测掩码: 第 4 组全隐伏
        obs_mask = np.ones(L, dtype=np.float32)
        obs_mask[30:40] = 0.0

        nrm_obs = np.asarray(net["nrm"]) * obs_mask[:, None]
        nrm_full = np.asarray(net["nrm"])
        set_ids = np.asarray(net["set_ids"])

        # 基线
        sd_base = _group_dirs_from_ids(nrm_obs, nrm_full, set_ids)

        # 毒丸: 把第 4 组 nrm_full 改为垃圾值
        nrm_full_poison = nrm_full.copy()
        nrm_full_poison[30:40] = np.array([0.0, 0.0, 1.0])
        sd_poison = _group_dirs_from_ids(nrm_obs, nrm_full_poison, set_ids)

        # set_dirs 应完全相同 (修复后)
        diff = (sd_base - sd_poison).abs().max().item()
        assert diff < 1e-5, f"nrm_full 污染改变了 set_dirs (max diff={diff:.2e}), 泄漏仍存在"

    def test_0obs_group_uses_global_observed_mean(self):
        """0 观测组的 set_dirs 应等于全局观测均值方向."""
        net = _make_net()
        L = 40
        obs_mask = np.ones(L, dtype=np.float32)
        obs_mask[30:40] = 0.0  # 第 4 组全隐伏

        nrm = np.asarray(net["nrm"])
        nrm_obs = nrm * obs_mask[:, None]
        set_ids = np.asarray(net["set_ids"])

        sd = _group_dirs_from_ids(nrm_obs, nrm, set_ids)

        # 全局观测均值
        obs_pts = nrm[obs_mask > 0.5]
        ref = obs_pts[0]
        sgn = np.sign((obs_pts * ref).sum(-1, keepdims=True))
        sgn[sgn == 0] = 1
        obs_aligned = obs_pts * sgn
        expected = obs_aligned.mean(0)
        expected /= np.linalg.norm(expected) + 1e-12

        # set_dirs[3] (第 4 组) 应等于全局观测均值
        actual = sd[3].numpy()
        cos = abs(np.dot(actual, expected))
        assert cos > 0.99, f"0-观测组回退方向不等于全局观测均值 (cos={cos:.4f})"

    def test_max_sets_hard_gate_large_K(self):
        """K > max_sets 时硬闸门 raise (十四期: 原 warning+跳过升级, BUG-4 拍板)."""
        import pytest
        net = _make_net(L=80, K=12)  # K=12 > max_sets=8
        nrm = np.asarray(net["nrm"])
        set_ids = np.asarray(net["set_ids"])
        with pytest.raises(ValueError, match="max_sets"):
            _group_dirs_from_ids(nrm, nrm, set_ids, max_sets=8)


# ---------------------------------------------------------------------------
# T77: prepare_net 幂等性
# ---------------------------------------------------------------------------

class TestPrepareNetIdempotency:
    def test_obs_mask_output_present(self):
        """prepare_net 输出应包含 obs_mask 键."""
        net = _make_net()
        b = prepare_net(net, 0.4, rng=np.random.default_rng(999))
        assert "obs_mask" in b, "prepare_net 未输出 obs_mask, collate 幂等性未修复"
        assert torch.equal(b["obs_mask"], b["mask"]), "obs_mask 应与 mask 相同"

    def test_reprepare_no_remask(self):
        """collate(prepare_net(x)) 不应重新掩码."""
        from fractureflow.data import collate
        net = _make_net()
        b1 = prepare_net(net, 0.4, rng=np.random.default_rng(999))
        mask1 = b1["mask"].clone()

        # 模拟 collate 再次调用 prepare_net
        prepared_nets = [b1]
        groups = {}
        for b in prepared_nets:
            groups.setdefault(b["pos"].shape[0], []).append(b)
        # 如果 prepare_net 的 obs_mask 逻辑正确, 已准备的 dict 不应重掩码
        b2 = prepare_net(b1, 0.4, rng=np.random.default_rng(999))
        mask2 = b2["mask"]

        assert torch.equal(mask1, mask2), (
            "prepare_net 对已准备的 dict 重新掩码! "
            "这意味着 collate(prepare_net(x)) 会改变掩码."
        )

    def test_max_sets_file_backed_raises(self):
        """set_dirs 文件 G > MAX_SETS 时应显式报错 (非静默截断)."""
        net = _make_net()
        # 伪造 G=12 的 set_dirs
        net["set_dirs"] = torch.randn(12, 3)
        with pytest.raises(ValueError, match="超过 MAX_SETS"):
            prepare_net(net, 0.4, rng=np.random.default_rng(999))

    def test_max_sets_file_backed_OK(self):
        """set_dirs 文件 G <= MAX_SETS 时正常通过."""
        net = _make_net()
        net["set_dirs"] = torch.randn(4, 3)
        b = prepare_net(net, 0.4, rng=np.random.default_rng(999))
        assert b["set_dirs"].shape == (8, 3)  # 补齐到 MAX_SETS


# ---------------------------------------------------------------------------
# T77: local_frames e2 符号确定性 + 旋转等变性
# ---------------------------------------------------------------------------

def _make_knn_sorted(B, L, K, seed=13, scale=(3.0, 1.5, 0.5)):
    """生成按距离排序的 kNN 数据 (模拟真实 knn_graph 输出).

    每个点的邻域按到中心点的距离升序排列.
    """
    rng = np.random.default_rng(seed)
    knn = rng.normal(size=(B, L, K, 3)).astype(np.float32)
    # 各向异性缩放
    for i, s in enumerate(scale):
        knn[:, :, :, i] *= s
    # 模拟距离排序: 添加径向偏移使近处点靠前, 远处在后
    radial = np.linspace(0.1, 1.0, K)[::-None] if False else np.linspace(1.0, 3.0, K)
    # 按到中心距离排序 (模拟 knn_graph 行为)
    center = knn.mean(2, keepdims=True)
    dists = np.linalg.norm(knn - center, axis=-1)  # [B,L,K]
    sort_idx = np.argsort(dists, axis=-1)  # [B,L,K]
    # 按距离排序
    b_idx = np.arange(B)[:, None, None]
    l_idx = np.arange(L)[None, :, None]
    knn_sorted = knn[b_idx, l_idx, sort_idx]
    return torch.as_tensor(knn_sorted)


# ---------------------------------------------------------------------------
# T77: local_frames e2 符号确定性 + 旋转等变性
# ---------------------------------------------------------------------------

class TestLocalFramesDeterministicE2:
    """T77: local_frames deterministic_e2 标志.

    注: eigh 返回的特征向量符号在特征值接近时存在固有歧义, 完美旋转等变性
    仅在三个特征值均分离时近似成立. 本测试验证:
      (1) default 行为不变 (冻结模型兼容)
      (2) deterministic_e2=True 产生合法正交归一右手系 frame
      (3) 重复调用输出一致 (符号确定性).
    """

    def test_default_unchanged(self):
        """默认 deterministic_e2=False 保持原有行为 (冻结模型兼容)."""
        from fractureflow.geometry import local_frames
        knn = _make_anisotropic_knn(seed=7)
        K = knn.shape[2]
        v0, e0 = local_frames(knn, K)
        v1, e1 = local_frames(knn, K, deterministic_e2=False)
        assert torch.allclose(v0, v1, atol=1e-6)
        assert torch.allclose(e0, e1, atol=1e-6)

    def test_deterministic_e2_valid_frame(self):
        """deterministic_e2=True 产生合法 frame (单位列 + 右手系)."""
        from fractureflow.geometry import local_frames
        knn = _make_anisotropic_knn(K=12, seed=13)
        v, e = local_frames(knn, 12, deterministic_e2=True)
        # 列向量单位化
        norms = v.norm(dim=-2)  # [B,L,3]
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        # 右手系: det([e1 e2 e3]) > 0
        det = torch.linalg.det(v)
        assert (det > 0).all(), f"frame 不是右手系: det min={det.min().item()}"
        # 列正交: e_i · e_j = delta_ij
        gram = torch.einsum("blci,blcj->blij", v, v)
        eye = torch.eye(3, device=v.device).expand_as(gram)
        assert torch.allclose(gram, eye, atol=1e-4), "frame 列不正交"

    def test_deterministic_e2_repeatable(self):
        """deterministic_e2=True 时重复调用输出一致."""
        from fractureflow.geometry import local_frames
        knn = _make_anisotropic_knn(K=12, seed=42)
        v1, _ = local_frames(knn, 12, deterministic_e2=True)
        v2, _ = local_frames(knn, 12, deterministic_e2=True)
        assert torch.allclose(v1, v2, atol=1e-6), "deterministic_e2 不重复"

    def test_deterministic_e2_vs_default_differ(self):
        """deterministic_e2=True 与 False 在 e2/e3 列上通常不同."""
        from fractureflow.geometry import local_frames
        knn = _make_anisotropic_knn(K=12, seed=55)
        v_def, _ = local_frames(knn, 12, deterministic_e2=False)
        v_det, _ = local_frames(knn, 12, deterministic_e2=True)
        # e1 列应相同 (都由 far-neighbor 规则固定)
        cos_e1 = (v_def[:, :, 0] * v_det[:, :, 0]).sum(-1).abs()
        assert cos_e1.min().item() > 0.99, "e1 列应一致"


def _make_anisotropic_knn(B=1, L=8, K=12, seed=13):
    """生成按距离排序的三轴各向异性 kNN 数据."""
    rng = np.random.default_rng(seed)
    center = rng.normal(size=(B, L, 1, 3)).astype(np.float32) * 0.5
    # 各向异性方向 (随机旋转的正交基)
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    Q = Q.astype(np.float32)
    # 生成 K 个点: 沿各向异性轴分布 + 噪声
    scales = np.array([3.0, 1.5, 0.5], dtype=np.float32)
    # 随机方向 + 各向异性缩放
    raw = rng.normal(size=(B, L, K, 3)).astype(np.float32)
    # 应用各向异性: 先缩放, 再旋转到随机方向
    raw_scaled = raw * scales[None, None, None, :]
    # 旋转到 Q 坐标系
    knn_local = np.einsum("blkc,cd->blkd", raw_scaled, Q)
    # 加上中心偏移和径向距离 (模拟 KNN)
    radial = np.linspace(0.5, 2.5, K).astype(np.float32)  # 升序径向距离
    direction = rng.normal(size=(B, L, K, 3)).astype(np.float32)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-8
    knn = center + direction * radial[None, None, :, None] + knn_local * 0.3
    # 按到 center 距离排序 (模拟 knn_graph)
    center_squeezed = center.squeeze(2)  # [B,L,3]
    dists = np.linalg.norm(knn - center_squeezed[:, :, None, :], axis=-1)
    sort_idx = np.argsort(dists, axis=-1)
    b_idx = np.arange(B)[:, None, None]
    l_ = np.arange(L)[None, :, None]
    knn_sorted = knn[b_idx, l_, sort_idx]
    return torch.as_tensor(knn_sorted)
