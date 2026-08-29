# -*- coding: utf-8 -*-
"""数据加载 / 掩码 / 批处理: 真实与合成网络统一接口。

net dict keys (与 TASKS.md 3.1 对齐): pos, nrm, len, lith, s1, s3, src, wid, widx
合成额外: nrm_full, obs_mask, set_ids, set_dirs
"""

import numpy as np
import torch


def _group_dirs_from_ids(nrm_obs, nrm_full, set_ids, max_sets=8):
    """用 set_ids + 法向构造组内方向 set_dirs [max_sets,3] (作 set_dir 监督伪标签).

    口径严格对齐几何 `set_aware_dirs`: 每组方向**优先用观测点** (nrm_obs, 掩码后,
    隐伏点法向=0 被忽略) 估计; 仅当该组观测点 = 0 时回退到**全局观测均值**
    (仅用观测点构造, 不含 nrm_full, 彻底消除隐伏真值泄漏). 无向法向用符号对齐均值,
    避免 ±n 互相抵消.

    T76 修复 (2026-08-22): 原实现回退到 nrm_full[sel] (含隐伏点真值), 造成
    set_dirs 泄漏. 实测 21.2% 网络 / 7.0% 组触发, 受影响组 leaky vs fixed 角距
    均值 15.0° / 最大 39.3°. 诚实榜 8 方法均不读取 set_dirs, 故诚实榜数字不受影响;
    但泄漏影响 e3gt_hybrid (geom_dir = set_dirs[sid]) 的诚实性, 必须修复.

    超出 max_sets 的组 (K > max_sets) 直接 raise (十四期硬闸门升级: 原
    T76 行为为 warning+跳过, 会静默把截断的 set_dirs 喂给神经通道 —— BUG-4
    拍板 K12 只走几何管线, 神经通道 n_sets 固定 8, 不允许 K12 数据流入)。
    """
    nrm_obs = np.asarray(nrm_obs.detach().cpu().numpy() if isinstance(nrm_obs, torch.Tensor)
                         else nrm_obs, dtype=np.float64)
    nrm_full = np.asarray(nrm_full.detach().cpu().numpy() if isinstance(nrm_full, torch.Tensor)
                          else nrm_full, dtype=np.float64)
    sids = np.asarray(set_ids.detach().cpu().numpy() if isinstance(set_ids, torch.Tensor)
                      else set_ids, dtype=np.int64).reshape(-1)
    K = int(sids.max()) + 1 if (sids.size and int(sids.max()) >= 0) else 0
    if K > max_sets:
        raise ValueError(
            f"_group_dirs_from_ids: K={K} > max_sets={max_sets}. "
            f"神经通道 n_sets 固定为 {max_sets} (Embedding 维度), 拒绝截断喂入 "
            f"(BUG-4: K12 商务锚点只走几何管线). 请降低数据 K 或重训增大 n_sets.")
    centers = np.zeros((max_sets, 3), dtype=np.float32)

    def _sign_aligned_mean(pts):
        pts = pts[np.linalg.norm(pts, axis=1) > 1e-6]
        if pts.shape[0] == 0:
            return None
        pts = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12)
        ref = pts[0]
        sgn = np.sign((pts * ref).sum(-1, keepdims=True))
        sgn[sgn == 0] = 1
        pts = pts * sgn
        c = pts.mean(0)
        n = np.linalg.norm(c)
        return c / (n + 1e-12) if n > 1e-6 else None

    # 全局观测均值 (T76: 无泄漏兜底方向, 仅用观测点)
    obs_mask = np.linalg.norm(nrm_obs, axis=1) > 1e-6
    obs_all = nrm_obs[obs_mask]
    if obs_all.shape[0] > 0:
        global_fallback = _sign_aligned_mean(obs_all)
        if global_fallback is None:
            global_fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        global_fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for k in range(K):
        sel = (sids == k)
        if sel.sum() == 0:
            continue
        obs = nrm_obs[sel]
        use = _sign_aligned_mean(obs)
        if use is None:                       # 观测为 0 -> 回退全局观测均值 (无泄漏)
            use = global_fallback
        if use is not None:
            centers[k] = use.astype(np.float32)
    return torch.as_tensor(centers, dtype=torch.float32)


def prepare_net(net, obs_frac, rng=None):
    """规范化一个网络: 法向单位化, 计算观测掩码 (掩码点法向置零), 长度转 log。

    返回 dict: pos, nrm(掩码后), nrm_full, log_len, lith, mask, s1, s3, src, wid, widx
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    L = net["pos"].shape[0]

    pos = net["pos"].float().clone()
    pos = pos - pos.mean(0, keepdim=True)

    nrm_full = net["nrm"].float().clone()
    if "nrm_full" in net:
        nrm_full = torch.as_tensor(np.asarray(net["nrm_full"]), dtype=torch.float32).clone()
    n2 = nrm_full.norm(dim=-1, keepdim=True)
    nrm_full = nrm_full / torch.clamp(n2, min=1e-6)

    obs = net.get("obs_mask")
    if obs is not None and np.asarray(obs).shape == (L,):
        mask = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(-1))
    else:
        mask = torch.as_tensor((rng.random(L) < obs_frac).astype(np.float32))
    nrm = nrm_full * mask[:, None]

    # R8 幂等修复: 同时回读 "len" 与 "lens" (首跑输出存 "lens",
    # 二次重入若只读 "len" 会回退 ones 导致 log_len/lens 漂移).
    lens_in = net.get("len", net.get("lens"))
    lens = torch.as_tensor(lens_in, dtype=torch.float32).reshape(-1, 1) if lens_in is not None \
        else torch.ones(L, 1)
    log_len = lens.log().clamp(-5, 5)

    lith = torch.as_tensor(np.asarray(net["lith"]), dtype=torch.int64).reshape(-1)
    if lith.shape[0] != L:
        lith = lith[:L].clone()
    lith = lith.clamp(0, 7)

    s1 = torch.as_tensor(np.asarray(net["s1"]).astype(np.float32))
    s3 = torch.as_tensor(np.asarray(net["s3"]).astype(np.float32))

    out = {
        "pos": pos, "nrm": nrm, "nrm_full": nrm_full, "log_len": log_len,
        "lens": lens, "lith": lith, "mask": mask, "s1": s1, "s3": s3,
        "wid": str(net.get("wid", "")), "widx": int(net.get("widx", 0)),
        "src": str(net.get("src", "")),
        "obs_mask": mask.clone(),  # T77 prepare_net 幂等: 输出 obs_mask 使 collate(prepare_net(x)) 不再重掩码
    }
    if "set_ids" in net:
        out["set_ids"] = torch.as_tensor(np.asarray(net["set_ids"]), dtype=torch.int64).reshape(-1)
        if "set_dirs" in net:
            sd = torch.as_tensor(np.asarray(net["set_dirs"]), dtype=torch.float32)
            MAX_SETS = 8
            G = sd.shape[0]
            if G > MAX_SETS:
                raise ValueError(
                    f"set_dirs 组数 G={G} 超过 MAX_SETS={MAX_SETS}. "
                    f"模型 n_sets 固定为 {MAX_SETS} (Embedding 维度). "
                    f"请重训时增大 n_sets, 或降低 K 至 ≤ {MAX_SETS}. "
                    f"(T77 修复: 替代原来的静默截断 sd[:{MAX_SETS}])")
            if G < MAX_SETS:
                sd = torch.cat([sd, torch.zeros(MAX_SETS - G, 3, dtype=torch.float32)], 0)
            out["set_dirs"] = sd
        else:
            # 真实数据 (路线 A) 仅带 set_ids 无 set_dirs: 用观测法向构造组内方向伪标签
            out["set_dirs"] = _group_dirs_from_ids(out["nrm"], out["nrm_full"], out["set_ids"])
    if "fracture_id" in net:
        out["fracture_id"] = torch.as_tensor(np.asarray(net["fracture_id"]),
                                             dtype=torch.int64).reshape(-1)
    return out


def collate(nets, obs_frac, rng, device="cuda"):
    """按 L 分组堆叠。返回 list, 每元素为一个网络的批量张量 dict。
    obs_frac/rng 用于真实网络掩码 (合成网络用自身 obs_mask)。"""
    groups = {}
    for net in nets:
        b = prepare_net(net, obs_frac, rng)
        groups.setdefault(b["pos"].shape[0], []).append(b)
    out = []
    for L, items in groups.items():
        B = len(items)
        b = {
            "pos": torch.stack([x["pos"] for x in items]).to(device),
            "nrm": torch.stack([x["nrm"] for x in items]).to(device),
            "nrm_full": torch.stack([x["nrm_full"] for x in items]).to(device),
            "log_len": torch.stack([x["log_len"] for x in items]).to(device),
            "lith": torch.stack([x["lith"] for x in items]).to(device),
            "mask": torch.stack([x["mask"] for x in items]).to(device),
            "s1": torch.stack([x["s1"] for x in items]).to(device),
            "s3": torch.stack([x["s3"] for x in items]).to(device),
            "wid": [x["wid"] for x in items],
            "src": [x["src"] for x in items],
            "n_L": L,
        }
        if "set_ids" in items[0]:
            b["set_ids"] = torch.stack([x["set_ids"] for x in items]).to(device)
            if "set_dirs" in items[0]:
                b["set_dirs"] = torch.stack([x["set_dirs"] for x in items]).to(device)
        if "fracture_id" in items[0]:
            b["fracture_id"] = torch.stack([x["fracture_id"] for x in items]).to(device)
        out.append(b)
    return out


def make_batches(nets, batch_size, obs_frac, rng, device="cuda"):
    """epoch 生成器: 打乱网络列表 -> 按 L 分组 -> 每批一个 collate"""
    nets = list(nets)
    perm = rng.permutation(len(nets))
    for i in range(0, len(perm), batch_size):
        chunk = [nets[j] for j in perm[i:i + batch_size]]
        yield collate(chunk, obs_frac, rng, device)