# -*- coding: utf-8 -*-
"""损失函数 (可替换, 注册式, 可组合)。

每个损失签名: fn(out: dict, b: dict, **kwargs) -> (scalar, comp_dict)
- out: 模型 forward 返回值 (至少含 "pred"; 可选 "aniso"/"set_dir_pred")
- b:   批次张量 dict
损失按配置列表逐个计算并累加; 缺某组件(如 plain 无 set_dir_pred)则自动跳过。
"""

import torch

from .core import LOSSES


@LOSSES.register("dir")
def dir_loss(out, b, w_hid=1.0, w_obs=0.3, aniso_floor=0.2, w_aniso=1.0, **kw):
    """隐伏点方向主损失 (按各向异性聚焦可预测点) + 观测点重建辅助。"""
    pred = out["pred"]
    mask = b["mask"]
    nrm_full = b["nrm_full"]
    cos = (pred * nrm_full).sum(-1).abs().clamp(0, 1)
    err1 = 1.0 - cos
    aniso = out.get("aniso")

    total = 0.0
    comp = {}
    hid = (1.0 - mask) > 0.5
    obs = mask > 0.5
    if hid.any():
        if aniso is not None:
            aw = torch.clamp(aniso / 0.5, aniso_floor, 1.0) * w_aniso
            l_hid = (err1[hid] * aw[hid]).mean() / max(float(aw[hid].mean()), 1e-3)
        else:
            l_hid = err1[hid].mean()
        total = total + w_hid * l_hid
        comp["l_hid"] = float(err1[hid].mean())
    if obs.any():
        l_obs = err1[obs].mean()
        total = total + w_obs * l_obs
        comp["l_obs"] = float(l_obs)
    comp["total"] = float(total)
    return total, comp


@LOSSES.register("set_dir")
def set_dir_loss(out, b, w_set=0.5, **kw):
    """辅助: 监督"预测所在裂隙组方向" (攻击指派瓶颈)。

    模型无 set_dir_pred, 或真实数据无 set_ids/set_dirs -> 自动跳过 (返回 0)。
    """
    if "set_dir_pred" not in out:
        return 0.0, {}
    if "set_ids" not in b or "set_dirs" not in b:
        return 0.0, {}
    Bb, Lb, _ = b["pos"].shape
    set_ids = b["set_ids"].long()
    set_dirs = b["set_dirs"]
    # P17 对齐: 与 e3gt_hybrid(_v2).loss 相同的零向量组过滤 —— 指向零向量的组
    # (构造失败/超 max_sets) 会产生 90° 伪梯度污染训练。
    sd_norm = set_dirs.norm(dim=-1)                            # [B,G]
    valid_group = sd_norm > 1e-3
    sid = set_ids.clamp(0, set_dirs.shape[1] - 1)
    valid = (set_ids >= 0) & valid_group.gather(1, sid)
    idx = sid.unsqueeze(-1).expand(Bb, Lb, 3).to(torch.long)
    tgt = torch.gather(set_dirs, 1, idx)                       # [B,L,3]
    cos_s = (out["set_dir_pred"] * tgt).sum(-1).abs().clamp(0, 1)
    err_s = 1.0 - cos_s
    if valid.any():
        l_set = err_s[valid].mean()
        return w_set * l_set, {"l_set": float(l_set)}
    return 0.0, {}


@LOSSES.register("memb")
def memb_loss(out, b, w_memb=1.0, **kw):
    """赋值瓶颈主攻损失: 监督逐点"组隶属" logits (仅合成数据有 set_ids 时激活)。

    模型无 out["memb"] 或真实数据无 set_ids -> 自动跳过 (返回 0)。
    与 dir 损失协同: dir 把 pred_assign 拉向真值法向, memb 把每个点分到正确的
    全局原型方向 -> 原型 p_k 被拉到真实组方向, 形成"少数连贯组方向"的混合物。
    """
    if "memb" not in out:
        return 0.0, {}
    if "set_ids" not in b:
        return 0.0, {}
    logits = out["memb"]                                      # [B,L,K]
    set_ids = b["set_ids"].long()                             # [B,L]
    valid = set_ids >= 0
    tgt = set_ids.clamp(0, logits.shape[-1] - 1)
    if valid.any():
        l = torch.nn.functional.cross_entropy(logits[valid], tgt[valid])
        return w_memb * l, {"l_memb": float(l)}
    return 0.0, {}


@LOSSES.register("aux_dir")
def aux_dir_loss(out, b, key="pred1", w_aux=0.3, w_obs=0.3, **kw):
    """级联中间级深监督: 对 out[key] (如 pred1) 施加与主损失同型的方向损失。

    键不存在 -> 自动跳过 (返回 0), 因此非级联骨干可复用同一配置。
    """
    if key not in out:
        return 0.0, {}
    pred = out[key]
    mask = b["mask"]
    nrm_full = b["nrm_full"]
    err1 = 1.0 - (pred * nrm_full).sum(-1).abs().clamp(0, 1)
    hid = (1.0 - mask) > 0.5
    obs = mask > 0.5
    total = 0.0
    comp = {}
    if hid.any():
        l = err1[hid].mean()
        total = total + l
        comp["l_aux"] = float(l)
    if obs.any():
        total = total + w_obs * err1[obs].mean()
    return w_aux * total, comp


@LOSSES.register("cand_sel")
def cand_sel_loss(out, b, w_csel=1.0, beta=8.0, **kw):
    """候选选择监督: 直接用真值法向构造"最优候选"软目标, 训练注意力权重 w 聚焦到
    与真值最一致的那个空间候选 (观测法向). 这是把 28° 推向 13° 的核心监督——
    它显式教网络"挑"出正确观测法向, 而非平滑平均。

    仅当 out 提供 "w"(注意力权重) 与 "cand"(候选方向) 且 b 有 nrm_full 时激活。
    """
    if "w" not in out or "cand" not in out or "nrm_full" not in b:
        return 0.0, {}
    w = out["w"].squeeze(-1)                      # [B,L,Kc]
    cand = out["cand"]                            # [B,L,Kc,3]
    true = b["nrm_full"]
    true = true / true.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    sim = (cand * true.unsqueeze(2)).sum(-1)      # [B,L,Kc]  cos 相似度
    target = torch.softmax(beta * sim, dim=-1)    # 聚焦到与真值最一致的候选
    wpos = w.clamp_min(1e-8)
    loss = -(target * wpos.log()).sum(-1).mean()  # KL(target || w) 的 -E_target[log w]
    return w_csel * loss, {"l_csel": float(loss)}
