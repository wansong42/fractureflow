# -*- coding: utf-8 -*-
"""M5 两阶段训练总程序。

阶段 A: 合成预训练 (synth_10000.pt, 学物理先验)
阶段 B: 真实微调 (391 网络, 80/10/10 固定种子切分, EMA + cosine + wd)

训练中每 eval_every 步快评隐藏点 MAE, 保存 best EMA 权重。
"""

import argparse
import math
import os
import time

import numpy as np
import torch
from torch.optim import AdamW

from .config import FracGenConfig, p
from .model import FracGen
from .data import make_batches


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def ema_update(ema, model, decay):
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if k in ema and v.dtype.is_floating_point:
                ema[k].mul_(decay).add_(v, alpha=1.0 - decay)


def save_checkpoint(path, model, ema, cfg, step, tag):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"ema": ema, "cfg": {k: v for k, v in vars(cfg).items()},
                "step": step, "tag": tag}, path)


def load_checkpoint(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = FracGenConfig()
    for k, v in ck.get("cfg", {}).items():
        setattr(cfg, k, v)
    cfg.device = device          # 以调用方 device 为准, 防止保存 cfg 里的旧 device 篡改
    model = FracGen(cfg).to(device)
    ema = ck.get("ema") or ck.get("state")
    if ema is not None:
        # T82 反模式清扫 (与 trainer.load_ema_into 同规则): 键过滤+strict=False
        # 的静默半加载在键名漂移时会拿半随机初始化的模型继续评测.
        from .trainer import load_ema_into
        load_ema_into(model, ema, tag=f"load_checkpoint {os.path.basename(path)}")
    return model, cfg, ck.get("step", 0), ema


def cycle_batches(nets, batch_size, obs_frac, rng, device):
    while True:
        for grouped in make_batches(nets, batch_size, obs_frac, rng, device):
            for bb in grouped:
                yield bb


def eval_netlist(model, nets, cfg, obs_frac=None, rng_seed=999):
    """全量评估一个网络列表: 返回 dict 与逐点原始量 (供后续分层/分场地)。

    掩码: 固定 rng_seed -> 评测掩码可复现 (评测协议 40% 观测)。
    """
    from .data import collate
    obs_frac = cfg.obs_frac if obs_frac is None else obs_frac
    rng = np.random.default_rng(rng_seed)
    errs, anis, kaps, srcs, wids = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for b in collate(nets, obs_frac, rng, cfg.device):
            out = model(b)
            cos = (out["mean"] * b["nrm_full"]).sum(-1).abs().clamp(-1, 1)
            e = torch.rad2deg(torch.acos(cos))            # [B,L]
            hid = (1.0 - b["mask"]) > 0.5
            for bi in range(b["pos"].shape[0]):
                h = hid[bi]
                errs.append(e[bi][h].cpu())
                anis.append(out["aniso"][bi][h].cpu())
                kaps.append(out["kappa"][bi][h].cpu())
                nh = int(h.sum())
                srcs.extend([b["src"][bi]] * nh)
                wids.extend([b["wid"][bi]] * nh)
    errs = torch.cat(errs)
    anis = torch.cat(anis)
    kaps = torch.cat(kaps)
    return {
        "errs": errs, "anis": anis, "kaps": kaps, "srcs": srcs, "wids": wids,
        "mae": float(errs.mean()), "p50": float(errs.median()),
        "p90": float(torch.quantile(errs, 0.90)), "n_hid": int(errs.numel()),
    }


def run_stage(cfg, model, nets, steps, lr, batch, warmup, tag, val_nets=None,
              ckpt_best=None, ckpt_end=None, seed_delta=0, eval_every=250,
              log_path=None, wd=0.0):
    """通用训练循环 (阶段 A/B 共用)。返回 EMA 参数字典。"""
    set_seed(cfg.seed + seed_delta)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()
           if v.dtype.is_floating_point}
    rng = np.random.default_rng(cfg.seed + seed_delta)
    best_mae = 1e9
    t0 = time.time()
    rows = []
    nlist = nets
    for step in range(1, steps + 1):
        t = step - 1
        if t < warmup:
            sched = t / max(1, warmup)
        else:
            sched = 0.5 * (1.0 + math.cos(math.pi * (t - warmup) / max(1, steps - warmup)))
        lr_now = lr * sched
        for g in opt.param_groups:
            g["lr"] = lr_now

        b = next(cycle_batches(nlist, batch, cfg.obs_frac, rng, cfg.device))
        opt.zero_grad(set_to_none=True)
        loss, det = model.loss(b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        ema_update(ema, model, cfg.ema_decay)

        if step % 20 == 0 or step == steps:
            print(f"[{tag}] step {step}/{steps}  loss={loss.item():.4f}  "
                  f"l_hid={det['l_hid']:.4f}  lr={lr_now:.2e}  "
                  f"{time.time() - t0:.0f}s", flush=True)

        if val_nets is not None and (step % eval_every == 0 or step == steps):
            r = eval_with_ema(model, ema, val_nets, cfg)
            rows.append({"step": step, "tag": tag, "mae": r["mae"],
                         "p50": r["p50"], "p90": r["p90"], "n": r["n_hid"]})
            print(f"   [eval] {tag} val mae={r['mae']:.3f} p50={r['p50']:.3f} "
                  f"p90={r['p90']:.3f} (n={r['n_hid']})", flush=True)
            if ckpt_best and r["mae"] < best_mae:
                best_mae = r["mae"]
                save_checkpoint(ckpt_best, model, ema, cfg, step, tag)
                print(f"   * best {tag} -> saved {ckpt_best}", flush=True)

    if ckpt_end:
        save_checkpoint(ckpt_end, model, ema, cfg, steps, tag)
    if log_path:
        _write_log(log_path, tag, rows)
    return ema


def eval_with_ema(model, ema, nets, cfg, obs_frac=None, rng_seed=999):
    """用 EMA 参数评估 (评估后还原模型参数)"""
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    try:
        model.load_state_dict(ema, strict=False)
        return eval_netlist(model, nets, cfg, obs_frac=obs_frac, rng_seed=rng_seed)
    finally:
        with torch.no_grad():
            model.load_state_dict(backup, strict=False)


def _write_log(log_path, tag, rows):
    import csv
    exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "tag", "mae", "p50", "p90", "n"])
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    # 帧护栏 (与 trainer.train_experiment 同规则): 本入口的 FracGen 经
    # build_graph_features 走 local_frames, 新训练必须显式确认 fixed 帧.
    from .geometry import ack_training_frame_policy
    ack_training_frame_policy("fixed")

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["A", "B", "both"], default="both")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--steps", type=int, default=None, help="覆盖阶段步数")
    args = ap.parse_args()
    if args.steps is not None:
        print(f"[warn] --steps={args.steps} 覆盖阶段步数, 结束时会以该步数权重覆写 ckpt (冒烟谨慎)", flush=True)

    cfg = FracGenConfig()
    torch.set_num_threads(4)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={cfg.device}")

    # 数据
    real = torch.load(cfg.real_path, weights_only=True)
    synth = torch.load(cfg.synth_path, weights_only=True)
    synth_val = torch.load(cfg.synth_val_path, weights_only=True)

    # 真实切分 (80/10/10, 固定种子)
    set_seed(cfg.seed)
    idx = np.random.default_rng(cfg.seed).permutation(len(real))
    n_tr = int(0.8 * len(real)); n_va = int(0.1 * len(real))
    tr = [real[i] for i in idx[:n_tr]]
    va = [real[i] for i in idx[n_tr:n_tr + n_va]]
    te = [real[i] for i in idx[n_tr + n_va:]]
    print(f"real split: train={len(tr)} val={len(va)} test={len(te)}")

    model = None
    if args.only in ("A", "both"):
        model = FracGen(cfg).to(cfg.device)
        val_sub = synth_val[:cfg.eval_nets]
        steps = cfg.sA_steps if args.steps is None else args.steps
        run_stage(cfg, model, synth, steps, cfg.sA_lr, cfg.sA_batch,
                  cfg.sA_warmup, "A-synth", val_nets=val_sub,
                  ckpt_end=cfg.ckpt_stageA, seed_delta=0,
                  eval_every=cfg.eval_every_A, log_path=cfg.log_csv, wd=cfg.wd)
    if args.only in ("B", "both"):
        start = cfg.ckpt_stageA if args.ckpt is None else args.ckpt
        model, cfg2, _, _ = load_checkpoint(start, cfg.device) if os.path.exists(start) \
            else (FracGen(cfg).to(cfg.device), cfg, 0, None)
        steps = cfg.sB_steps if args.steps is None else args.steps
        run_stage(cfg, model, tr, steps, cfg.sB_lr, cfg.sB_batch,
                  cfg.sB_warmup, "B-real", val_nets=va,
                  ckpt_best=cfg.ckpt, ckpt_end=cfg.ckpt, seed_delta=1,
                  eval_every=cfg.eval_every_B, log_path=cfg.log_csv, wd=cfg.wd)
    print("done")


if __name__ == "__main__":
    main()