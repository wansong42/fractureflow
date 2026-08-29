# -*- coding: utf-8 -*-
"""通用训练器: 由配置 dict 驱动, 跑 A(合成预训练)/B(真实微调) 两阶段。

特点 (支撑"快速试错/不行就换"):
- backbone / augments / losses 全由配置里的名字组装, 不写死。
- EMA + 余弦 LR + 梯度裁剪; 训练期按 augments 列表逐批增广。
- 阶段末用锁定口径评测器产出 "模型 vs 几何基线" 对照, 杜绝换口径虚假达标。
"""

import os
import math
import time
import json

import numpy as np
import torch
from torch.optim import AdamW

from .core import BACKBONES, AUGMENTS, LOSSES, DEVICE
from . import backbones, augments, losses  # 触发注册
from .splits import load_split
from .data import make_batches
from .evaluator import compare_report, evaluate_model
from . import config as _cfg

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def rp(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


REAL_PATH = _cfg.p("data/real/loaded_real_nets.pt")
SYNTH_PATH = _cfg.p("data/synth/synth_10000.pt")
SYNTH_VAL_PATH = _cfg.p("data/synth/synth_val.pt")

# 模块级缓存: 同一进程内多次 train_experiment 不重复从磁盘读 10000 个合成网络。
_LOAD_CACHE = {}


def _load_cached(path, weights_only=False):
    p = rp(path)
    if p not in _LOAD_CACHE:
        _LOAD_CACHE[p] = torch.load(p, weights_only=weights_only)
    return _LOAD_CACHE[p]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def ema_update(ema, model, decay):
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if k in ema and v.dtype.is_floating_point:
                ema[k].mul_(decay).add_(v, alpha=1.0 - decay)


def build_backbone(spec):
    cls = BACKBONES[spec["name"]]
    kw = {k: v for k, v in spec.items() if k != "name"}
    return cls(**kw).to(DEVICE)


def apply_augments(b, specs, rng):
    for s in specs:
        fn = AUGMENTS[s["name"]]
        b = fn(b, rng, **{k: v for k, v in s.items() if k != "name"})
    return b


def compute_loss(model, b, loss_specs):
    out = model(b)
    total = 0.0
    comp = {}
    for s in loss_specs:
        fn = LOSSES[s["name"]]
        l, c = fn(out, b, **{k: v for k, v in s.items() if k != "name"})
        total = total + l
        comp.update(c)
    return out, total, comp


def eval_official(model, te, obs_frac, device=DEVICE):
    return evaluate_model(model, te, obs_frac, device)


def load_ema_into(model, ckpt_ema, tag=""):
    """EMA 加载: 按 state_dict 全键集过滤 (含 buffer), 键名打错时显式报警。

    旧实现按 named_parameters 过滤 —— (a) 静默丢弃所有 buffer;
    (b) checkpoint 键名打错/模型换骨干后剩余键过少也只静默 strict=False,
    模型处于半随机初始化状态继续评测, 结果不可信。此处改为:
    命中 state_dict 键集 + 形状一致才加载; 覆盖率过低 (<50%) 直接抛错。
    """
    sd_model = model.state_dict()
    loaded = {}
    skipped = []
    for k, v in ckpt_ema.items():
        if k in sd_model and sd_model[k].shape == v.shape:
            loaded[k] = v
        else:
            skipped.append(k)
    cov = len(loaded) / max(1, len(sd_model))
    if cov < 0.5:
        raise RuntimeError(
            f"[{tag}] EMA 加载覆盖率过低: {len(loaded)}/{len(sd_model)} "
            f"(checkpoint 多余键 {len(skipped)} 个, 例: {skipped[:3]}). "
            f"疑似 checkpoint 与模型结构不匹配, 拒绝静默半加载.")
    if skipped:
        print(f"[{tag}] EMA 加载警告: 丢弃 {len(skipped)} 个不匹配键 "
              f"(例: {skipped[:3]})", flush=True)
    model.load_state_dict(loaded, strict=False)


def eval_official_ema(model, ema, te, obs_frac):
    # P17 修复 (训练期 eval 模式锁死): evaluate_model 内部 model.eval() 之后
    # 从不恢复 -> 周期性验证一开跑, 后续所有训练步 dropout/obs_drop 全部失效
    # (~90% 步数无正则)。此处记录进入前模式, finally 中恢复。
    was_training = model.training
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    try:
        model.load_state_dict(ema, strict=False)
        return eval_official(model, te, obs_frac)
    finally:
        model.load_state_dict(backup, strict=False)
        model.train(was_training)


def load_ema_strict(model, ckpt_ema, tag=""):
    """严格版 EMA 加载: 模型侧任何缺键/形状失配直接 raise (训练脚本入口专用)。

    为什么不用 load_ema_into: 其 50% 覆盖率闸门挡不住"小头失配"——
    实测 models/e3gt_hybrid.pt (旧架构 gate_head/set_dir_head) 载入当前类
    缺 residual_head 4 键仍有 109/113=96% 覆盖, 只打 warning 放行, 半随机
    初始化静默污染整段训练/评测。作为训练起点/最终复现的加载必须全键匹配。
    checkpoint 侧多余键 (如旧头) 容忍但显式列出。
    """
    sd_model = model.state_dict()
    missing = [k for k in sd_model if k not in ckpt_ema]
    mismatched = [k for k in sd_model
                  if k in ckpt_ema
                  and tuple(sd_model[k].shape) != tuple(ckpt_ema[k].shape)]
    if missing or mismatched:
        raise RuntimeError(
            f"[{tag}] EMA 严格加载失败: 模型缺键 {len(missing)} 个 "
            f"(例: {missing[:4]}), 形状失配 {len(mismatched)} 个 "
            f"(例: {mismatched[:4]})。"
            f"疑似 checkpoint 与当前模型结构不匹配 (冻结模型请用其训练时代的"
            f"代码评测, 或显式走对应架构)。拒绝静默半加载。")
    extra = [k for k in ckpt_ema if k not in sd_model]
    if extra:
        print(f"[{tag}] EMA 严格加载警告: checkpoint 多余键 {len(extra)} 个 "
              f"(例: {extra[:4]}), 已忽略", flush=True)
    model.load_state_dict(ckpt_ema, strict=False)


def cycle(nets, batch_size, obs_frac, rng, device):
    while True:
        for grouped in make_batches(nets, batch_size, obs_frac, rng, device):
            for bb in grouped:
                yield bb


def run_stage(model, nets, steps, lr, batch, warmup, tag, obs_frac, cfg,
              val_nets=None, eval_every=2000, wd=0.0, seed_delta=0,
              aug_specs=None, loss_specs=None, start_step=1,
              partial_ckpt=None, save_every=2000):
    set_seed(cfg.get("seed", 42) + seed_delta)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()
           if v.dtype.is_floating_point}
    rng = np.random.default_rng(cfg.get("seed", 42) + seed_delta)
    t0 = time.time()
    model.prior_fast = True  # 训练步用廉价几何先验加速; 评测前会切回精确
    it = cycle(nets, batch, obs_frac, rng, DEVICE)
    for step in range(start_step, steps + 1):
        t = step - 1
        if t < warmup:
            sched = t / max(1, warmup)
        else:
            sched = 0.5 * (1 + math.cos(math.pi * (t - warmup) / max(1, steps - warmup)))
        lr_now = lr * sched
        for g in opt.param_groups:
            g["lr"] = lr_now
        b = next(it)
        if aug_specs:
            b = apply_augments(b, aug_specs, rng)
        opt.zero_grad(set_to_none=True)
        _, loss, comp = compute_loss(model, b, loss_specs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        ema_update(ema, model, 0.9995)
        if step % 50 == 0 or step == steps:
            print(f"[{tag}] {step}/{steps} loss={loss.item():.4f} "
                  f"l_hid={comp.get('l_hid', 0):.4f} l_obs={comp.get('l_obs', 0):.4f} "
                  f"l_set={comp.get('l_set', 0):.4f} lr={lr_now:.2e} {time.time() - t0:.0f}s",
                  flush=True)
        if val_nets is not None and (step % eval_every == 0 or step == steps):
            model.prior_fast = False  # 评测必须用精确先验, 保证地板保证
            r = eval_official_ema(model, ema, val_nets, obs_frac)
            model.prior_fast = True
            print(f"   [eval] {tag} val mae={r['mae']:.2f} p50={r['p50']:.2f} "
                  f"p90={r['p90']:.2f}", flush=True)
        # 周期性落盘 (防沙箱杀进程导致整段训练进度丢失)
        if partial_ckpt is not None and (step % save_every == 0 or step == steps):
            torch.save({"ema": ema, "step": step}, partial_ckpt)
            print(f"   [ckpt] {tag} partial step={step} -> "
                  f"{os.path.basename(partial_ckpt)}", flush=True)
    return ema


def train_experiment(cfg, stage_override=None, steps_override=None):
    from .geometry import training_frame_acked
    if not training_frame_acked():
        raise RuntimeError(
            "train_experiment 拒绝开跑: 未调用 geometry.ack_training_frame_policy"
            "('fixed')。T82 后 local_frames 默认 legacy (非等变, BUG-2), 新训练必须"
            "在入口显式确认使用 fixed 帧。")
    real = _load_cached(REAL_PATH, weights_only=False)
    synth = _load_cached(SYNTH_PATH, weights_only=False)
    synth_val = _load_cached(SYNTH_VAL_PATH, weights_only=False)
    tr, va, te = load_split(REAL_PATH, cfg.get("seed", 42))
    obs_frac = cfg.get("obs_frac", 0.4)
    wd = cfg.get("wd", 5e-5)

    model = build_backbone(cfg["backbone"])
    aug_specs = cfg.get("augments", [])
    loss_specs = cfg.get("losses", [{"name": "dir"}])
    stages = cfg.get("stages", {})

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[build] backbone={cfg['backbone']['name']} params={n_params:,} "
          f"augments={[a['name'] for a in aug_specs]} "
          f"losses={[l['name'] for l in loss_specs]}", flush=True)

    final_ckpt_path = None

    # ---------- 阶段 A: 合成预训练 ----------
    if stages.get("A", {}).get("enabled") and stage_override in (None, "A", "both"):
        A = stages["A"]
        aug_specs_A = A.get("augments", aug_specs)
        stageA_ckpt = rp(A.get("ckpt", "models/_stageA.pt"))
        stageA_partial = (stageA_ckpt[:-3] + "_partial.pt") if stageA_ckpt.endswith(".pt") \
            else stageA_ckpt + "_partial"
        # 断点续跑:
        #  - 全量 ckpt 存在 (上一轮跑完) -> 直接加载跳过 A;
        #  - 否则 partial ckpt 存在 (上一轮被杀, A 未跑完) -> 加载 ema 从断点续跑;
        #  - 否则从头跑。每 eval_every 自动落 partial ckpt, 防沙箱杀手清零进度。
        if os.path.exists(stageA_ckpt) and stage_override in (None, "both"):
            print(f"[stage A] 跳过 (已存在全量 ckpt {stageA_ckpt}), 直接加载用于 B",
                  flush=True)
            ck = torch.load(stageA_ckpt, map_location=DEVICE, weights_only=False)
            load_ema_into(model, ck["ema"], tag="stage A")
            final_ckpt_path = stageA_ckpt
        else:
            print(f"[stage A] augments={[a['name'] for a in aug_specs_A]}", flush=True)
            steps = steps_override if steps_override else A.get("steps", 40000)
            aug_specs_A = A.get("augments", aug_specs)
            start_step = 1
            if os.path.exists(stageA_partial):
                # 只捕获 torch.load 层面的文件损坏; load_ema_into 的覆盖率
                # RuntimeError = checkpoint 与模型结构失配, 必须向上抛 ——
                # 旧写法宽 except 把失配当"损坏"删档, 静默丢弃全部进度从零重训.
                try:
                    ck = torch.load(stageA_partial, map_location=DEVICE, weights_only=False)
                except Exception:
                    try:
                        os.remove(stageA_partial)
                    except OSError:
                        pass
                    print(f"[stage A] partial ckpt 损坏, 从头开始", flush=True)
                else:
                    load_ema_into(model, ck["ema"], tag="stage A resume")
                    start_step = int(ck.get("step", 0)) + 1
                    print(f"[stage A] 续跑: 加载 partial ckpt step={start_step}", flush=True)
            if start_step > steps:
                ema = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
            else:
                ema = run_stage(model, synth, steps, A.get("lr", 3e-4), A.get("batch", 32),
                                A.get("warmup", 500), "A-synth", obs_frac, cfg,
                                val_nets=synth_val[:200], eval_every=A.get("eval_every", 4000),
                                wd=wd, aug_specs=aug_specs_A, loss_specs=loss_specs,
                                start_step=start_step, partial_ckpt=stageA_partial,
                                save_every=A.get("save_every", A.get("eval_every", 4000)))
            torch.save({"ema": ema}, stageA_ckpt)
            if os.path.exists(stageA_partial):
                try:
                    os.remove(stageA_partial)
                except OSError:
                    pass  # sandbox safe-delete shim may block; leftover partial is harmless
            final_ckpt_path = stageA_ckpt

    # ---------- 阶段 B: 真实微调 ----------
    if stages.get("B", {}).get("enabled") and stage_override in (None, "B", "both"):
        B = stages["B"]
        aug_specs_B = B.get("augments", aug_specs)
        print(f"[stage B] augments={[a['name'] for a in aug_specs_B]}", flush=True)
        stageA_ckpt = rp(stages["A"].get("ckpt", "models/_stageA.pt")) if stages.get("A") else None
        stageB_ckpt = rp(B.get("ckpt", "models/_final.pt"))
        stageB_partial = (stageB_ckpt[:-3] + "_partial.pt") if stageB_ckpt.endswith(".pt") \
            else stageB_ckpt + "_partial"
        steps = steps_override if steps_override else B.get("steps", 30000)
        aug_specs_B = B.get("augments", aug_specs)
        start_step_B = 1
        # 全量 ckpt 存在 -> 跳过; 否则 partial 存在 -> 断点续跑 (直接加载 B 的 ema);
        # 否则从 stageA ema 起步训练 B。
        if os.path.exists(stageB_ckpt):
            print(f"[stage B] 跳过 (已存在全量 ckpt {stageB_ckpt})", flush=True)
            ck = torch.load(stageB_ckpt, map_location=DEVICE, weights_only=False)
            load_ema_into(model, ck["ema"], tag="stage B")
        else:
            if os.path.exists(stageB_partial):
                # 同阶段 A: 文件损坏才删档重来; 结构失配 (load_ema_into) 直接抛.
                try:
                    ck = torch.load(stageB_partial, map_location=DEVICE, weights_only=False)
                except Exception:
                    try:
                        os.remove(stageB_partial)
                    except OSError:
                        pass
                    print(f"[stage B] partial ckpt 损坏, 从头开始", flush=True)
                else:
                    load_ema_into(model, ck["ema"], tag="stage B resume")
                    start_step_B = int(ck.get("step", 0)) + 1
                    print(f"[stage B] 续跑: 加载 partial ckpt step={start_step_B}", flush=True)
            elif B.get("from_stageA") and stageA_ckpt and os.path.exists(stageA_ckpt):
                ck = torch.load(stageA_ckpt, map_location=DEVICE, weights_only=False)
                load_ema_into(model, ck["ema"], tag="B<-A")
                print(f"[B] loaded stageA ema from {stageA_ckpt}", flush=True)
            if start_step_B > steps:
                ema = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
            else:
                ema = run_stage(model, tr, steps, B.get("lr", 1e-4), B.get("batch", 16),
                                B.get("warmup", 500), "B-real", obs_frac, cfg,
                                val_nets=va, eval_every=B.get("eval_every", 2000), wd=wd,
                                seed_delta=1, aug_specs=aug_specs_B, loss_specs=loss_specs,
                                start_step=start_step_B, partial_ckpt=stageB_partial,
                                save_every=B.get("save_every", B.get("eval_every", 2000)))
            torch.save({"ema": ema}, stageB_ckpt)
            if os.path.exists(stageB_partial):
                try:
                    os.remove(stageB_partial)
                except OSError:
                    pass  # sandbox safe-delete shim may block; leftover partial is harmless
        final_ckpt_path = stageB_ckpt

    # ---------- 最终锁定口径评测 ----------
    if final_ckpt_path and os.path.exists(final_ckpt_path):
        ck = torch.load(final_ckpt_path, map_location=DEVICE, weights_only=False)
        load_ema_into(model, ck["ema"], tag="final")

    model.prior_fast = False  # 最终评测必须用精确先验, 否则地板保证失效
    rep = compare_report(model, te, cfg)
    print("\n=== FINAL (test split, rng999, 锁定逐点 acos 口径) ===")
    print(json.dumps(rep, indent=2, ensure_ascii=False, allow_nan=False))
    out_eval = rp(cfg.get("out_eval", "results/exp_eval.json"))
    with open(out_eval, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False, allow_nan=False)
    print("wrote", out_eval)
    return rep
