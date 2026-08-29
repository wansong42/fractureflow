# -*- coding: utf-8 -*-
"""实验编排入口: 读 JSON 配置 -> 构建 -> 训练/评测。

设计(用户要求): 换组件 = 改配置一行, 不碰代码。
- 骨干: backbone.name 在 {"e3gt_v4", "plain_mlp", ...} 中选。
- 增广: stages.A.augments / stages.B.augments 列表, 每项 {"name": ...}。
- 损失: losses 列表, 每项 {"name": ...} (可组合)。

评测口径铁律: 锁定逐点 acos(|<pred,true>|), 始终附几何基线 top2_ens 对照
(见 evaluator.compare_report), 绝不切换口径凑达标。

用法(在 src/ 目录下):
  python -m fractureflow.run --config fractureflow/configs/exp_e3gt_v4.json --smoke
  python -m fractureflow.run --config fractureflow/configs/exp_e3gt_v4.json
  python -m fractureflow.run --config fractureflow/configs/exp_e3gt_v4.json --stage A --steps 40000
"""

import os
import json
import argparse

import numpy as np
import torch
from torch.optim import AdamW

from .core import DEVICE
from . import backbones, augments, losses  # 触发所有注册
from .trainer import (build_backbone, apply_augments, compute_loss,
                      eval_official, train_experiment)
from .splits import load_split
from . import config as _cfg

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REAL_PATH = _cfg.p("data/real/loaded_real_nets.pt")
SYNTH_PATH = _cfg.p("data/synth/synth_10000.pt")


def smoke_run(cfg):
    """快速验证: 该配置能构建、能训练几步、能评测 (证明"换组件"机制可用)。"""
    from .data import make_batches
    obs_frac = cfg.get("obs_frac", 0.4)
    model = build_backbone(cfg["backbone"])
    aug_specs = cfg.get("augments", []) or []
    loss_specs = cfg.get("losses", [{"name": "dir"}])
    synth = torch.load(SYNTH_PATH, weights_only=False)
    _, _, te = load_split(REAL_PATH)
    opt = AdamW(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    print(f"[smoke] backbone={cfg['backbone']['name']} "
          f"augments(default)={[a['name'] for a in aug_specs]} "
          f"losses={[l['name'] for l in loss_specs]}", flush=True)
    steps_done = 0
    for grouped in make_batches(synth[:16], 8, obs_frac, rng, DEVICE):
        for bb in grouped:
            if aug_specs:
                bb = apply_augments(bb, aug_specs, rng)
            out, loss, comp = compute_loss(model, bb, loss_specs)
            loss.backward()
            opt.step()
            opt.zero_grad()
            steps_done += 1
            if steps_done % 4 == 0:
                print(f"  synth step {steps_done} loss={float(loss):.3f} {comp}",
                      flush=True)
            if steps_done >= 24:
                break
        if steps_done >= 24:
            break
    r = eval_official(model, te[:8], obs_frac)
    print("  smoke eval te[:8]:", r)
    print("[smoke] OK - 组件可换、管线能跑。", flush=True)


def run(config_path, stage=None, steps=None, smoke=False):
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if smoke:
        return smoke_run(cfg)
    return train_experiment(cfg, stage_override=stage, steps_override=steps)


def main():
    ap = argparse.ArgumentParser(description="fractureflow 模块化实验入口")
    ap.add_argument("--config", required=True, help="实验 JSON 配置路径")
    ap.add_argument("--stage", default=None, choices=["A", "B", "both"],
                    help="只跑某阶段 (默认跑配置里 enabled 的阶段)")
    ap.add_argument("--steps", type=int, default=None, help="覆盖该阶段步数")
    ap.add_argument("--smoke", action="store_true", help="快速验证管线, 不跑完整训练")
    args = ap.parse_args()
    run(args.config, stage=args.stage, steps=args.steps, smoke=args.smoke)


if __name__ == "__main__":
    main()
