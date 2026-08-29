# -*- coding: utf-8 -*-
"""FracGen 全局配置。所有超参集中于此, 命令行参数可覆盖。"""

import os
from dataclasses import dataclass, field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def p(rel):
    return os.path.join(ROOT, rel)


@dataclass
class FracGenConfig:
    seed: int = 42
    device: str = "cuda"

    # ---- 几何 / 网络结构 ----
    d_model: int = 256          # 逐点特征维度 (增加容量)
    k_knn: int = 16             # kNN 邻域 (扩大感受野)
    n_proto: int = 12           # 组原型池大小 (增加容量)
    n_proto_min: int = 1
    n_proto_max: int = 4
    temp_assign: float = 10.0   # 软 vMF 隶属度温度
    kappa_max: float = 80.0     # 更大浓度上限
    gnn_layers: int = 3         # 更深 GNN
    dropout: float = 0.15       # 稍大 dropout 防过拟合
    use_attention: bool = True  # GNN 中加入注意力

    # 数据
    obs_frac: float = 0.4       # 观测比例(其余隐伏, 法向置零)
    real_path: str = field(default_factory=lambda: p("data/real/loaded_real_nets.pt"))
    synth_path: str = field(default_factory=lambda: p("data/synth/synth_10000.pt"))
    synth_val_path: str = field(default_factory=lambda: p("data/synth/synth_val.pt"))
    synth_n: int = 10000
    synth_val_n: int = 2000

    # ---- 阶段 A: 合成预训练 ----
    sA_steps: int = 50000       # 更长预训练
    sA_lr: float = 3e-4
    sA_batch: int = 32
    sA_warmup: int = 500
    eval_every_A: int = 5000

    # ---- 阶段 B: 真实微调 ----
    sB_steps: int = 20000       # 大幅延长微调
    sB_lr: float = 1e-4         # 更小学习率
    sB_batch: int = 16          # 更小 batch 更稳定
    sB_warmup: int = 500
    eval_every_B: int = 1000
    wd: float = 5e-5            # 更小 weight decay

    # ---- 场地留一验证 ----
    loo_steps: int = 2000
    loo_lr: float = 5e-4

    # ---- 损失权重 ----
    w_sym: float = 0.05         # 对称损失(降低)
    w_anderson: float = 0.1     # Anderson 物理先验正则 (增加)
    w_aniso: float = 2.0        # 各向异性权重乘数 (增加, 重点优化高可预测点)
    w_obs: float = 0.3          # 观测点 NLL 校准权重 (降低)
    w_ent: float = 0.02         # 组熵惩罚 (增加)
    w_assign: float = 1.0       # 标签传播监督 (增加)
    w_prop: float = 1.5         # 显式观测→隐伏传播分支监督 (增加)
    w_true: float = 3.0         # 传播分支直接回归掩码真值 (大幅增加, 核心监督)
    w_setg: float = 0.3         # 合成真值组指派监督 (原型匹配, 新增)

    ema_decay: float = 0.9995   # 更慢 EMA
    eval_nets: int = 200        # 快速评估用网络数

    # ---- 输出路径 ----
    ckpt: str = field(default_factory=lambda: p("models/fracgen_v1.pt"))
    ckpt_stageA: str = field(default_factory=lambda: p("models/fracgen_v1_stageA.pt"))
    log_csv: str = field(default_factory=lambda: p("results/train_log.csv"))
    result_json: str = field(default_factory=lambda: p("results/result.json"))
    cmp_json: str = field(default_factory=lambda: p("results/cmp_vs_baseline.json"))
    report_md: str = field(default_factory=lambda: p("results/final_report.md"))
    synth_preview_dir: str = field(default_factory=lambda: p("results/synth_preview"))
    viz_dir: str = field(default_factory=lambda: p("results/viz_compare"))

    def update(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self