# -*- coding: utf-8 -*-
"""orthostats —— 方向统计层 (R185)。

给组系产状补上「区间」而不是只有点估计。四个模块:

    _sphere        通用 p 维 Bingham 配分函数 (numpy/torch 双后端单算法)
    bingham        轴分布拟合 (凸 MLE) / 密度 / 精确拒绝采样
    acg            角中心高斯: 可微重参数化采样 + 与 Bingham 的两种映射
    terzaghi_layer 可微 Terzaghi 采样偏差校正层 (torch)
    intervals      区间构造 (Wald 椭圆 / 参数 bootstrap / 非参 bootstrap) + 校准

设计红线
--------
1. **轴数据口径**: 一切分布都满足 f(x) = f(-x)。任何出现「把法向当有向向量」
   的地方都是 bug。
2. **零 NN 训练**: 本包是统计软件件, 不训练网络; 可微只服务于梯度反传与
   未来的分布头, 不引入任何可学习参数 (除非调用方显式包 nn.Module)。
3. **不做静默降级**: 拟合不收敛 / 采样预算耗尽 / 区间不可辨识, 一律响亮抛出
   或在结果里标 `converged=False`, 不做裁剪美化。
4. **数字直读制**: 落盘 JSON 一律全精度, 不 round; round 只出现在展示层。

依赖: numpy + scipy (打分/采样) + torch (拟合/可微)。
"""

from ._sphere import (
    sphere_area,
    log_bingham_norm,
    log_bingham_norm_np,
    HAS_TORCH,
    verify_double_backward,
)
from .bingham import (
    Bingham,
    fit_bingham,
    fit_bingham_batch,
    bingham_from_scatter,
    bingham_params_from_scatter_t,
)
from . import acg as _acg
from .acg import (
    ACG,
    fit_acg,
    acg_from_bingham,
    acg_from_bingham_calibrated,
    bingham_acg_moment_report,
    rms_opening_deg,
)
from . import terzaghi_layer as _terzaghi_layer
from .terzaghi_layer import (
    TerzaghiBinghamLayer,
    terzaghi_weights_t,
    terzaghi_scatter,
    terzaghi_ess,
    censoring_mask,
    soft_clamp,
)
from . import intervals as _intervals
from .intervals import (
    NOMINAL_LEVELS,
    chi2_2,
    wald_uncertainty,
    wald_cover,
    bootstrap_angles,
    bootstrap_cone,
    fit_calibration_scale,
    sample_observed,
)

__all__ = [
    "sphere_area",
    "log_bingham_norm",
    "log_bingham_norm_np",
    "HAS_TORCH",
    "DOUBLE_BACKWARD_OK",
    "verify_double_backward",
    "Bingham",
    "fit_bingham",
    "fit_bingham_batch",
    "bingham_from_scatter",
    "bingham_params_from_scatter_t",
    "ACG",
    "fit_acg",
    "acg_from_bingham",
    "acg_from_bingham_calibrated",
    "bingham_acg_moment_report",
    "rms_opening_deg",
    "TerzaghiBinghamLayer",
    "terzaghi_weights_t",
    "terzaghi_scatter",
    "terzaghi_ess",
    "censoring_mask",
    "soft_clamp",
    "NOMINAL_LEVELS",
    "chi2_2",
    "wald_uncertainty",
    "wald_cover",
    "bootstrap_angles",
    "bootstrap_cone",
    "fit_calibration_scale",
    "sample_observed",
]

__version__ = "r187.v1"


def __getattr__(name: str):
    # PEP 562 惰性转发 (R187-T1, 坑 33): DOUBLE_BACKWARD_OK 不再在包 import 期
    # 急切求值 —— 模块级探测曾在 import 时跑 autograd+leggauss, 撞 torch/MKL
    # 双 OpenMP 即 OMP Error #15 硬崩。现 `from fractureflow.orthostats import
    # DOUBLE_BACKWARD_OK` 仍兼容, 但探测延迟到真正访问时 (首次后缓存)。
    if name == "DOUBLE_BACKWARD_OK":
        from ._sphere import verify_double_backward
        return verify_double_backward()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
