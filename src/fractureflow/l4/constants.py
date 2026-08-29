# -*- coding: utf-8 -*-
"""L4 包公共常量 —— 单点维护, 禁止各处硬编码.

决策阈值 (来自 T34 多源一致性审计 + 多井联合审计, 已验证):
  ANGLE_CONSISTENT   = 20°   最大交叉角距 < 此值 → "一致" (可池化)
  ANGLE_PARTIAL      = 35°   20-35° → "部分一致" (仅合并最稳对)
  ANGLE_CONFLICT     = 35°   >= 此值 → "冲突" (独立处理, 如实报告)

融合权重:
  SOURCE_QUALITY_INITIAL = 1.0   各源初始等权, 由 T46 回测数据校验/修订

bootstrap:
  N_BOOTSTRAP        = 1000  置信锥 bootstrap 次数
  CONFIDENCE_LEVEL   = 0.95  标称覆盖率

后验集合:
  M_ENSEMBLE         = 20    DFN 实现数

评测:
  SEED_HONEST        = 42    诚实评测种子
"""
# 决策阈值 (度)
ANGLE_CONSISTENT: float = 20.0
ANGLE_PARTIAL: float = 35.0
ANGLE_CONFLICT: float = 35.0

# 融合初始权重
SOURCE_QUALITY_INITIAL: float = 1.0

# bootstrap 参数
N_BOOTSTRAP: int = 1000
CONFIDENCE_LEVEL: float = 0.95

# 后验集合大小
M_ENSEMBLE: int = 20

# 评测种子
SEED_HONEST: int = 42

# 数值安全
EPS: float = 1e-12

# ---------------------------------------------------------------------------
# 结构面类型枚举 (T50 类型感知证据模型)
# ---------------------------------------------------------------------------
# ISRM 非连续结构类型族 + 数据驱动类型 (natural/induced)
# 缺省 unknown 时向后兼容 (混入各分组空间, 不划独立子空间)
STRUCTURE_TYPE_JOINT: str = "joint"
STRUCTURE_TYPE_BEDDING: str = "bedding"
STRUCTURE_TYPE_FOLIATION: str = "foliation"
STRUCTURE_TYPE_FAULT: str = "fault"
STRUCTURE_TYPE_INDUCED: str = "induced"
STRUCTURE_TYPE_NATURAL: str = "natural"
STRUCTURE_TYPE_CONTACT: str = "contact"
STRUCTURE_TYPE_UNKNOWN: str = "unknown"

STRUCTURE_TYPES = frozenset({
    "joint", "bedding", "foliation", "fault",
    "induced", "natural", "contact", "unknown",
})

# ---------------------------------------------------------------------------
# 类型混淆决策规则 (T53)
# ---------------------------------------------------------------------------
# 回答"什么时候必须打类型标签，什么时候可以混？"
# 来源: results/type_confusion_rule.json (5 角距 × 3 样本量 × 2 K × 10 seed = 300 次实验)
#
# 混聚损失 = 混聚 modal_err − 隔离 modal_err
#   损失 > 0 → 混合聚类更差 → 必须打类型标签隔离
#   损失 < 0 → 混合聚类更好 → 可混入 (k-means 自然分开)
#
# 三行规则 (均值阈值, 分位数切分):
#   1. 角距 <= TYPE_ISOLATE_MAX_DEG → 必须隔离
#   2. 角距 >= TYPE_MIX_MIN_DEG → 可混入 (隔离反劣)
#   3. 每类型样本 <= TYPE_ISOLATE_MIN_OBS → 隔离反劣 (每组点数不够 k-means 稳定)
TYPE_ISOLATE_MAX_DEG: float = 10.0    # 角距 <= 此值 → 必须隔离
TYPE_MIX_MIN_DEG: float = 45.0        # 角距 >= 此值 → 可混入
TYPE_ISOLATE_MIN_OBS: int = 20        # 每类型样本 <= 此值 → 隔离反劣
TYPE_ISOLATE_THRESHOLD: float = -0.5  # 均值阈值 (损失 > 此值 → 隔离区)
TYPE_MIX_THRESHOLD: float = -3.0      # 均值阈值 (损失 < 此值 → 混入区)
