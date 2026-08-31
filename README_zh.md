# FractureFlow（裂隙流）

[English](README.md) | **中文**

> **把地质工程师手工干三天的"结构面统计"，变成十分钟自动出报告。**

岩土工程勘察报告里，结构面统计（裂隙组系表、玫瑰图、间距与连通性筛查）
是规范要求的必备章节——但目前几乎全靠人工完成，一个场地通常要花数天。
FractureFlow 用一条自动化流水线替代这个流程：

```
钻孔编录表 (深度, 倾角, 倾向)
   -> 自动组系划分（无需任何人工标注）
   -> 组系表 + 走向玫瑰图 + 赤平投影极点图
   -> 离散裂隙网络 (DFN) + 渗流筛查
   -> 中文统计报告
```

同时，本项目把"这套方法到底能走多远、哪里走不动"用防泄漏评测协议测了出来，
连同失败结果一起公开——见文末[诚实性与负结果](#诚实性与负结果)。

---

## 先看效果（demo 产物，合成数据）

下图由一键 demo 在**合成**的 3 组钻孔编录数据上生成，不依赖任何真实场地
数据。完整交付样例（报告 + 组系表 + 图件）在 [`examples/`](examples/)；
还有一个可交互的展示页（含可旋转的 3D DFN）：
[GitHub Pages 门户](https://wansong42.github.io/fractureflow/)。

| 走向玫瑰图 | 极点图（下半球投影） | DFN 与渗流筛查 |
|---|---|---|
| ![玫瑰图](examples/rose_diagram.png) | ![极点图](examples/stereonet.png) | ![DFN 三维](docs/screenshots/dfn_3d.png) |

## 两分钟跑起来

```bash
git clone https://github.com/wansong42/fractureflow.git
cd fractureflow

# Python >= 3.10，CPU 即可，无需 GPU
pip install .              # 安装 fractureflow 及全部依赖
#   或: pip install -r requirements.txt
#   或: conda env create -f environment.yml && conda activate fractureflow

./run_demo.sh            # Windows 用户: run_demo.cmd
```

demo 会自动生成一份合成钻孔编录数据，然后跑完整条产品链：自动打标 →
组系表 CSV → 玫瑰图/极点图 → DFN 实现 → 渗流筛查（`p_conn(P32)` 曲线 +
场景指标）→ 中文报告。不需要下载任何外部数据。

产物在 `results/demo/`；也可以先看 [`examples/`](examples/) 里提交的静态
样例。

## 用你自己的数据

```bash
# 钻孔编录表 CSV（深度/倾角/倾向，常见列名自动识别）
python scripts/auto_label_borehole.py --csv 你的编录表.csv --out labeled.pt

# Excel 编录表 -> 报告
python scripts/borehole_excel_entry.py --help

# FMI 类 LAS 成像文件，端到端：
# 打标 -> 组系表 -> DFN -> 渗流 -> 报告
python scripts/full_pipeline.py --input-las 你的井.las --domain 50 50 50

# 多井场地模型（逐井打标 + 井间一致性审计 + 四视角图）
python -m fractureflow.eval --site-model --wells 多井数据.npz \
    --site-domain 50 50 50 --site-out-dir results/site_model/
```

## 开源的算法清单

以下算法全部在 `src/fractureflow/`（MIT 许可），可读、可调用、有测试覆盖：

| 功能 | 位置 |
|---|---|
| **路线 A 自动打标**——对无向裂隙法向做球面 k-means（`|cos|` 指派 + 符号对齐），组数自适应选择 | `setlabel.py`, `adaptive_k.py` |
| **Fréchet 中位数产状预测器**（`l1_local`）——只用局部观测几何预测隐伏裂隙产状，是无标签场景最强的逐点预测器 | `inference.py` |
| **Terzaghi 采样校正**——用 1/|n·a| 加权修正钻孔与裂隙交角的采样偏差 | `terzaghi.py` |
| **Baecher 圆盘 DFN 生成器**——由组系表 + P32 强度 + 幂律尺寸生成离散裂隙网络 | `dfn.py` |
| **渗流筛查**——Balberg 排除体积阈值、`p_conn(P32)` 曲线、EGS/矿山/处置三场景指标 | `percolation.py` |
| **多井联合决策规则**——按井间组心一致性判断"合并池化 / 保持独立" | `site_model.py` |
| **冲突门控多源融合**——编录表 + 成像测井 + 露头证据，只在多源一致处融合 | `l4/` |
| **无标签稠密点云法向**——RANSAC 平面分割 + 局部 SVD（面向三维扫描数据） | `segmentation.py` |
| **BlindInput 诚实评测框架**——隐伏点协议、毒丸测试、泄漏红旗审计 | `honest_eval.py`, `set_table_eval.py` |
| **等变神经网络骨干**——研究线模型，作为有完整记录的负结果发布（见诚实性一节） | `backbones/` |

常用命令：

```bash
# 从 CSV 编录表直接出组系表 + 图件
python scripts/auto_label_borehole.py --csv 你的编录表.csv --plot

# 在随仓基准队列上跑诚实榜评测
python -m fractureflow.eval --point-mode l1local
```

## 基准：无标签推断能走多远

能力按观测档位组织（L0 → L4）。下表数字是随仓结果文件的冻结锚点（每行给出
路径），评测协议经过泄漏审计（脚注见下表）。

| 档位 | 数据 | 指标（诚实协议*） | 数值 | 证据 |
|---|---|---|---|---|
| L0 | 钻孔编录（22 井，880 条裂隙） | 隐伏点 MAE | **36.69°** | [`results/honest_leaderboard/l1_local__beishan_22.json`](results/honest_leaderboard/l1_local__beishan_22.json) |
| L0 | 同上 | 组系表模态误差（K=12，观测-only k-means） | **9.82° ± 0.66°** | [`results/global_honest_leaderboard/beishan.json`](results/global_honest_leaderboard/beishan.json) |
| L1 | 钻孔成像（FORGE，2 井，4328 条） | 隐伏点 MAE | **39.70° ± 0.33°** | [`results/global_honest_leaderboard/forge.json`](results/global_honest_leaderboard/forge.json) |
| L1 | 钻孔成像（FORGE） | 组系表模态误差 | **12.37° ± 4.00°** | [`results/global_honest_leaderboard/forge.json`](results/global_honest_leaderboard/forge.json) |
| L1 | DFN 基准（DECOVALEX，1089 条） | 组系表模态误差（K=4） | **0.05°** | [`results/global_honest_leaderboard/decovalex.json`](results/global_honest_leaderboard/decovalex.json) |
| L1 | 同上，带 `fracture_id`（路线 B） | 隐伏点 MAE | **0.0054°** | [`results/decovalex_routeB.json`](results/decovalex_routeB.json) |
| L3 | 稠密点云（合成四壁基准） | 隐伏点 MAE | **0.37°** | [`results/pointcloud_gate.json`](results/pointcloud_gate.json) |

\* *诚实协议* = BlindInput 评测：隐伏点对预测器完全不可见，掩码固定
（`obs_frac=0.4`, `rng=999`），指标为隐伏点上 `mean acos(|<pred, true>|)`，
10 个随机种子；评测框架内置毒丸/自泄漏审计。

口径注：FORGE 组系表数字按随仓结果文件原样引用（池化 obs-only k-means，
10 种子）。项目的类型感知重建管线在同一数据上报告 **11.05° ± 2.14°**——
两者都是修复 bug 后的诚实口径，差别在分组协议，不在正确性。

**这张表的工程读法**：无标签钻孔数据的逐点重构天花板在 ~37–49°——这是
信息极限，不是模型不行。但**组系表档位**（每组 ≥5 条观测，K=4–12）的
模态方向误差可达 **7–12°**，落在 ≤12° 的工程阈值之内。上面那条产品链
交付的就是这个档位——这也正是勘察报告实际使用的统计粒度。

## 诚实性与负结果

本项目测量了自己的天花板并公开发布——带失败模式的有界声明，比没有证据的
无限声明对工程师更有价值。要点：

- **早期一个"13°"结论是泄漏造成的。** 在防泄漏的 BlindInput 协议下，同一
  方法得 **36.69°**，而不是当初报告的 13°。修正历史被保留而非抹除（见
  `CHANGELOG.md`）。
- **无标签信息天花板约 31.7° ± 0.5°**（逐点口径）：相距不到四分之一半径
  的两条迹线，法向可以差 ~30°——没有结构信号（`fracture_id`、迹线连通）
  时，任何模型都无法恢复指派。
- **神经网络没有打赢几何基线。** 等变消息传递模型得分*劣于*几何
  `l1_local` 预测器，且被同一信息天花板封顶——作为有记录的负结果发布。
- **多走向稠密点云：BLOCKED-BY-DATA。** 2026-08 的公开数据调研没有发现
  同时满足"真实 + 多走向 + 三维点云 + 可下载 + 带真值"的数据集。L3 档
  结论基于合成四壁基准（MAE 0.37°，错分率 0.4%），不外推到壁面分离以外
  的场景。
- **废弃数字只标注、不偷换。** 凡因 bug 修复而改变的公开数字（如 FORGE
  管线的 sin/cos 互换），新旧值都可在审计记录中追溯，结果文件带
  `post-fix` 来源注记。

## 仓库结构

```
src/fractureflow/      核心库（上述算法）
scripts/               产品链 + demo + 几何约定守卫
examples/              已提交的 demo 产物（报告、组系表、图件）
tests/                 单测 + 回归套件（CI: pytest 3.10/3.11）
results/               支撑基准表的冻结结果快照
data/                  可再分发的派生数据（见数据政策）
docs/                  GitHub Pages 门户源码（交互式 DFN 演示）
```

所有代码与文档均为纯文本 Python / Markdown。仓库中唯一的二进制文件是
数据样本（`data/**/*.pt`，派生自开放许可的 FORGE 测井）和图件（`*.png`），
没有任何代码以二进制形式发布。

## 数据政策

按合规审计（2026-08-29），第三方数据分三类裁定（完整表格见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)）：

- **收录**——来自开放许可源的派生文件（Utah FORGE，CC BY 4.0，署名见
  声明文件）与项目合成样例（`data/real/r60_wells_csv/`）。
- **仅链接**——DECOVALEX、FORGE 原始编录、Pontrelli、GeoCrack、INEL-1、
  EGS Collab、OpenTopography 点云。请从声明文件引用的官方渠道下载。
- **排除**——北山预选区数据（及含北山井的混合文件）**不可再分发**，
  未收录于本仓库。

仓库**数据自足**：demo 与测试套件完全使用合成或已收录数据。

## 测试

```bash
python -m pytest tests -q
```

CI 在 Python 3.10/3.11 上运行全套测试 + 几何约定 grep 守卫。套件覆盖
几何约定（倾角↔法向往返、Terzaghi 权重、符号对齐）、数据泄漏守卫、
DFN/渗流不变量、报告生成、多井联合决策规则，以及发布守卫本身。需要
"仅链接"真实数据的回归测试在数据缺席时按记录的原因跳过——逐条登记表
见 `RELEASE_VERIFICATION.md`。

## 引用

见 [`CITATION.cff`](CITATION.cff)。若在学术工作中使用本仓库，请引用软件
本身及底层数据集（声明文件 §3）。

## 许可

MIT——见 [`LICENSE`](LICENSE)。第三方组件与数据裁定：
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 致谢

本工作在中国矿业大学（北京）大学生创新创业训练计划支持下完成，指导教师：
刘鹏教授。

作者：易嘉诚，2998812494@qq.com。
