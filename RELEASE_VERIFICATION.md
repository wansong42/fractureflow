# RELEASE_VERIFICATION — 开源发布仓自证台账 (R146)

> 构建日期 2026-08-29 ｜ 依据 `tasks/R145_任务书_★☆_开源合规清单...md`（裁定源）
> 与 `tasks/R146_任务书_★★☆_开源发布包构建...md`（构建单）。
> 本文件**随仓库分发**（豁免表与登记项是给仓库消费者的诚实声明）；
> 原始门禁日志在 `_release_checks/`（不入 git）。
> 主项目树（本仓库的源工作区）构建期零改动，仍为唯一权威。

## 1. 门禁结果 (T3, 构建环境亲跑)

| 门禁 | 结果 | 日志 |
|---|---|---|
| pytest（副本环境，git 转化后终态） | **111 passed / 0 failed / 0 skipped** | `_release_checks/pytest_full.log` |
| 敏感扫描 | 128 命中，**0 未豁免**（豁免表见 §4） | `_release_checks/scan_report.md` / `.json` |
| 合成管线冒烟（一键 demo） | PASS：内置 60 行 3 组合成样本 → 组系表（3 组×20 条，组内离散 4.3–5.2°）+ 玫瑰图 + 极点图 + Markdown 报告 | `_release_checks/demo_smoke.log` |

测试环境：Python 3.10（项目 GPU 训练环境），Windows；`PYTHONUTF8=1 KMP_DUPLICATE_LIB_OK=TRUE`（GBK 控制台双变量已写进 README 与 `run_demo.cmd`）。CI（GitHub Actions, ubuntu）在 push 后自动复跑同一套件。

## 2. 内容白名单（按 R145 裁定执行）

- **src/fractureflow/** 全量（64 模块运行时导入全绿），**唯一排除**：
  - `outcrop_trace.py` —— 硬耦合到未随仓的冻结研究线模块
    （`scripts/l2_outcrop_trace_pipeline.py` → `vision/extract.py`）。
    R145 §1.2 只标注了 imio 依赖（已按其建议处理）；第二个耦合是运行时
    导入冒烟发现的，属 R145 静态扫描盲区，如实登记。
- **scripts/** 产品链 + demo + 守卫（9 件）：`full_pipeline.py`、
  `auto_label_borehole.py`、`dfn_from_borehole.py`、`demo_run.py`、
  `read_forge_las.py`、`borehole_report.py`、`borehole_excel_entry.py`、
  `check_geometry_conventions.py`、`release_sensitivity_scan.py`。
  后两件是 demo 冒烟逐级暴露的依赖闭包（R145 §4.6 未点名），属产品链本体。
  其余 550+ 研究线脚本按 R145 建议全部不入仓。
- **tests/** 14 件（13 件核心回归 + 本守卫件）。**排除登记**：
  `test_c3_dryrun.py` / `test_pad_assembly_contract.py` /
  `test_ubi_assembly_contract.py`（依赖 fmi_attr，R145 §4.3 整体排除）、
  `test_p15_deliverable_fixes.py`（依赖研究线脚本 forge_fmi_pipeline /
  report_replay）、`test_neural_training_fixes.py`（依赖训练线脚本）、
  其余 `test_v_r*` 研究线测试（依赖未随仓的 results/ 产物与脚本）。
- **results/** 冻结数字白名单：`honest_leaderboard/`、
  `global_honest_leaderboard/`、`pointcloud_gate.json`、
  `decovalex_routeB.json`、`r110_b1/b1_scorecard.json`。
- **data/**：随仓分发件 = FORGE 派生 net（forge16A / forge2024_multi /
  forge2024meq_×2，CC BY 4.0 归属见 THIRD_PARTY_NOTICES §4）、
  `utah_forge_fmi/` 派生件（routeA.pt ×2 + group_table.csv ×2 +
  forge_fmi_2wells.pt + summary/survey JSON）、自产合成夹具
  `r60_wells_csv/`。禁止/只链接件全部不随仓（.gitignore 策略层拦截）。
- **未随仓（待架构师单独决定）**：`demos/portal/`（R145 F 类建议项，
  R146 任务书内容清单未纳入）。

## 3. 副本内代码编辑登记（全部为主树零改动的发布适配）

| 文件 | 编辑 | 原因 |
|---|---|---|
| `src/fractureflow/borehole_report.py` | 报告"环境"行：本机 conda python 绝对路径 → `sys.executable`（+`import sys`） | 机器特定路径（T2） |
| `src/fractureflow/v25_strategy.py` | `v25_official`/`h1_harness` 导入加守卫；缺模块时 `run_v25()` 友好报错 | 未随仓研究线依赖；包导入期保持 64/64 全绿 |
| `src/fractureflow/outcrop_trace.py`、`imio.py` | 已从发布剔除（imio vendoring 随之撤销） | 见 §2 排除登记 |
| `scripts/demo_run.py` | `generate_good_sample` 由 beishan npz 改为**确定性 3 组合成样本**（rng=42）；删除失去调用者的随机回退；内置样例 `--K` 默认 6→3 | 数据自足 + 禁止分发数据零依赖；K 与样本组数一致 |
| `scripts/full_pipeline.py`、`dfn_from_borehole.py` | 文档示例中 `beishan_wells.npz` / `loaded_real_nets_setid.pt` → 通用占位名 | 示例不再指向未随仓文件 |

冻结数字零改写；以上编辑均不触碰任何数值口径。

## 4. 敏感扫描：token 定义与豁免表

模式（R146 T3.2）`beishan|试点|NDA|客户|AGENTS|看板|架构师|交接|task_|R1dd`，
两处保意图收紧（脚本头有记录）：`NDA` 按大小写敏感缩写+词边界（防
"standard" 假阳）；`R1dd` 加词边界。冻结锚点数字（36.687/12.37/0.37）为
应保留科学内容，不属命中。扫描器自排除；台账与本守卫件按 FILE_OVERRIDES
登记（二者按设计引用模式串与禁止名单原文）。

128 个豁免命中全部落在五类（完整逐条清单见 `_release_checks/scan_report.md`）：

| 类别 | 理由 |
|---|---|
| `beishan`（场地名） | 科学语境的评测队列命名/数据政策声明。**场地数据本身**按 R145 §2.1 硬裁定零随仓（.gitignore 拦截）；`results/*.json` 内为聚合指标（冻结锚点），非原始或逐裂隙数据 |
| `R1dd`（线编号） | 冻结代码注释/结果溯源中的内部线编号，纯标识符，无任务书文本 |
| `客户` | docstring 描述编录表方言/演示用途，无客户名、无承诺话术 |
| `架构师` | 代码注释/冻结结果叙述中的决策署名词（如 K>8 硬闸、T30 判定注），无流程材料 |

## 5. 测试跳过登记

最终套件**零跳过**：随仓数据（FORGE 派生 net + r60 CSV）与合成夹具足以
覆盖全部 14 件测试。git 历史守卫在无 tag 环境（CI checkout）自动 skip
（属环境条件而非数据缺失）。

R146 守卫（`tests/test_v_r146_.py`，8 条）：结构完整 / 禁止数据零在场 /
.gitignore 政策标记 / LICENSE+NOTICES+CITATION / README 诚实节 /
零机器路径 / 扫描可复跑 / git 首提交+tag。

## 6. 引用数字的 provenance 注记

README 数据阶梯表只引用仓内 JSON 可直读的数字并附文件指针。
两点如实注记：
- `results/global_honest_leaderboard/pontrelli.json` 的 routeB=6.6 是
  R90.1 复算前的历史快照（其后口径为 vs 真平面 0.003° / vs 实测法向
  14.70°），README **不引用**该行；文件按冻结纪律原样随仓。
- `global_honest_leaderboard/forge.json` 的 set-table=12.37°±4.0 为该文件
  自标注 post-sin/cos-fix 口径；与主项目台账 NNS-019（11.05±2.14，另一
  重建路径）并存，由架构师在验收时定夺 README 是否换引。

## 7. push 前待办（PUSH_GUIDE 步骤 2）

`LICENSE` 版权行、`CITATION.cff` 作者与仓库 URL、`SECURITY.md` 联系邮箱
——三处占位符由甲方定稿后方可公开。
