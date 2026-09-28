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

### 3.1 v0.1.1 追加登记（2026-09-19）

发布面由**机算依赖闭包**决定（入口 = `0.1.0` 已发布的 `scripts/*.py` 产品件，
递归 import + 已核实的 subprocess 边），不手工挑件。闭包 22 件 = **入库 9 件**
（5 改 + 4 新）+ **不动 13 件**。下表前两行登记入库件所重放的 §3 适配与脱敏，
后四行登记不动件的判定理由：

| 文件 | 处置 | 原因 |
|---|---|---|
| `scripts/dfn_from_borehole.py`、`scripts/full_pipeline.py` | 重放 §3 适配 | 主树侧文档示例重新指回 `data/real/beishan_wells.npz` / `loaded_real_nets_setid.pt`；已按 §3 原文改回通用占位名 |
| `scripts/auto_label_borehole.py` | 脱敏 3 处 | 新增行的注释/产物元数据 `provenance` 字段含内部任务证据件路径；已改为不含路径的表述，计算逻辑零改动 |
| `scripts/console_safety.py` | 脱敏 1 处 | 新增件 docstring 含内部任务证据件路径；已改为不含路径的表述 |
| `scripts/demo_run.py` | 不动 | 与 `0.1.0` 的差异**恰为** §3 登记的合成样本适配本身（主树自 2026-08-19 无产品改动）；拷入等于把演示数据依赖回退到禁止分发件 |
| `src/fractureflow/borehole_report.py` | 不动 | 差异恰为 §3 登记的 `sys.executable` 适配；拷入等于泄漏本机绝对路径 |
| `scripts/forge_fmi_pipeline.py` | **不发** | 在闭包内（`_em_group_table_csv` 惰性 import），但 §2 已按研究线脚本剔除并为此排除 `test_p15_deliverable_fixes.py`；其注释含内部裁定词汇。公开面扩大不可逆、后补可逆 ⇒ 维持剔除。已知边界见 `CHANGELOG.md` [0.1.1] |
| 其余 10 件 | 不动 | 与已提交 blob **逐字节相同**（差异只是 `core.autocrlf` 造成的工作树行尾） |

卫生检查比本仓扫描器额外多查三类（判废数字家族按权威清单直读、本机绝对路径、
指向未随仓证据件的具体路径），且**只扫新增行** —— 已公开存量不重复追究；
应用后复扫零残留。`EXEMPTIONS` 豁免表本轮**一字未改**。

### 3.2 v0.1.2 追加登记（2026-09-19，发布面口径清偿）

本轮**零代码改动**，只改 `docs/index.html` 的对外措辞与四处发布元数据。判定与执行
记在内部台账（R279），仓内只登记结果面：

| 项 | 处置 | 依据 |
|---|---|---|
| ③ 屏标题与"对外承诺"句 | 改判为研究线精度线，数值承诺句删除 | 对外零量化承诺（长期规则） |
| 三条组系表读数 | 逐行加 `TRUTH` / `CIRCULAR` 证据档，CIRCULAR 行同框写明循环对象 | 证据分级要求 |
| "贴住地板"句式 | 改为"须 oracle 指派才可达、不可部署" | 该句式已被判废 |
| 汇率表未注册的百分比派生值 | 删除，且**不**补任何替代数字 | 无源比例一律不公开 |
| 连通判定"近阈值场地"误称 | 改为高强度档（远阈值 +35%），与其上方表格一致 | 口径自纠 |
| 多井联合负结果 | 补源件指针并披露"反向"，数值不入公开页 | 诚实性要求 + 零新增公开数字 |
| 死指针（已归档报告） | 改指归档路径 | 指针可得性 |
| 历史里的旧承诺 | **不改写 git 历史**，只在 README/CHANGELOG 声明"只认当前版本" | 共享公开仓 force push 属高危不可逆 |

数字守恒（本轮机器判据）：`docs/index.html` 改正后的小数与整数百分比读数集合是改正
前的**子集**（只删不加），诚实限定语计数不降。两项均可一键复算。`EXEMPTIONS` 豁免表
本轮一字未改。
### 3.3 v0.1.2 追加登记（发布面溯源改道）

本轮**仍零代码改动**，只改公开 HTML 的溯源承载方式，并新增两个随页附表面件。
数字全部由内部复算件直读（这些复算件按发布纪律不随本仓分发）：

| 项 | 读数 | 说明 |
|---|---|---|
| 剥除的 `data-src` | 98（解析属性 92 + 脚本模板内组装 6） | 内联属性不再出现在公开 HTML；字面计数含 CSS `attr(data-src)` 一处在被删的悬停规则里 |
| `results/` 路径提及 | 89 -> 0 | 仓内不可解析的假反查入口清零；页面主张数字未变 |
| 附表面件行数 | 100 | 98 条来自被剥属性 + 2 条声明式补登记（指针原本写在散文里的反向读数） |
| 附表面件红线 | 绝对路径 0 / 判废数字非退役语境 0 / 混标闸非豁免命中 0 | 附表面件本身就是对外面，按 R263 + 台账判废清单 + 绝对路径三条复扫 |
| 派生可重放 | 独立进程 ×3 逐字节一致 | 发布件是产物不是第二份源；页面 sha `02fd39049b63f5aaa116376cc1bfa21f04516b3a9a7f682686b75424467d1683` |
| 对外承诺词表 | 主树门户与公开面 L1 断言级同为 0 | 沿用 R286 三级语义与冻结词表，不按裸 token 计数判红 |

反向指针位置变更已预登记：`RS-8 / RS-9` 的「源件指针在场」判据满足域由页面内联改为
「页面 + 随页附表面件」，判据强度未变（同仓同页一跳可达），见本单 `prereg_v2.json`。


### 3.4 追加登记（README 对外措辞清偿 + 常驻把守）

本轮**零代码改动、零数值改动、零新增研究数字**，只改 README 两个语言版本的对外
措辞，并新增一条常驻把守。判定与逐件证据在内部证据件（readmefix_20260925，
按发布纪律不随本仓分发），此处只登记结果面：

| 项 | 处置 | 依据 |
|---|---|---|
| 阶梯表把逐点档 MAE 与组系表模态误差并排 | 两种估计量分表；逐点档留主表 | 不同 estimand 不可同列直比 |
| 两条 K=12 组系表读数（beishan / FORGE） | 出正文承诺位，移入标注“非承诺线”的附录小节，逐行写明循环对象（各站点自身记录法向的 k-means 表，与打标同一估计器）与该统计量对 K 的上限与退化行为 | 一致性读数须封顶 K ≤ n_obs/2，K=12 只能进附录 |
| 三条组系表读数的证据级 | 逐行加 TRUTH / CIRCULAR 与循环对象 | 无独立分类的站点只可称一致性 |
| DECOVALEX 0.05° | 标为**一致性**：随仓读数（`set_table`，K=4）的参照是同源同估计器的观测点 k-means 表，属循环口径；族标签真值重测是另一个读数，本页不印 | 集合档准入裁定对“第二重循环”的判定；站点可称 TRUTH，本行读数不可 |
| “7–12° 且落在 ≤12° 阈值内 / 产品链交付的就是这个档位” | 改为“支持与不支持”段：阈值保留为验收判据，读数不再被表述为满足阈值；无源的 7–12° 区间整体删除（**不**补替代数字） | 对外零量化承诺；无源数字禁入对外材料 |
| FORGE 行 12.37° | 加撞值提醒：同串数字在本仓另指“须 oracle 指派才可达的逐点误差地板（不可部署）”与“混池分组口径”，均非本行 | 撞值消歧 |
| 常驻把守 | 新增对外面扫描器：集合档读数无级别标签或未写循环对象、或落在交付能力句式内 ⇒ 判红；含底本无关毒丸 | 一个 bug 换一条规则 |

数字守恒（机器判据）：两个语言版本的**小数读数集合**改后是改前的子集，新增 = 0；
被删的小数与整数读数逐条登记在内部证据件 `artifacts/number_delta.json`。
本轮未触碰 `CHANGELOG.md` 与 `docs/index.html`（不在授权写面内），其后果见 §6。

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
  重建路径）并存，由架构师在验收时定夺 README 是否换引。定夺结果（README
  侧）：两行**都保留**并同框标为一致性（CIRCULAR），因为它们量的是同一个
  一致性量的两种分组协议，换引任意一行都不解决证据级问题。
- `global_honest_leaderboard/decovalex.json` 的 set-table=0.05°（K=4）**不是**
  族标签真值读数：其参照为同源同估计器的观测点 k-means 表（循环口径）。因此
  README 该行标为一致性；而 `docs/index.html` ③ 屏把同一读数标为 TRUTH，
  **两处目前互矛**（该文件不在本轮写面内，已如实移交页面属线处理）。
- 对外面上 `12.37°` 这一串数字在本仓承载三个不同对象（FORGE 组系表一致性 /
  逐点档 oracle 指派误差地板 / 混池分组口径）。`docs/number_pointers.md` 的
  T009 / T020 / T033 三条各指其一，引用时须随带对象说明。

## 7. push 前待办（PUSH_GUIDE 步骤 2）

`LICENSE` 版权行、`CITATION.cff` 作者与仓库 URL、`SECURITY.md` 联系邮箱
——三处占位符由甲方定稿后方可公开。
