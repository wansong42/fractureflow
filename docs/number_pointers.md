# 数字溯源附表面件 (number pointers)

本页公开读数的**反查关系**不再以内联属性承载，改由本件给出。每条 = 页面上的读数 → 台账锚 / 源件指针 / 口径限定语。
用法：在下面的表里搜索该读数（`36.69°`、`9.82±0.66°`…）。机器可读版：`number_pointers.json`。

口径提醒：本表是**溯源记录**，不是精度承诺；带 `NNS-` 的条目指向项目数字台账，`#` 后为 JSON 子键。

诚实标注：末列说明该行点名的路径**能否在本仓库内打开**。打不开的行指向上游证据库——本项目随发布只分发聚合读数（`results/` 冻结子集）与示例产物（`examples/`）。把这类行说成「点一下就能看到原始数据」会是假的，因此它们在此只作为**可核验性的声明**存在：需要核验上游证据，请按 `SECURITY.md` / 仓库 issue 的渠道提出。

| 号 | 屏 | 页面读数 | 台账锚 / 源件指针 / 口径 | 本仓可打开? |
|---|---|---|---|---|
| T001 | - | 36.69° | NNS-001 · results/honest_leaderboard/l1_local__beishan_22.json#mae_mean (std ±1.20, 10 seed) | 否（上游证据库，未随本发布分发） |
| T002 | - | 9.82°±0.66 | NNS-017 · results/set_table_beishan.json#12.modal_err_mean (std ±0.66) | 否（上游证据库，未随本发布分发） |
| T003 | - | 0.37° | NNS-021 · results/pointcloud_gate.json#evaluation.mae_hidden_deg (K=4 多走向壁, 100k 点) | 否（上游证据库，未随本发布分发） |
| T004 | - | 0.0054° | NNS-020 · results/decovalex_routeB.json#4fracplus_example.hidden_mae_mean | 否（上游证据库，未随本发布分发） |
| T005 | ss1 | 36.69° | NNS-001 · results/honest_leaderboard/l1_local__beishan_22.json#mae_mean (std ±1.20, 10 seed) | 否（上游证据库，未随本发布分发） |
| T006 | ss1 | 36.69–39.88° | NNS-001~NNS-008 · results/honest_leaderboard/ 各方法 mae_mean 最小/最大 | 是 |
| T007 | ss1 | 16.12° | NNS-015 · results/p1_honest_decoder.json (kmeans_obs K=6 strict) | 否（上游证据库，未随本发布分发） |
| T008 | ss1 | 20.6° | 差值 NNS-001 − NNS-015（两端实测相减） | 否（上游证据库，未随本发布分发） |
| T009 | ss1 | 12.37° | NNS-011 · results/phase0_oracle_floor_strict.json#beishan_22.12.mae_mean | 否（上游证据库，未随本发布分发） |
| T010 | ss1 | ~31.7°±0.5° | NNS-045 · results/l4_evidence_grading.json#threshold_sources.information_ceiling_31_7deg.value（L1 留一探针，非方法误差） | 否（上游证据库，未随本发布分发） |
| T011 | ss1 | ≈3.8° | 差值 NNS-015 − NNS-011（两端实测相减） | 否（上游证据库，未随本发布分发） |
| T012 | ss1 | 0.0054° | NNS-020 · results/decovalex_routeB.json#4fracplus_example.hidden_mae_mean | 否（上游证据库，未随本发布分发） |
| T013 | ss1 | 49.27±0.84° | results/global_honest_leaderboard/decovalex.json#point_eval.mae_mean (std ±0.84, l1_local) | 否（上游证据库，未随本发布分发） |
| T014 | ss1 | 0.37° | NNS-021 · results/pointcloud_gate.json#evaluation.mae_hidden_deg (K=4 多走向壁, 100k 点) | 否（上游证据库，未随本发布分发） |
| T015 | ss1 | 0.40% | results/pointcloud_gate.json#evaluation.misclass_rate_hidden (&lt;15% 阈值 PASS) | 否（上游证据库，未随本发布分发） |
| T016 | ss1 | 1.47° | NNS-022 · results/l4_fuse_gate.json#fused_result.modal_err_deg | 否（上游证据库，未随本发布分发） |
| T017 | ss1 | 18.22° | results/l4_fuse_gate.json#best_single_source | 否（上游证据库，未随本发布分发） |
| T018 | ss1 | 91.9% | NNS-022 · results/l4_fuse_gate.json (improvement 91.9%) | 否（上游证据库，未随本发布分发） |
| T019 | ss1 | 36.69–39.88° | NNS-001~NNS-008 · results/honest_leaderboard/ | 是 |
| T020 | ss1 | 12.37° | NNS-011 · results/phase0_oracle_floor_strict.json | 否（上游证据库，未随本发布分发） |
| T021 | ss2 | ~31.7°±0.5° | NNS-045 · results/l4_evidence_grading.json#threshold_sources.information_ceiling_31_7deg.value（局部真值 L1 留一探针上限） | 否（上游证据库，未随本发布分发） |
| T022 | ss2 | 30.5° | docs/论文素材包/README.md L15 · docs/论文素材包/智能性论证章节.md §2.2（局部真值 L1 留一探测） | 否（上游证据库，未随本发布分发） |
| T023 | ss2 | 36.28° | docs/论文素材包/智能性论证章节.md §2.2 表（复跑 36.48°） | 否（上游证据库，未随本发布分发） |
| T024 | ss2 | 34.63° | docs/论文素材包/智能性论证章节.md §2.2 表（几何基线） | 否（上游证据库，未随本发布分发） |
| T025 | ss2 | 35.19° | docs/论文素材包/智能性论证章节.md §2.2 表（复跑 34.84°） | 否（上游证据库，未随本发布分发） |
| T026 | ss2 | 33.27° | docs/论文素材包/智能性论证章节.md §2.2 表（复跑 32.95°） | 否（上游证据库，未随本发布分发） |
| T027 | ss2 | ≤0.1° 收益 | docs/13度目标可达性审计.md §7.4 · results/skeptic_repro/（最大收益 ≤0.1°） | 否（上游证据库，未随本发布分发） |
| T028 | ss2 | 全部不如 k-means | docs/论文素材包/关键数字.md §2.2 · results/p1_honest_decoder.json | 否（上游证据库，未随本发布分发） |
| T029 | ss3 | 0.05° | NNS-018 · results/set_table_decovalex.json#4.modal_err_mean (std ±0.0) | 否（上游证据库，未随本发布分发） |
| T030 | ss3 | 9.82±0.66° | NNS-017 · results/set_table_beishan.json#12.modal_err_mean (std ±0.66) | 否（上游证据库，未随本发布分发） |
| T031 | ss3 | 11.05±2.14° | NNS-019 · results/t58_b2b3_forge_rebuild.json#new_numbers.K12.mean (std ±2.14, T58 法向修复后复算) | 否（上游证据库，未随本发布分发） |
| T032 | ss3 | ≤12° | docs/_archive_已完成使命报告/A1-B3_诚实评测与组系表产品化_总结报告.md（三份站点均 &lt; 12° 阈值的判定记录；该句原为对外承诺语气，已按 R279 从公开面撤除，此处仅作溯源记录，不构成口径）· 数字台账 NNS-051 | 否（上游证据库，未随本发布分发） |
| T033 | ss4 | 12.37° | NNS-025 · results/t51_type_isolation_revalidation.json#new_numbers.mixed（T58 重验） | 否（上游证据库，未随本发布分发） |
| T034 | ss4 | 9.29° | NNS-025 · results/t51_type_isolation_revalidation.json#new_numbers.isolated（T58 重验） | 否（上游证据库，未随本发布分发） |
| T035 | ss4 | +3.08° | NNS-025 · results/t51_type_isolation_revalidation.json#new_numbers.improvement（预注册闸门 ≥0.5° PASS） | 否（上游证据库，未随本发布分发） |
| T036 | ss4 | 18.22° | results/l4_fuse_gate.json#best_single_source | 否（上游证据库，未随本发布分发） |
| T037 | ss4 | 1.47° | NNS-022 · results/l4_fuse_gate.json#fused_result.modal_err_deg | 否（上游证据库，未随本发布分发） |
| T038 | ss4 | −91.9% | NNS-022 · results/l4_fuse_gate.json#gates.gate2_improvement.improvement_pct（3/3 GATES PASS） | 否（上游证据库，未随本发布分发） |
| T039 | ss4 | 0.214 | NNS-023 · results/l4_ensemble_gate.json#ensemble_result.p32_crit_interval.median | 否（上游证据库，未随本发布分发） |
| T040 | ss4 | 0% | NNS-042 · results/l4_calibration.json#raw_coverage | 否（上游证据库，未随本发布分发） |
| T041 | ss4 | 90.6% | NNS-024 · results/l4_calibration.json#recalibrated_coverage（目标 90%） | 否（上游证据库，未随本发布分发） |
| T042 | ss4 | 0% → 90.6% | NNS-042 → NNS-024 · results/l4_calibration.json | 否（上游证据库，未随本发布分发） |
| T043 | ss4 | 89.77% | NNS-043 · results/l4_calibration.json | 否（上游证据库，未随本发布分发） |
| T044 | ss4 | 87.5% | NNS-044 · results/l4_calibration.json | 否（上游证据库，未随本发布分发） |
| T045 | ss4 | 36.69–39.88° | NNS-001~NNS-008 · results/honest_leaderboard/ | 是 |
| T046 | ss4 | ~31.7°±0.5° | NNS-045 · results/l4_evidence_grading.json | 否（上游证据库，未随本发布分发） |
| T047 | ss4 | 约 5° | 差值 NNS-001 − NNS-045（两端实测相减） | 否（上游证据库，未随本发布分发） |
| T048 | ss4 | 19.47° | NNS-031 · results/p0_full_audit.json#multiwell strict K=12 | 否（上游证据库，未随本发布分发） |
| T049 | ss4 | 19.49° | NNS-014 · results/p0_full_audit.json#single strict K=12 | 否（上游证据库，未随本发布分发） |
| T050 | ss5 | 25.76° | results/v_r57_extrapolation_interwell.json#cells.w1.modal_err.mean | 否（上游证据库，未随本发布分发） |
| T051 | ss5 | 21.94° | results/v_r57_extrapolation_interwell.json#cells.w2.modal_err.mean | 否（上游证据库，未随本发布分发） |
| T052 | ss5 | 19.82° | results/v_r57_extrapolation_interwell.json#cells.w5.modal_err.mean | 否（上游证据库，未随本发布分发） |
| T053 | ss5 | 19.30° | results/v_r57_extrapolation_interwell.json#cells.w10.modal_err.mean | 否（上游证据库，未随本发布分发） |
| T054 | ss5 | 18.07° | results/v_r57_extrapolation_interwell.json#cells.loo.modal_err.mean | 否（上游证据库，未随本发布分发） |
| T055 | ss5 | 1.24° | 差值 cells.w10 − cells.loo = 1.24° | 否（上游证据库，未随本发布分发） |
| T056 | ss5 | 6.46° | 差值 cells.w1 − cells.w10 = 6.46° | 否（上游证据库，未随本发布分发） |
| T057 | ss5 | 3.6 | results/v_r47_cost_curve.json#L0_wells_wholewell_1.n_obs_mean | 否（上游证据库，未随本发布分发） |
| T058 | ss5 | 1% | results/v_r47_cost_curve.json#L0_wells_wholewell_1.p10_axis.acc_site_strict | 否（上游证据库，未随本发布分发） |
| T059 | ss5 | 7.0 | results/v_r47_cost_curve.json#L0_wells_wholewell_2.n_obs_mean | 否（上游证据库，未随本发布分发） |
| T060 | ss5 | 40% | results/v_r47_cost_curve.json#L0_wells_wholewell_2.p10_axis.acc_site_strict | 否（上游证据库，未随本发布分发） |
| T061 | ss5 | 13.8 | results/v_r47_cost_curve.json#L0_wells_wholewell_4.n_obs_mean | 否（上游证据库，未随本发布分发） |
| T062 | ss5 | 97% | results/v_r47_cost_curve.json#L0_wells_wholewell_4.p10_axis.acc_site_strict | 否（上游证据库，未随本发布分发） |
| T063 | ss5 | 28.2 | results/v_r47_cost_curve.json#L0_wells_wholewell_8.n_obs_mean | 否（上游证据库，未随本发布分发） |
| T064 | ss5 | 100% | results/v_r47_cost_curve.json#L0_wells_wholewell_8.p10_axis.acc_site_strict | 否（上游证据库，未随本发布分发） |
| T065 | ss5 | 40.5 | results/v_r47_cost_curve.json#L3_wells_wholewell_4.n_obs_mean | 否（上游证据库，未随本发布分发） |
| T066 | ss5 | 71% | results/v_r47_cost_curve.json#L3_wells_wholewell_4.p10_axis.acc_site_strict | 否（上游证据库，未随本发布分发） |
| T067 | ss5 | 81.9 | results/v_r47_cost_curve.json#L3_wells_wholewell_8.n_obs_mean | 否（上游证据库，未随本发布分发） |
| T068 | ss5 | 76% | results/v_r47_cost_curve.json#L3_wells_wholewell_8.p10_axis.acc_site_strict | 否（上游证据库，未随本发布分发） |
| T069 | ss5 | 160.3 | results/v_r47_cost_curve.json#L3_wells_wholewell_16.n_obs_mean | 否（上游证据库，未随本发布分发） |
| T070 | ss5 | 82% | results/v_r47_cost_curve.json#L3_wells_wholewell_16.p10_axis.acc_site_strict | 否（上游证据库，未随本发布分发） |
| T071 | ss5 | 322.3 | results/v_r47_cost_curve.json#L3_wells_wholewell_32.n_obs_mean | 否（上游证据库，未随本发布分发） |
| T072 | ss5 | 90% | results/v_r47_cost_curve.json#L3_wells_wholewell_32.p10_axis.acc_site_strict | 否（上游证据库，未随本发布分发） |
| T073 | ss5 | 642.8 | results/v_r47_cost_curve.json#L3_wells_wholewell_64.n_obs_mean | 否（上游证据库，未随本发布分发） |
| T074 | ss5 | 98% | results/v_r47_cost_curve.json#L3_wells_wholewell_64.p10_axis.acc_site_strict | 否（上游证据库，未随本发布分发） |
| T075 | ss5 | 10 口井 | results/v_r57_extrapolation_interwell.json#cells.w10 | 否（上游证据库，未随本发布分发） |
| T076 | ss5 | 19.3° | results/v_r57_extrapolation_interwell.json#cells.w10.modal_err.mean | 否（上游证据库，未随本发布分发） |
| T077 | ss5 | 1.2° | 差值 cells.w10 − cells.loo = 1.24° | 否（上游证据库，未随本发布分发） |
| T078 | ss5 | 8 口 | results/v_r47_cost_curve.json#L0_wells_wholewell_8 | 否（上游证据库，未随本发布分发） |
| T079 | ss5 | 32 口 | results/v_r47_cost_curve.json#L3_wells_wholewell_32 | 否（上游证据库，未随本发布分发） |
| T080 | ss6 | beishan Auto-K=4 | results/dfn_pipeline_final.json#set_table（beishan 22 井 880 法向, Auto-K=4） | 否（上游证据库，未随本发布分发） |
| T081 | ss6 | 0.240 [0.215, 0.276] | results/dfn_pipeline_final.json#percolation.p32_crit [lower, upper] | 否（上游证据库，未随本发布分发） |
| T082 | ss6 | 3.5（假设） | results/dfn_pipeline_final.json#percolation.beta（假设值; 迹长拟合可收窄, 见关键数字 §2.7） | 否（上游证据库，未随本发布分发） |
| T083 | ss6 | 50 × 50 × 50 m | results/dfn_pipeline_final.json#input.domain_m | 否（上游证据库，未随本发布分发） |
| T084 | ss6 | 13644 → 2000 盘 | data/dfn_demo_discs.json#meta.sampling（冻结生成器 + 固定种子确定性重排抽样） | 否（上游证据库，未随本发布分发） |
| T085 | ssetl | 详见 JSON honest_boundary 字段 | results/dfn_pipeline_final.json#honest_boundary | 否（上游证据库，未随本发布分发） |
| T086 | ss7-s | 4 起 | docs/论文/论文全文_EN_v0.7.md §Leakage（four leakage incidents） | 否（上游证据库，未随本发布分发） |
| T087 | ss7-s | 18 条 | results/digital_ledger.json#deprecated（DEP-001 ~ DEP-018 新旧对照记录） | 否（上游证据库，未随本发布分发） |
| T088 | ss7-s | 边界详情 | docs/论文素材包/诚实边界.md §3.1 · results/pointcloud_gate.json | 是 |
| T089 | ss7-s | 边界详情 | results/dfn_pipeline_final.json#honest_boundary | 否（上游证据库，未随本发布分发） |
| T090 | ss7-s | 边界详情 | docs/论文/论文全文_EN_v0.7.md §6.6 · results/l4_calibration.json | 否（上游证据库，未随本发布分发） |
| T091 | ss7-s | 台账条目 | NNS-045 · results/digital_ledger.json | 否（上游证据库，未随本发布分发） |
| T092 | ss7-s | 本屏水印声明 | data/recon_line.json#meta.watermark（随页分发） | 否（上游证据库，未随本发布分发） |
| T093 | ss7-s | ' + s.dip_mean_deg + '° | data/dfn_demo_discs.json#sets[' + k + ']（展示实现统计） | 否（上游证据库，未随本发布分发） |
| T094 | ss7-s | ' + s.dipdir_circmean_deg + '° | data/dfn_demo_discs.json#sets[' + k + ']（展示实现统计，圆均值） | 否（上游证据库，未随本发布分发） |
| T095 | ss7-s | ' + s.kappa +
      ' | results/dfn_pipeline_final.json#set_table.concentrations[' + k + '] | 否（上游证据库，未随本发布分发） |
| T096 | ss7-s | ' + (s.proportion_target * 100).toFixed(1) + '% | results/dfn_pipeline_final.json#set_table.proportions[' + k + '] | 否（上游证据库，未随本发布分发） |
| T097 | ss7-s | ' + s.n_displayed + ' | data/dfn_demo_discs.json#sets[' + k + '].n_displayed | 否（上游证据库，未随本发布分发） |
| T098 | ss7-s | ' + esc(o.t) + ' | ' + esc(o.src) + ' | 否（上游证据库，未随本发布分发） |
| X001 | s4 | 多井池化：FORGE 双井联合相对单井为劣化（反向读数） | results/forge_joint_strict_audit.json#comparison_with_original.honest_improvement · 台账锚 NNS-037 · 旧红利读数已判废 DEP-013 · 数值属研究线读数，公开页不展示 | 否（上游证据库，未随本发布分发） |
| X002 | s1 | 指派缺口（不知真值时） | 台账锚 NNS-185 (K=4) / NNS-180 (K=12) · results/r201_identifiability_theory/reconciliation.json#rows[id=S-04] · 集合档口径，非精度声明 | 否（上游证据库，未随本发布分发） |

本表共 100 行，其中 **4 行**点名的路径可在本仓库内直接打开。
—— 由 `scripts/r279_publish_derive.py`（主树派生链）生成；与页面同版本发布，勿手工维护。

