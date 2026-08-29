# Changelog

All notable changes to this repository are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/) + semantic versioning.

## [0.1.0] - 2026-08-29

Initial public release (push-ready tag; public push is performed manually by
the project owner — see `PUSH_GUIDE.md`).

### Added
- Core library `src/fractureflow/`: geometry conventions, label-free set
  labeling, point predictors, honest BlindInput evaluation harness with
  poison-pill leak audit, set-table evaluation, DFN generation +
  percolation screening, multi-well site model, multi-source fusion (L4),
  equivariant network backbones (released as a documented negative result).
- Product chain: `scripts/full_pipeline.py` (LAS → label → DFN → percolation
  → report), `scripts/auto_label_borehole.py`, `scripts/dfn_from_borehole.py`,
  `scripts/demo_run.py` (one-click self-contained demo),
  `scripts/check_geometry_conventions.py` (grep gate).
- Frozen result snapshots under `results/` (whitelisted per the open-source
  compliance audit of 2026-08-29) and redistributed derived data under
  `data/` (CC BY 4.0 attribution in `THIRD_PARTY_NOTICES.md`).

### Frozen anchors quoted at release time
- Honest point-level MAE (l1_local, BlindInput, obs_frac=0.4, rng=999,
  10 seeds): **36.687°** (loaded mixed-real cohort; 36.6871 ± 1.1965 on the
  22-well beishan cohort) — `results/honest_leaderboard/l1_local__beishan_22.json`
- Honest oracle floor (K=12, observed-only grouping + true assignment):
  **12.37°** (beishan cohort)
- L3 point-cloud multi-orientation gate: **0.37°** hidden MAE, 0.4%
  misclassification, PASS — `results/pointcloud_gate.json`
- Route B (fracture_id) on DECOVALEX 4-frac_plus: **0.0054°** ± 0.0001 —
  `results/decovalex_routeB.json`

### Quality gates at release time (2026-08-29, source-project terminal rerun)
- selfcheck: 33/35 PASS (2 failures are pre-existing ledger/integration
  accounts, documented in the project audit trail)
- pytest (root suite): 969 passed / 13 failed — every failure is a
  pre-existing ratchet/ledger account, individually attributed
- mutation testing: 6/6 killed (100%)
- reproducibility certification: 9/9 PASS, frozen anchors zero-drift
  (36.687 / 12.37 / 0.37)
- Release-copy gate: see `RELEASE_VERIFICATION.md`
  (test pass/skip registry, sensitivity scan, demo smoke log).

### Notes
- This release excludes internal research ledgers, client/business materials,
  and all data ruled link-only or forbidden by the compliance audit.
