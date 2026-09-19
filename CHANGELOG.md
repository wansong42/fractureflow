# Changelog

All notable changes to this repository are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/) + semantic versioning.

## [0.1.2] - 2026-09-19

Public-wording compliance release. It closes quantitative-commitment wording on
the GitHub Pages portal that contradicted the project's standing rule of **zero
quantitative public commitments**. No code path changes; no benchmark reading is
added, changed or strengthened anywhere in this repository.

### Changed
- `docs/index.html`, screen 3 — retitled from a commercial "commitment line" to a
  research-line accuracy line, marked **not an external commitment**. The sentence
  that read as an outward commitment (a per-group error band) and the sentence
  asserting that all three sites pass the engineering threshold are removed; the
  threshold itself stays, because a threshold is an operative acceptance
  criterion while the readings under it are not.
- `docs/index.html`, screen 3 — the three rows are now labelled `TRUTH` or
  `CIRCULAR`, and the reference object of each `CIRCULAR` row is stated in the
  same box: those two sites have no independent ground truth, so their numbers
  measure consistency against the same campaign's catalogue, not accuracy.
- `docs/index.html` — a retired "hugging the floor" phrasing is replaced by the
  statement that the floor requires oracle assignment and is not deployable; an
  unregistered percentage derivation in the observation-cost section is deleted
  with no replacement figure published; a connectivity cell that was mislabelled
  as near-threshold is corrected to the far-threshold high-density cell it is in
  the table directly above; the multi-well sentence now discloses that the
  two-well joint run came out as a **degradation** and names its source file,
  without publishing that research-line value; one dead internal document
  pointer is repointed to the archived report.
- Machine check: the set of numeric tokens rendered on `docs/index.html` after
  this release is a **subset** of the set before it (deletions only), and the set
  of honesty-qualifier markers did not shrink. Both assertions are re-runnable.

### Added
- Standing rule (see `README.md` / `README_zh.md`): public wording is operative
  **only for the currently published version**. Git history deliberately is not
  rewritten — history in a shared public repository is irreversible to rewrite,
  and the audit trail matters more than tidiness. Any numeric commitment found
  in history, in superseded tags, or in a Pages cache is void.

### Not changed
- No source file, no product entry point, no test, no data file. The published
  accuracy figures of `0.1.0` / `0.1.1` remain exactly as they were.

## [0.1.1] - 2026-09-19

Engineering maintenance release. No new benchmark readings are introduced and
no precision claim in this repository is added, changed or strengthened: the
accuracy figures quoted by the project remain exactly those of `0.1.0`.

### Added
- `src/fractureflow/em_decoder.py` — polar Bingham-mixture EM decoder with a
  declared applicability domain (borehole axis required, effective-sample-size
  gate, K bounded by the number of observations). When a precondition is not
  met it degrades to an explicitly labelled plain variant instead of failing
  silently.
- `src/fractureflow/orthostats/` — orthonormal statistics layer, including the
  corrected Terzaghi weighting used by the labelling product.
- `scripts/console_safety.py` — one shared stdout/stderr encoding guard for all
  product entry points on Windows GBK consoles.

### Changed
- `scripts/auto_label_borehole.py` — new `--decoder kmeans|em` option; the
  default remains `kmeans`, so existing behaviour and outputs are unchanged.
  Borehole-axis resolution now discloses when a vertical approximation is used;
  input problems report the offending column/key and exit with status 2
  (`FRACTUREFLOW_DEBUG=1` restores full tracebacks).
- `scripts/borehole_excel_entry.py`, `scripts/dfn_from_borehole.py`,
  `scripts/full_pipeline.py` — console-encoding guard installed, failure exit
  status unified to 2, and customer-visible error text sorted so that repeated
  runs of the same input produce byte-identical output.
- `src/fractureflow/terzaghi.py` — `terzaghi_summary` accepts an explicit borehole
  axis and reports how many weights were clipped.

### Known boundary of this release
- The `--decoder em` group-table CSV writer imports `forge_fmi_pipeline`, which is
  deliberately not part of the release surface. The EM decoder itself works; only
  that optional CSV sub-path is unavailable here.

### Release-surface method
- The shipped file set was computed as the transitive import closure of the product
  entry points already published in `0.1.0`, not chosen by hand; refusals and
  no-ops are registered in the release verification ledger.

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
