# FractureFlow

Label-free structural-orientation statistics and honest machine-learning
benchmarks for rock-fracture data — from borehole logs to dense point clouds.

FractureFlow packages the pipelines of a research project on **intelligent
reconstruction of discontinuous structures in complex rock masses**. It turns
raw fracture observations into engineering deliverables (auto-labeled set
tables, rose diagrams, DFN screening reports) **without requiring
fracture-level ground-truth labels**, and ships the honest evaluation
harness — including the negative results and measured information ceilings
that bound what any method can extract from unlabeled data.

> **Honesty is a feature, not a bug.** This repository deliberately publishes
> negative results, deprecated numbers with their correction history, and the
> information ceilings that limit the whole problem class. See
> [Honesty & negative results](#honesty--negative-results).

---

## Why

Structural-orientation statistics (joint-set tables, rose diagrams, spacing
and connectivity screening) are a mandatory chapter in geotechnical
investigation reports — and are still produced almost entirely by hand. The
engineering value of this package is replacing that manual workflow:

```
borehole log (dip / dip-direction)  ->  auto-labeled fracture sets  ->  set table
                                    ->  DFN + percolation screening ->  report
```

The research value is the measurement of **how far** label-free inference can
go at each data tier — and where it provably cannot go further.

## The data ladder

Capability is organized by observation tier (L0 → L4). Numbers are frozen
anchors from the shipped result files (path given per row):

| Tier | Data | Metric (honest protocol*) | Value | Evidence |
|---|---|---|---|---|
| L0 | borehole logs (22 wells, 880 fractures) | hidden-point MAE | **36.69°** | [`results/honest_leaderboard/l1_local__beishan_22.json`](results/honest_leaderboard/l1_local__beishan_22.json) |
| L0 | same | set-table modal error (K=12, obs-only k-means) | **9.82° ± 0.66°** | [`results/global_honest_leaderboard/beishan.json`](results/global_honest_leaderboard/beishan.json) |
| L1 | borehole imaging (FORGE, 2 wells, 4328 fractures) | hidden-point MAE | **39.70° ± 0.33°** | [`results/global_honest_leaderboard/forge.json`](results/global_honest_leaderboard/forge.json) |
| L1 | borehole imaging (FORGE) | set-table modal error | **12.37° ± 4.00°** | [`results/global_honest_leaderboard/forge.json`](results/global_honest_leaderboard/forge.json) |
| L1 | DFN benchmark (DECOVALEX, 1089 fractures) | set-table modal error (K=4) | **0.05°** | [`results/global_honest_leaderboard/decovalex.json`](results/global_honest_leaderboard/decovalex.json) |
| L1 | same, with `fracture_id` (Route B) | hidden-point MAE | **0.0054°** | [`results/decovalex_routeB.json`](results/decovalex_routeB.json) |
| L3 | dense point cloud (synthetic 4-wall benchmark) | hidden-point MAE | **0.37°** | [`results/pointcloud_gate.json`](results/pointcloud_gate.json) |

\* *Honest protocol* = BlindInput evaluation: hidden points are never visible
to the predictor, masks are fixed (`obs_frac=0.4`, `rng=999`), and the metric
is `mean acos(|<pred, true>|)` over hidden points, 10 seeds. A
poison-pill/self-leak audit runs inside the harness.

Caliber note: the FORGE set-table figure quotes the linked result file as
shipped (pooled obs-only k-means, 10 seeds). The project's type-aware rebuild
pipeline reports **11.05° ± 2.14°** on the same data — both are honest
post-bug-fix calibers; they differ in grouping protocol, not in correctness.

The `fracture_id` row is the key honest contrast: when observations carry the
same-fracture grouping (Route B), the error collapses to measurement
precision (0.0054°). Without it — the L0/L1 reality — the same data tops out
at ~37–49° point-wise, and ~10–12° for set-table modality. **The bottleneck
is information, not model capacity.**

## Quick start

```bash
# 1. Environment (Python >= 3.10)
conda env create -f environment.yml && conda activate fractureflow
#    or: pip install -r requirements.txt   (CPU torch is sufficient)

# 2. One-click demo: synthetic borehole log -> label -> set table ->
#    rose/polar plots -> DFN -> percolation -> report  (fully self-contained)
./run_demo.sh            # Windows: run_demo.cmd

# 3. End-to-end product chain on your own data
python scripts/full_pipeline.py --input-las your_well.las --domain 50 50 50

# 4. Test suite
python -m pytest tests -q
```

Windows GBK consoles need two environment variables (already set inside
`run_demo.cmd`): `PYTHONUTF8=1` and `KMP_DUPLICATE_LIB_OK=TRUE`.

### What the demo produces

`scripts/demo_run.py` generates a synthetic borehole log internally, then
runs the full commercial chain: Route-A auto-labeling, group table CSV, rose
and polar diagrams, a DFN realization, percolation screening
(`p_conn(P32)` curve + scenario indicators), and a Markdown report. No
external data required.

### Main evaluation entry points

```bash
# honest point-level leaderboard entry (BlindInput protocol)
python -m fractureflow.eval --point-mode l1local
# set-table evaluation
python -m fractureflow.set_table_eval
# geometry-convention guard (grep gate over the package; used in CI)
python scripts/check_geometry_conventions.py
```

## Repository layout

```
src/fractureflow/      core library
  geometry.py            dip/dip-direction <-> normal conversions (single source of truth)
  setlabel.py            spherical k-means labeling (|cos| assignment + sign alignment)
  inference.py           point predictors (l1_local Fréchet median, set-aware, ...)
  honest_eval.py         BlindInput harness: poison pills + leak red-flag audit
  set_table_eval.py      group-table evaluation (Hungarian matching)
  dfn.py / percolation.py  Baecher-disk DFN + percolation screening
  site_model.py          multi-well site model + joint-vs-independent decision rule
  l4/                    multi-source conflict-gated fusion
  backbones/             equivariant message-passing networks (research line)
scripts/               product chain + demo + guard
tests/                 unit + regression suite
results/               frozen result snapshots (whitelisted; see THIRD_PARTY_NOTICES.md)
data/                  derived data that is licensed for redistribution
```

## Honesty & negative results

This project measured its own ceiling and publishes it. Highlights:

- **An early "13°" claim was a leak.** The original point-level target was
  based on an evaluation in which hidden points leaked into grouping. Under
  the honest BlindInput protocol the same method scores **36.69°**, not 13°.
  The correction history is preserved rather than erased.
- **The unlabeled information ceiling is ~31.7°±0.5°** (point-level,
  L1-iteration sensitivity). Two fracture traces closer than a quarter
  radius can have normals differing by ~30° — without structure signals
  (`fracture_id`, trace connectivity), no model can recover the assignment.
  Probe experiments (group/rank probes, learned aggregation) all confirmed
  the assignment bottleneck is an information wall, not an algorithmic one.
- **Neural networks do not beat the geometric baseline.** Equivariant
  message-passing models (including a hybrid attention architecture) score
  *worse* than the geometric `l1_local` predictor and stay capped by the
  same information ceiling. The neural line is released as a documented
  negative result.
- **Set-table modality is the commercially meaningful regime**: with ≥5
  observations per group, modal set orientations reach 7–12° error
  (engineering threshold ≤12°) — this is what the product chain delivers.
- **Multi-orientation dense point clouds: BLOCKED-BY-DATA.** A 2026-08
  survey (A/B/C tiers) found no public dataset satisfying
  real + multi-orientation + 3D point cloud + downloadable + ground-truth.
  The L3 multi-orientation claim therefore rests on a synthetic benchmark
  (K=4 spatially separated walls: MAE 0.37°, misclassification 0.4%,
  [`results/pointcloud_gate.json`](results/pointcloud_gate.json)) and is
  explicitly not generalized beyond wall-level separation.
- **Selection-criterion search closed at 12.554°**: an ensemble-based
  selection over 100 clustering restarts could not push the modal set error
  of the self-produced (vision-derived) picks below the ≤12° engineering
  gate — reported as a completed negative search, not a lost one.
- **Deprecated numbers are marked, not silently replaced.** Where a bug fix
  changed a published number (e.g. a sin/cos transpose in the FORGE
  pipeline), both old and new values are traceable in the audit trail, and
  result files carry `post-fix` provenance notes.

We believe publishing bounded claims with their failure modes is worth more
to practitioners than unbounded claims without evidence.

## Data policy

Per the compliance audit (2026-08-29), third-party data falls into three
rulings (full table in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)):

- **Included** — project-derived files from openly licensed sources (Utah
  FORGE, CC BY 4.0; attribution in `THIRD_PARTY_NOTICES.md`) and
  project-synthetic fixtures (`data/real/r60_wells_csv/`).
- **Link-only** — DECOVALEX, raw FORGE logs, Pontrelli, GeoCrack, INEL-1,
  EGS Collab, OpenTopography point clouds. Download from the official
  sources cited in the notices file.
- **Excluded** — the Beishan pre-selection area data (and mixed files
  containing it) is **not redistributable** and is excluded from this
  repository.

The repository is **data-self-sufficient**: the demo and the test suite run
entirely on synthetic or included data.

## Testing

```bash
python -m pytest tests -q
```

The suite covers geometry conventions (dip↔normal round-trips, Terzaghi
weights, sign alignment), the data-leak guards, DFN/percolation invariants,
report generation, the multi-well joint decision rule, and the release
guards themselves (`tests/test_v_r146_.py`). Some regression tests for
real-data pipelines are skipped with a recorded reason when the underlying
(link-only) data is absent — see `RELEASE_VERIFICATION.md` for the
per-test skip/exemption registry.

## Citation

See [`CITATION.cff`](CITATION.cff). If this repository is used in academic
work, please cite the software and the underlying datasets (notices file §3).

## License

MIT — see [`LICENSE`](LICENSE). Third-party components and data rulings:
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Acknowledgments

This work was carried out at the China University of Mining and Technology
(Beijing) under the Undergraduate Innovation and Entrepreneurship Training
Program (大学生创新创业训练计划), with guidance from the project supervisor
Prof. Liu Peng (刘鹏).

Author: Jiacheng Yi (易嘉诚), 2998812494@qq.com.
