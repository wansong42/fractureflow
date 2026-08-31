# Examples — demo outputs (static snapshots)

Everything in this folder was produced by the one-click demo
(`run_demo.sh` / `run_demo.cmd`) on a **synthetic** borehole log
(3 fracture sets, 60 fractures — no real site data). The files are
committed here so you can see the deliverables before installing
anything; regenerate them any time with the demo command.

| File | What it is |
|---|---|
| `borehole_report.md` | The full statistics report (Chinese, as delivered to engineers) |
| `group_table.csv` | The fracture-set table (machine-readable) |
| `rose_diagram.png` | Strike rose diagram |
| `stereonet.png` | Lower-hemisphere pole plot with set centroids |

The full chain behind these files:

```
synthetic borehole log
  -> Route-A auto-labeling (spherical k-means, |cos| assignment)
  -> set table CSV
  -> rose diagram + stereonet
  -> DFN realization + percolation screening
  -> Markdown report
```

## Your own data

Replace the synthetic log with your borehole record:

```bash
# CSV/Excel borehole log (depth / dip / dip-direction)
python scripts/auto_label_borehole.py --csv your_log.csv --out your_log_labeled.pt

# LAS borehole imaging file, end-to-end
python scripts/full_pipeline.py --input-las your_well.las --domain 50 50 50
```
