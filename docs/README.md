# docs/ — GitHub Pages portal (project landing page)

This directory is the **GitHub Pages root** of the repository. Once Pages is
enabled (see `PUSH_GUIDE.md`, "GitHub Pages" section), the site is served at

```
https://<your-account>.github.io/fractureflow/
```

## Files

| Path | Purpose |
|---|---|
| `index.html` | Single-file landing page (hero, badges, screenshots, data ladder, honesty sections, interactive 3D DFN demo) |
| `plotly.min.js` | Local Plotly 2.35.2 bundle — **no CDN, no webfonts, no external requests** |
| `data/dfn_demo_discs.{js,json}` | Frozen synthetic DFN demo (13,644 generated → 2,000 displayed discs, fixed seed) |
| `data/recon_line.{js,json}` | Research-sprint screen data (baseline lineage, κ dose-response, equivariance gates) |
| `screenshots/*.png` | Product figures generated from synthetic data only: rose diagram, stereonet, 3D DFN scene |

## First-load size

The page is fully self-contained, which costs **~4.4 MB** on first load
(almost all of it `plotly.min.js`). Everything is served from the same origin
and cached by the browser afterwards. There is no network dependency at
runtime — the page works offline and inside air-gapped reviews.

## Local preview

```bash
cd docs
python -m http.server 8000
# open http://localhost:8000/
```

Opening `index.html` via `file://` also works, but the http server matches
Pages behavior more closely.

## Number discipline (data-src)

Every quantitative claim on the page carries a `data-src` attribute with its
provenance: an `NNS-###` entry of the project's frozen digital ledger plus a
JSON path (e.g. `NNS-011 · results/phase0_oracle_floor_strict.json`). Hover
any number to see it.

- The `results/*.json` paths that ship in this repository resolve locally.
- Some provenance strings reference upstream ledger paths (`docs/...`,
  `results/v_*...json`) from the internal project tree; those are
  documentation pointers, not links, and the ledger entries are the
  authority.
- Research-line numbers (the §7 sprint screen) are watermarked as such on
  the page and are excluded from the headline commitment line.

Regenerating `data/*.js` or the screenshots is a main-project operation
(scripts `r135_portal_data.py`, demo figure pipeline); do not hand-edit the
generated data files.
