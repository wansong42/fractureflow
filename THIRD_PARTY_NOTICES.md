# THIRD_PARTY_NOTICES

This product includes software developed by third parties. The following
components are used at runtime by the released code and are covered by their
respective licenses. Redistribution obligations: retain the copyright and
license notice of each component.

## 1. Runtime dependencies (BSD / MIT family)

| Component | Tested version | License | License text |
|---|---|---|---|
| NumPy | 2.2.5 | BSD-3-Clause | https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| SciPy | 1.15.3 | BSD-3-Clause | https://github.com/scipy/scipy/blob/main/LICENSE.txt |
| PyTorch | 2.6.0 | BSD-3-Clause | https://github.com/pytorch/pytorch/blob/main/LICENSE |
| matplotlib | 3.10.9 | Matplotlib License (BSD-3-compatible, PSF-based) | https://github.com/matplotlib/matplotlib/tree/main/LICENSE |
| scikit-learn | 1.7.2 | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn/blob/main/COPYING |
| pandas | 2.3.3 | BSD-3-Clause | https://github.com/pandas-dev/pandas/blob/main/LICENSE |

## 2. Audited components NOT required by the released code

The full-project dependency audit (2026-08-29) additionally covered:
torchvision (BSD-3), Pillow (MIT-CMU), Shapely (BSD-3), tifffile (BSD-3),
laspy (BSD-2), opencv-python (Apache-2.0) and dlisio (LGPL-3.0). None of
these is imported by any code file shipped in this repository; they appear
here for completeness of the audit trail.

**dlisio (Equinor, LGPL-3.0)** deserves an explicit standing statement:
dlisio is consumed by some *internal research* scripts that are deliberately
not part of this release. No dlisio source or object code is copied, modified,
or statically linked anywhere in this repository, and the released pipelines
do not require it. License text: https://www.gnu.org/licenses/lgpl-3.0.html

## 3. Data acknowledgements

Datasets used to produce the frozen result files in `results/` are **not
redistributed** with this repository, with the exception of the derived Utah
FORGE files listed in §4. Key citations:

- **Utah FORGE** — U.S. DOE Geothermal Data Repository (GDR, https://gdr.openei.org), licensed CC BY 4.0. E.g. well 16B(78)-32 FMI reinterpretation, DOI 10.15121/1452547.
- **DECOVALEX 2023-2027** 4-frac DFN benchmark — Zenodo, DOI 10.5281/zenodo.14873207 (CC BY 4.0). Linked only; not redistributed.
- **GeoCrack** — Yaqoob et al., *Scientific Data* 11:1318 (2024), DOI 10.1038/s41597-024-04107-0; dataset DOI 10.7910/DVN/E4OXHQ. Linked only.
- **Åland Islands** fracture dataset — Skyttä et al. 2019; data © the authors and the Geological Survey of Finland (KYT KARIKKO project). **Not redistributable.**
- **Pontrelli quarry** digital outcrop — Francioni et al. 2021, *Solid Earth*, DOI 10.5194/se-12-2055-2021. Linked only.
- **pySimFrac** — Los Alamos National Laboratory (2023), BSD-3-Clause. Cited as an external tool.
- **OpenTopography UT19 (Bunds)** point clouds — see the dataset page for terms. Linked only.

Sensitive-site data (the Beishan pre-selection area for high-level
radioactive waste disposal) is **excluded** from this repository and must not
be redistributed in any form.

## 4. Redistributed derived data

The following files under `data/` are *derived* from openly licensed Utah
FORGE data (U.S. DOE GDR, CC BY 4.0) by this project's pipelines, and are
redistributed here under CC BY 4.0 with attribution:

- `data/real/forge16A_net.pt`
- `data/real/forge2024_multi.pt`
- `data/real/forge2024meq_multi.pt`, `data/real/forge2024meq_single.pt`
- `data/external/utah_forge_fmi/*_routeA.pt` and `*_routeA_group_table.csv`
- `data/external/utah_forge_fmi/forge_fmi_2wells.pt` (+ summary/survey JSON)
