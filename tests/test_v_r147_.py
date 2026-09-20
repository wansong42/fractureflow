# -*- coding: utf-8 -*-
"""R147 guards: GitHub Pages portal (docs/) structure, zero-dead-link policy,
badge row, and number discipline of the open-source landing page.

The portal must be truly offline (no external resource loads), work when
served from a project subpath (https://<account>.github.io/<repo>/), and
keep the ledger-anchored number discipline of the main project portal.
"""
from __future__ import annotations

import os
import re

RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(RELEASE_ROOT, "docs")

REQUIRED_DOCS_ASSETS = [
    "index.html",
    "plotly.min.js",
    os.path.join("data", "dfn_demo_discs.js"),
    os.path.join("data", "dfn_demo_discs.json"),
    os.path.join("data", "recon_line.js"),
    os.path.join("data", "recon_line.json"),
    os.path.join("screenshots", "rose_diagram.png"),
    os.path.join("screenshots", "stereonet.png"),
    os.path.join("screenshots", "dfn_3d.png"),
]

# Deprecated / leaked / oracle-existence numbers that must never appear on
# the public landing page (boundary-matched so substrings cannot pass).
FORBIDDEN_NUMBERS = [
    "12.39", "14.22", "15.82", "14.96", "16.08", "9.35", "7.78", "10.99",
    "10.62", "14.10", "11.01", "9.83", "7.50", "3.81", "20.81",
]

ANCHOR_HREF = re.compile(r'(?<![\w-])href="#([^"]+)"')
LOCAL_HREF = re.compile(r'(?<![\w-])href="([^"][^"]*)"')
RES_SRC = re.compile(r'(?<![\w-])src="([^"][^"]*)"')
ID_DEF = re.compile(r'id="([^"]+)"')


def _read(rel: str) -> str:
    with open(os.path.join(DOCS, rel), encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
def test_pages_assets_in_place():
    missing = [rel for rel in REQUIRED_DOCS_ASSETS
               if not os.path.exists(os.path.join(DOCS, rel))]
    assert not missing, f"missing docs/ assets: {missing}"
    # plotly bundle must be the real local copy (offline-first), not a stub
    assert os.path.getsize(os.path.join(DOCS, "plotly.min.js")) > 1_000_000, \
        "plotly.min.js looks truncated"
    for png in ("rose_diagram.png", "stereonet.png", "dfn_3d.png"):
        size = os.path.getsize(os.path.join(DOCS, "screenshots", png))
        assert size > 10_000, f"{png} looks like a placeholder ({size}B)"


def test_english_hero_and_badge_row():
    html = _read("index.html")
    low = html.lower()
    for token in ["borehole logs in", "fracture twins out", "honestly"]:
        assert token in low, f"English display headline lacks: {token}"
    assert "诚实优先" in html, "bilingual kicker missing"
    # hand-built badge row (no shields.io / no external badge images)
    for label in ["CI", "License", "DOI", "Python", "GitHub"]:
        assert f">{label}<" in html, f"badge row lacks: {label}"
    assert "shields.io" not in html, "external badge service must not be used"
    assert 'lang="en"' in html and 'lang="zh-CN"' in html, \
        "bilingual lang attrs missing"
    # synthetic-data-only screenshots statement
    assert "synthetic data only" in html


def test_zero_dead_links_and_relative_paths():
    html = _read("index.html")
    ids = set(ID_DEF.findall(html))

    # 1. in-page anchors resolve
    dangling_anchors = [a for a in ANCHOR_HREF.findall(html) if a not in ids]
    assert not dangling_anchors, f"dangling in-page anchors: {dangling_anchors}"

    # 2. local href targets exist on disk (docs/ or repo root via ../)
    externals = []
    for target in LOCAL_HREF.findall(html):
        if target.startswith("#"):
            continue
        if target.startswith(("http://", "https://", "//")):
            externals.append(target)
            continue
        assert not target.startswith("/"), \
            f"root-relative href breaks project-subpath serving: {target}"
        resolved = os.path.normpath(os.path.join(DOCS, target))
        assert os.path.exists(resolved), f"dead link: {target}"

    # 3. resource loads: local-relative only, no external requests
    for src in RES_SRC.findall(html):
        if "'" in src or "+" in src:   # JS string concatenation, not an attr
            continue
        assert not src.startswith(("http://", "https://", "//", "/", "data:")), \
            f"non-relative resource load (breaks offline + subpath): {src}"
        resolved = os.path.normpath(os.path.join(DOCS, src))
        assert os.path.exists(resolved), f"dead resource: {src}"

    # 4. the only external navigations point at the repo itself on GitHub
    for target in externals:
        assert target.startswith("https://github.com/") \
            and "/fractureflow" in target, \
            f"undocumented external link: {target}"


TABLE_ROWS = ("| T", "| X")


def _table_rows(text):
    return [ln for ln in text.splitlines() if ln.startswith(TABLE_ROWS)]


def test_number_discipline_on_landing_page():
    """R279 / 裁定 83 形态：主张数字须能在随页附表面件里反查（不再靠读者点不开的内联属性）。"""
    html = _read("index.html")
    table = _read("number_pointers.md")
    assert "data-src" not in html, "inline provenance attributes should have been stripped"
    for anchor in ["36.69", "9.82", "0.37", "0.0054"]:
        lines = [ln for ln in html.splitlines() if anchor in ln]
        assert lines, f"landing page lacks anchor number: {anchor}"
        assert any(anchor in ln for ln in _table_rows(table)), \
            f"anchor number not traceable in the shipped pointer table: {anchor}"
    n_num = html.count('class="num"')
    rows = _table_rows(table)
    assert len(rows) >= 50, "pointer table coverage collapsed"
    assert len(rows) >= n_num, \
        f"pointer rows {len(rows)} < published .num readings {n_num}: coverage insufficient"
    # research-line watermark on the sprint screen
    assert "研究线" in html, "S7 research-line watermark missing"
    # deprecated / leaked numbers must not appear (boundary matched)
    hits = []
    for num in FORBIDDEN_NUMBERS:
        pat = re.compile(rf"(?<![\d.]){re.escape(num)}(?![\d])")
        if pat.search(html):
            hits.append(num)
    assert not hits, f"deprecated numbers on public page: {hits}"


def test_pointer_table_discipline_guard_has_teeth():
    """判别式负例：附表面件的读数查找必须能区分在场与不在场（防空跑/恒真判绿）。"""
    rows = _table_rows(_read("number_pointers.md"))
    assert len(rows) >= 50, "附表行数塌陷：判据将在退化底面上判绿"
    for anchor in ["36.69", "9.82", "0.37", "0.0054"]:
        assert any(anchor in ln for ln in rows), "在场读数查不到 = 判据错红: " + anchor
    for absent in ["12.3456", "99.999", "7.7777"]:
        assert not any(absent in ln for ln in rows), "不存在读数被判为可反查 = 判据无牙: " + absent
    carriers = [ln for ln in rows if "0.0054" in ln]
    assert carriers, "底面里就没有承载行，负例无效"
    pruned = [r for r in rows if r not in carriers]
    assert len(pruned) < len(rows), "负例注入后与底本相同 = 该负例无效"
    assert not any("0.0054" in ln for ln in pruned), "删承载行后仍判为可反查 = 判据无牙"


def test_docs_readme_documents_first_load_and_provenance():
    readme_path = os.path.join(DOCS, "README.md")
    assert os.path.exists(readme_path), "docs/README.md missing"
    with open(readme_path, encoding="utf-8") as fh:
        readme = fh.read()
    assert "plotly" in readme.lower(), "docs/README must explain the plotly bundle"
    assert "4.4" in readme, "docs/README must state the ~4.4MB first-load size"
    assert "data-src" in readme, "docs/README must explain the data-src provenance"
