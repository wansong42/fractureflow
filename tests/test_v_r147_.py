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


def test_number_discipline_on_landing_page():
    html = _read("index.html")
    # anchor numbers present and provenance-carrying (data-src on the line)
    for anchor in ["36.69", "9.82", "0.37", "0.0054"]:
        lines = [ln for ln in html.splitlines() if anchor in ln]
        assert lines, f"landing page lacks anchor number: {anchor}"
        assert any("data-src" in ln for ln in lines), \
            f"anchor number without data-src provenance: {anchor}"
    assert html.count("data-src=") >= 50, "provenance coverage collapsed"
    # research-line watermark on the sprint screen
    assert "研究线" in html, "S7 research-line watermark missing"
    # deprecated / leaked numbers must not appear (boundary matched)
    hits = []
    for num in FORBIDDEN_NUMBERS:
        pat = re.compile(rf"(?<![\d.]){re.escape(num)}(?![\d])")
        if pat.search(html):
            hits.append(num)
    assert not hits, f"deprecated numbers on public page: {hits}"


def test_docs_readme_documents_first_load_and_provenance():
    readme_path = os.path.join(DOCS, "README.md")
    assert os.path.exists(readme_path), "docs/README.md missing"
    with open(readme_path, encoding="utf-8") as fh:
        readme = fh.read()
    assert "plotly" in readme.lower(), "docs/README must explain the plotly bundle"
    assert "4.4" in readme, "docs/README must state the ~4.4MB first-load size"
    assert "data-src" in readme, "docs/README must explain the data-src provenance"
