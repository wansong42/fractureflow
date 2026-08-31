# -*- coding: utf-8 -*-
"""R146 release guards: structural, policy and reproducibility invariants of
the push-ready open-source repository.

These guards run INSIDE the release repository (not the main project tree)
and are part of the shipped test suite, so every future release candidate
can re-certify itself.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_TOP = [
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CITATION.cff",
    "CHANGELOG.md", "SECURITY.md", "PUSH_GUIDE.md", "RELEASE_VERIFICATION.md",
    "requirements.txt", "environment.yml", ".gitignore",
    "run_demo.sh", "run_demo.cmd",
    "src", "scripts", "tests", "results", "data",
    ".github/workflows/ci.yml",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/feature_request.md",
    "PULL_REQUEST_TEMPLATE.md",
]

REQUIRED_SCRIPTS = [
    "full_pipeline.py", "auto_label_borehole.py", "dfn_from_borehole.py",
    "demo_run.py", "read_forge_las.py", "borehole_report.py",
    "borehole_excel_entry.py", "check_geometry_conventions.py",
    "release_sensitivity_scan.py",
]

REQUIRED_RESULTS = [
    "results/pointcloud_gate.json", "results/decovalex_routeB.json",
    "results/r110_b1/b1_scorecard.json",
    "results/honest_leaderboard/l1_local__beishan_22.json",
    "results/global_honest_leaderboard/beishan.json",
]

# Files that must NEVER exist in the release (R145 hard rulings).
FORBIDDEN_DATA = [
    "data/real/beishan_wells.npz",
    "data/real/loaded_real_nets.pt",
    "data/real/loaded_real_nets_setid.pt",
    "data/real/loaded_real_nets_setid_opt.pt",
    "data/real/aland_20m.pt",
    "data/real/pontrelli_multi.pt",
    "data/real/pontrelli_single.pt",
    "data/real/forge2024_single.pt",
]


def _p(*parts: str) -> str:
    return os.path.join(RELEASE_ROOT, *parts)


# ---------------------------------------------------------------------------
def test_release_structure_complete():
    missing = [rel for rel in REQUIRED_TOP if not os.path.exists(_p(rel))]
    assert not missing, f"missing release artifacts: {missing}"
    missing_scripts = [s for s in REQUIRED_SCRIPTS
                       if not os.path.exists(_p("scripts", s))]
    assert not missing_scripts, f"missing scripts: {missing_scripts}"
    assert os.path.isdir(_p("src", "fractureflow")), "core package missing"
    missing_results = [rel for rel in REQUIRED_RESULTS
                       if not os.path.exists(_p(rel))]
    assert not missing_results, f"missing frozen results: {missing_results}"


def test_forbidden_data_absent():
    present = [rel for rel in FORBIDDEN_DATA if os.path.exists(_p(rel))]
    assert not present, f"forbidden data files present: {present}"


def test_gitignore_carries_exclusion_policy():
    with open(_p(".gitignore"), encoding="utf-8") as fh:
        gi = fh.read()
    for marker in ["beishan_wells.npz", "loaded_real_nets", "aland_20m.pt",
                   "AGENTS.md", "tasks/", "试点材料", "fmi_attr/", "vision/",
                   "models/", "efracture_data", "EFRACTURE_DATA.md",
                   "_release_checks"]:
        assert marker in gi, f".gitignore lacks policy marker: {marker}"


def test_license_notices_citation_present():
    with open(_p("LICENSE"), encoding="utf-8") as fh:
        lic = fh.read()
    assert "MIT License" in lic, "LICENSE must be the MIT license"
    assert "Copyright (c)" in lic, "LICENSE must carry a copyright line"
    with open(_p("THIRD_PARTY_NOTICES.md"), encoding="utf-8") as fh:
        notice = fh.read()
    for marker in ["Utah FORGE", "DECOVALEX", "Beishan", "CC BY 4.0",
                   "LGPL"]:
        assert marker in notice, f"THIRD_PARTY_NOTICES lacks: {marker}"
    with open(_p("CITATION.cff"), encoding="utf-8") as fh:
        cff = fh.read()
    for key in ["cff-version:", "title:", "version:", "authors:", "license:"]:
        assert key in cff, f"CITATION.cff lacks required key: {key}"


def test_release_documentation_honesty_section():
    with open(_p("README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    assert "Honesty & negative results" in readme, \
        "README must carry the honesty / negative-results section"
    for anchor in ["36.69", "0.0054", "0.37", "BLOCKED-BY-DATA"]:
        assert anchor in readme, f"README lacks honest anchor: {anchor}"


def test_no_absolute_or_personal_paths():
    """T2 regression guard: no machine-specific paths may enter the repo.

    Gitignored runtime-output directories (results/demo, data/demo,
    _release_checks) are exempt: they are regenerated locally and never
    pushed.
    """
    bad = re.compile(r"\b[A-Za-z]:[/\\]|app2|晚菘|Anacoda")
    offenders = []
    self_rel = os.path.relpath(os.path.abspath(__file__),
                               RELEASE_ROOT).replace("\\", "/")
    skip_dirs = {".git", "__pycache__", "_release_checks", ".pytest_cache",
                 "node_modules"}
    for base, dirs, files in os.walk(RELEASE_ROOT):
        rel_base = os.path.relpath(base, RELEASE_ROOT).replace("\\", "/")
        dirs[:] = [d for d in dirs
                   if d not in skip_dirs
                   and f"{rel_base}/{d}".strip("./") not in
                   {"results/demo", "data/demo"}]
        for fn in files:
            if os.path.splitext(fn)[1] not in {".py", ".md", ".json", ".yml",
                                               ".cff", ".txt", ".csv", ".cmd",
                                               ".sh", ".cff"}:
                continue
            path = os.path.join(base, fn)
            rel = os.path.relpath(path, RELEASE_ROOT)
            rel = rel.replace("\\", "/")
            if rel == self_rel or rel == "scripts/release_sensitivity_scan.py":
                continue  # self-audit files document the token list by design
            with open(path, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if bad.search(line):
                        offenders.append(f"{rel}:{i}: {line.strip()[:100]}")
    assert not offenders, "machine-specific paths found:\n" + "\n".join(offenders[:10])


def test_sensitivity_scan_self_rerun_clean():
    """T3.2 reproducibility: the scan must pass inside the shipped repo."""
    proc = subprocess.run(
        [sys.executable, _p("scripts", "release_sensitivity_scan.py")],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"unexempted sensitivity hits:\n{proc.stdout[-2000:]}"
    )
    with open(_p("_release_checks", "scan_report.json"), encoding="utf-8") as fh:
        report = json.load(fh)
    assert report["unexempted"] == 0
    assert report["total_hits"] >= 0


def test_git_tag_and_first_commit_in_place():
    git_dir = _p(".git")
    if not os.path.isdir(git_dir):
        pytest.skip("not a git checkout (e.g. CI export/tarball)")
    def git(*args):
        return subprocess.run(["git", *args], cwd=RELEASE_ROOT,
                              capture_output=True, text=True, timeout=60)
    log = git("log", "--reverse", "--format=%s", "HEAD")
    assert log.returncode == 0, f"git log failed: {log.stderr}"
    commits = [c for c in log.stdout.strip().splitlines() if c.strip()]
    assert commits, "repository has no commits"
    shallow = _p(".git/shallow")
    if os.path.isfile(shallow):
        pytest.skip("shallow clone (CI fetch-depth=1); first-commit check requires full history")
    assert "v0.1.0" in commits[0] or "release" in commits[0].lower(), \
        f"first commit is not the release commit: {commits[0]!r}"
    tags = git("tag", "--list")
    assert tags.returncode == 0
    tag_names = [t for t in tags.stdout.split() if t]
    if not tag_names:
        pytest.skip("no tags present (shallow/tagless checkout, e.g. CI)")
    assert "v0.1.0" in tag_names, f"expected tag v0.1.0, found {tag_names}"
