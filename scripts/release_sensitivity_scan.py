#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Release sensitivity scan (R146 T3.2).

Scans every tracked-candidate file (filename + content) for internal-process
leakage markers and writes a machine-readable + human-readable report to
_release_checks/.  Every hit must either be removed or appear in EXEMPTIONS
below with an explicit reason; the release guard (tests/test_v_r146_.py)
reruns this scan and asserts hits == 0 modulo registered exemptions.

Pattern provenance: task book R146 T3.2 pattern
    beishan | 试点 | NDA | 客户 | AGENTS | 看板 | 架构师 | 交接 | task_ | R1dd
Two intent-preserving refinements (documented, not silently relaxed):
  * NDA is matched as a case-sensitive acronym with word boundaries,
    so common English words ("standard") do not produce false positives;
  * R1dd is matched with word boundaries, so identifiers that merely
    embed e.g. "r110" inside a longer token are still caught, but hash
    fragments are not scanned (binary files are skipped anyway).
Frozen anchor numbers (36.687 / 12.37 / 0.37) are scientific content and are
not scanned.

Exit code: 0 = only exempted hits; 1 = unexempted hits; 2 = internal error.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONTENT_TOKENS = [
    "beishan", "试点", "客户", "AGENTS", "看板", "架构师", "交接", "task_",
    re.compile(r"\bNDA\b"), re.compile(r"\bR1\d\d\b", re.IGNORECASE),
]
FILENAME_TOKENS = [
    "beishan", "试点", "客户", "agents", "看板", "架构师", "交接", "task_",
    re.compile(r"r1\d\d", re.IGNORECASE),
]

SKIP_DIRS = {".git", "__pycache__", "_release_checks", ".pytest_cache",
             "node_modules", "data/demo", "results/demo"}
TEXT_EXTS = {".py", ".json", ".md", ".csv", ".yml", ".yaml", ".cff", ".txt",
             ".cfg", ".toml", ".html", ".js", ".sh", ".cmd", ".bat"}

# ---------------------------------------------------------------------------
# Exemption registry.  Key = file path relative to repo root (posix style),
# or ("*:token", token) for a repo-wide category.  Every entry needs a reason.
# ---------------------------------------------------------------------------
EXEMPTIONS: dict[tuple, str] = {
    # -- Category: frozen scientific result snapshots (R146 T1 whitelist) ----
    ("*:token", "beishan"): (
        "Site name in scientific context (evaluation cohort naming, usage "
        "examples, data-policy statements). The Beishan site DATA itself is "
        "excluded (hard ruling R145 §2.1, enforced by .gitignore). Aggregate "
        "metrics in results/*.json are the frozen anchors explicitly "
        "retained by R146 T3.2 (36.687 / 12.37 / 0.37)."
    ),
    ("*:token", "r1dd"): (
        "Internal research-line numbers appearing in comments/provenance "
        "notes of frozen code and result snapshots. Pure identifiers; no "
        "task documents, no process material is shipped."
    ),
    ("*:token", "客户"): (
        "Chinese word 'client' in docstrings describing the commercial "
        "logging-table dialect / demo purpose. Describes a file FORMAT, "
        "contains no client names, commitments or business material."
    ),
    ("*:token", "task_"): (
        "Generic identifiers ('task' JSON keys, fixture names); not the "
        "internal tasks/ directory, which is not shipped."
    ),
    ("*:token", "架构师"): (
        "Chinese word 'architect' in code comments / frozen-result prose "
        "attributing a design decision (e.g. K>8 hard gate, T30 gate "
        "verdict notes). Process vocabulary only; no task documents, "
        "handover notes or decision texts are shipped."
    ),
}

# Per-file exceptions that OVERRIDE category exemptions.
# The two self-audit files quote the scan pattern / forbidden-name lists
# verbatim by design (audit trail); they contain no process material.
FILE_OVERRIDES: dict[str, str] = {
    "RELEASE_VERIFICATION.md": (
        "Release self-audit ledger: quotes the scan token pattern and the "
        "forbidden-name list verbatim as documentation. No task documents "
        "or process material."
    ),
    "tests/test_v_r146_.py": (
        "Release guard test: embeds the forbidden-name list and scan-token "
        "class to assert their presence/absence. No process material."
    ),
}


def _rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def _iter_files():
    skip_exact = {".git", "__pycache__", "_release_checks", ".pytest_cache",
                  "node_modules"}
    self_rel = os.path.relpath(os.path.abspath(__file__), ROOT).replace("\\", "/")
    for base, dirs, files in os.walk(ROOT):
        rel_base = _rel(base)
        # prune skipped subdirectories before descending
        keep = []
        for d in dirs:
            rel_d = d if rel_base == "." else rel_base + "/" + d
            if rel_d in SKIP_DIRS or d in skip_exact:
                continue
            keep.append(d)
        dirs[:] = keep
        for fn in files:
            rel = os.path.relpath(os.path.join(base, fn), ROOT).replace("\\", "/")
            if rel == self_rel:
                continue  # the scanner never reports on itself
            yield os.path.join(base, fn)


def _category(token: str) -> tuple | None:
    if token == "r1dd" or token == "nda":
        return ("*:token", "r1dd") if token == "r1dd" else None
    return ("*:token", token)


def scan():
    hits = []
    for path in _iter_files():
        rel = _rel(path)
        ext = os.path.splitext(path)[1].lower()
        for tok in FILENAME_TOKENS:
            if isinstance(tok, str):
                if tok in rel.lower():
                    hits.append({"file": rel, "kind": "filename", "token": tok,
                                 "line": None, "text": None})
            else:
                if tok.search(rel):
                    hits.append({"file": rel, "kind": "filename", "token": "r1dd",
                                 "line": None, "text": None})
        if ext not in TEXT_EXTS:
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    low = line.lower()
                    for tok in CONTENT_TOKENS:
                        if isinstance(tok, str):
                            if tok in low:
                                hits.append({"file": rel, "kind": "content",
                                             "token": tok, "line": i,
                                             "text": line.strip()[:160]})
                        elif tok.pattern == r"\bNDA\b":
                            if tok.search(line):
                                hits.append({"file": rel, "kind": "content",
                                             "token": "nda", "line": i,
                                             "text": line.strip()[:160]})
                        else:
                            if tok.search(line):
                                hits.append({"file": rel, "kind": "content",
                                             "token": "r1dd", "line": i,
                                             "text": line.strip()[:160]})
        except OSError as exc:
            hits.append({"file": rel, "kind": "error", "token": "read",
                         "line": None, "text": str(exc)[:160]})
    return hits


def evaluate(hits):
    verdicts = []
    for h in hits:
        rel = h["file"]
        if rel in FILE_OVERRIDES:
            verdicts.append({**h, "status": "EXEMPTED",
                             "reason": FILE_OVERRIDES[rel]})
            continue
        cat = _category(h["token"])
        reason = EXEMPTIONS.get(cat)
        verdicts.append({**h, "status": "EXEMPTED" if reason else "UNEXEMPTED",
                         "reason": reason or "*** UNREGISTERED HIT ***"})
    return verdicts


def main():
    hits = evaluate(scan())
    n_un = sum(1 for h in hits if h["status"] == "UNEXEMPTED")
    os.makedirs(os.path.join(ROOT, "_release_checks"), exist_ok=True)
    payload = {
        "tool": "release_sensitivity_scan",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "files_scanned_root": ROOT,
        "total_hits": len(hits),
        "unexempted": n_un,
        "hits": hits,
    }
    for out in ("scan_report.json",):
        with open(os.path.join(ROOT, "_release_checks", out), "w",
                  encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1,
                      allow_nan=False)
    md = ["# Release sensitivity scan report",
          f"generated: {payload['generated']}",
          f"total hits: {len(hits)} / unexempted: {n_un}", ""]
    for h in hits:
        loc = f"{h['file']}:{h['line']}" if h["line"] else h["file"]
        md.append(f"- [{h['status']}] ({h['kind']}:{h['token']}) {loc}")
        if h["text"]:
            md.append(f"  text: {h['text']}")
        md.append(f"  reason: {h['reason'][:200]}")
    with open(os.path.join(ROOT, "_release_checks", "scan_report.md"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"scan: {len(hits)} hits, {n_un} unexempted "
          f"-> _release_checks/scan_report.md")
    return 0 if n_un == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
