#!/usr/bin/env bash
# FractureFlow one-click demo (Linux/macOS).
set -e
cd "$(dirname "$0")"
export PYTHONUTF8=1
export KMP_DUPLICATE_LIB_OK=TRUE
python scripts/demo_run.py "$@"
