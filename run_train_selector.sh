#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
mkdir -p models results
python scripts/fit_candidate_selector.py
