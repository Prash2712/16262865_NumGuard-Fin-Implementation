#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
python scripts/run_final_test.py
python scripts/run_counterfactual_audit.py
