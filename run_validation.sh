#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p validation
python scripts/verify_source_manifest.py
pytest -q -p no:cacheprovider | tee validation/test_run.txt
python scripts/run_engineering_verification.py
find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.run_state' \) -prune -exec rm -rf {} +
python scripts/quality_gate.py
