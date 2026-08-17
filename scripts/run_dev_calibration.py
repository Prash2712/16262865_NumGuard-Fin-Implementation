import json
from pathlib import Path

from numguard_fin.cli import main

selector = Path("models/candidate_selector.json")
if not selector.exists():
    raise SystemExit(
        "Missing models/candidate_selector.json. Run bash run_train_selector.sh first."
    )

candidate_dir = Path("results/rerun/candidate_diagnostic")
development_dir = Path("results/rerun/development")
figures_dir = Path("results/rerun/figures")

# Cheap CPU gate before full model inference. References are used only for this
# development diagnostic, never for candidate construction or decoding.
gate = [
    "inspect", "--data-dir", "data/raw", "--split", "dev",
    "--retrieval-top-k", "48", "--max-direct-candidates", "48",
    "--max-derived-candidates", "192",
    "--selector-file", str(selector),
    "--minimum-candidate-recall", "0.30",
    "--minimum-direct-recall", "0.75",
    "--enforce-gates",
    "--output", str(candidate_dir / "dataset_audit.csv"),
]
if main(gate):
    raise SystemExit(2)

run = [
    "run", "--data-dir", "data/raw", "--split", "dev",
    "--model-name", "google/flan-t5-base", "--device", "cuda",
    "--retrieval-top-k", "48", "--max-direct-candidates", "48",
    "--max-derived-candidates", "192", "--seed", "42",
    "--selector-file", str(selector),
    "--resume",
    "--output-dir", str(development_dir), "--figures-dir", str(figures_dir),
]
if main(run):
    raise SystemExit(1)

semantic_path = development_dir / "semantic_calibration.json"
semantic = [
    "calibrate", "--predictions", str(development_dir / "predictions.csv"),
    "--method", "selector_guided_proof_lock", "--risk-type", "semantic",
    "--target-risk", "0.20", "--confidence", "0.95",
    "--minimum-coverage", "0.05", "--minimum-accepted", "30",
    "--output", str(semantic_path),
]
if main(semantic):
    raise SystemExit(1)

provenance = [
    "calibrate", "--predictions", str(development_dir / "predictions.csv"),
    "--method", "selector_guided_proof_lock", "--risk-type", "provenance",
    "--target-risk", "0.01", "--confidence", "0.95",
    "--minimum-coverage", "0.05", "--minimum-accepted", "30",
    "--output", str(development_dir / "provenance_calibration.json"),
]
if main(provenance):
    raise SystemExit(1)

payload = json.loads(semantic_path.read_text(encoding="utf-8"))
if payload.get("feasible", False):
    print("SEMANTIC CALIBRATION GATE PASSED. Public-test execution is authorised.")
else:
    print(
        "SEMANTIC CALIBRATION GATE NOT MET. The development run completed, "
        "but public-test execution remains blocked."
    )
