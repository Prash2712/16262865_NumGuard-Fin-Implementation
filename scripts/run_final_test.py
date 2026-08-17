import hashlib
import json
from pathlib import Path

from numguard_fin.cli import main


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


selector = Path("models/candidate_selector.json")
development = Path("results/rerun/development")
calibration = development / "semantic_calibration.json"
dev_predictions = development / "predictions.csv"
dev_manifest = development / "experiment_manifest.json"
required = [selector, calibration, dev_predictions, dev_manifest]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(
        "Missing rerun development artefacts; public test is blocked: " + ", ".join(missing)
    )

calibration_payload = json.loads(calibration.read_text(encoding="utf-8"))
if calibration_payload.get("feasible") is not True:
    raise SystemExit("Development semantic calibration is not feasible; public test is blocked.")
if calibration_payload.get("risk_type") != "semantic":
    raise SystemExit("The public-test threshold must come from semantic-risk calibration.")
if float(calibration_payload.get("coverage", 0.0)) < 0.05:
    raise SystemExit("Development calibration coverage is below the fixed 5% gate.")
if int(calibration_payload.get("accepted_examples", 0)) < 30:
    raise SystemExit("Development calibration accepted fewer than 30 examples.")
if calibration_payload.get("source_predictions_sha256") != sha256(dev_predictions):
    raise SystemExit("Development predictions changed after calibration; public test is blocked.")

dev_payload = json.loads(dev_manifest.read_text(encoding="utf-8"))
if dev_payload.get("selector_sha256") != sha256(selector):
    raise SystemExit("The candidate selector differs from the development run; public test is blocked.")
source_configs = calibration_payload.get("source_configuration_ids") or []
if source_configs and dev_payload.get("configuration_id") not in source_configs:
    raise SystemExit("Calibration and development manifest configuration IDs do not match.")

output_dir = Path("results/rerun/public_test")
command = [
    "run", "--data-dir", "data/raw", "--split", "test",
    "--model-name", "google/flan-t5-base", "--device", "cuda",
    "--retrieval-top-k", "48", "--max-direct-candidates", "48",
    "--max-derived-candidates", "192", "--seed", "42",
    "--selector-file", str(selector),
    "--resume",
    "--risk-threshold-file", str(calibration),
    "--output-dir", str(output_dir), "--figures-dir", "results/rerun/public_test_figures",
]
if main(command):
    raise SystemExit(1)
raise SystemExit(main([
    "validate", "--predictions", str(output_dir / "predictions.csv"),
    "--maximum-candidates", "240",
]))
