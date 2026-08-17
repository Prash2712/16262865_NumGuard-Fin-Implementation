from __future__ import annotations

import hashlib
import json
import re
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from numguard_fin.evaluation import summarise_records
from numguard_fin.reporting import _paired_comparisons

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"
ENGINEERING = VALIDATION / "engineering_checks"


def require(condition: bool, message: str, checks: list[dict]) -> None:
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_count() -> int:
    text = (VALIDATION / "test_run.txt").read_text(encoding="utf-8")
    matches = re.findall(r"(\d+) passed", text)
    if not matches:
        raise RuntimeError("Could not read the pytest count.")
    return int(matches[-1])


def parse_certificate(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def strict_json_files() -> list[str]:
    failures: list[str] = []

    def reject_constant(value: str):
        raise ValueError(f"non-standard constant {value}")

    for path in ROOT.rglob("*.json"):
        if "data/raw" in path.relative_to(ROOT).as_posix():
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
        except Exception as exc:  # noqa: BLE001 - audit must report every malformed file
            failures.append(f"{path.relative_to(ROOT).as_posix()}: {exc}")
    return failures


def xlsx_metadata_failures() -> list[str]:
    failures: list[str] = []
    for path in ROOT.rglob("*.xlsx"):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if any(name.startswith("docProps/") for name in names):
                failures.append(path.relative_to(ROOT).as_posix())
                continue
            if "_rels/.rels" in names:
                rels = archive.read("_rels/.rels").decode("utf-8", errors="ignore")
                if any(
                    token in rels
                    for token in (
                        "core-properties",
                        "extended-properties",
                        "custom-properties",
                    )
                ):
                    failures.append(path.relative_to(ROOT).as_posix())
    return failures


def png_metadata_failures() -> list[str]:
    failures: list[str] = []
    forbidden = {b"tEXt", b"zTXt", b"iTXt", b"tIME", b"eXIf", b"pHYs"}
    for path in ROOT.rglob("*.png"):
        raw = path.read_bytes()
        if raw[:8] != b"\x89PNG\r\n\x1a\n":
            failures.append(path.relative_to(ROOT).as_posix())
            continue
        offset = 8
        while offset + 12 <= len(raw):
            length = struct.unpack(">I", raw[offset : offset + 4])[0]
            chunk_type = raw[offset + 4 : offset + 8]
            if chunk_type in forbidden:
                failures.append(path.relative_to(ROOT).as_posix())
                break
            offset += 12 + length
            if chunk_type == b"IEND":
                break
    return failures


def path_trace_failures() -> list[str]:
    failures: list[str] = []
    patterns = (
        "/mnt/data/",
        "/home/oai/",
        "/Users/",
        "sediment://",
        "file_000000",
        "C:\\Users\\",
    )
    roots = [ROOT / "models", ROOT / "results", ROOT / "validation", ROOT / "reproducibility"]
    suffixes = {".json", ".csv", ".log", ".txt", ".md", ".sha256"}
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(pattern in text for pattern in patterns):
                failures.append(path.relative_to(ROOT).as_posix())
    return failures


def repository_hygiene_failures() -> list[str]:
    failures: list[str] = []
    forbidden_dirs = {"__pycache__", ".pytest_cache", ".run_state", ".ipynb_checkpoints"}
    forbidden_files = {".DS_Store"}
    forbidden_name_patterns = (
        re.compile(r"ProofLock-v[1-5]", re.IGNORECASE),
        re.compile(r"V[1-5]_CHANGELOG", re.IGNORECASE),
        re.compile(r"dev_diagnostic_v[1-5]", re.IGNORECASE),
    )
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT).as_posix()
        if path.is_dir() and path.name in forbidden_dirs:
            failures.append(rel)
        if path.is_file() and path.name in forbidden_files:
            failures.append(rel)
        if any(pattern.search(rel) for pattern in forbidden_name_patterns):
            failures.append(rel)
    return sorted(set(failures))


def notebook_is_clean() -> bool:
    notebook = json.loads((ROOT / "notebooks/NumGuard_Fin_Colab.ipynb").read_text(encoding="utf-8"))
    if notebook.get("metadata") not in ({}, None):
        return False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            if cell.get("execution_count") is not None or cell.get("outputs"):
                return False
    return True


def main() -> int:
    checks: list[dict] = []
    predictions = pd.read_csv(ENGINEERING / "predictions.csv")
    audit = pd.read_csv(VALIDATION / "dataset_audit.csv")
    counterfactual = pd.read_csv(VALIDATION / "counterfactual_audit.csv")

    require(test_count() >= 145, "At least 145 automated tests passed", checks)
    require(len(audit) == 12, "All 12 transparent validation cases were audited", checks)
    require(int(audit["candidate_recall"].sum()) == 11, "All 11 supported fixture answers were recalled", checks)

    unsupported_case = predictions[
        (predictions["method"] == "derivation_proof_lock")
        & predictions["example_id"].astype(str).str.endswith("report_12.pdf-1")
    ]
    require(
        len(unsupported_case) == 1 and bool(unsupported_case.iloc[0]["abstained"]),
        "The deliberately unsupported fixture produced an explicit abstention",
        checks,
    )

    hard_methods = {
        "direct_proof_lock",
        "derivation_proof_lock",
        "selector_guided_proof_lock",
        "risk_controlled_proof_lock",
    }
    hard = predictions[predictions["method"].isin(hard_methods)]
    require(not hard["unsupported_complete_answer"].astype(bool).any(), "Hard locks produced no constraint escapes", checks)
    require(
        (~hard["valid_numeric_response"].astype(bool) | hard["proof_valid"].astype(bool)).all(),
        "Every hard-lock numeric response carried a valid proof",
        checks,
    )
    committed = hard[hard["valid_numeric_response"].astype(bool)]
    certificates = committed["proof_certificate"].map(parse_certificate)
    require(all(bool(item.get("evidence_claims")) for item in certificates), "Committed hard-lock certificates contain structured evidence claims", checks)
    require(all(item.get("candidate_confidence") is not None for item in certificates), "Committed hard-lock certificates record candidate confidence", checks)

    required_prediction_fields = {
        "input_token_count",
        "input_truncated",
        "verification_seconds",
        "total_method_seconds",
        "reference_operation",
        "predicted_operation",
        "proof_operation_match",
        "selected_candidate_margin",
        "provenance_risk_score",
        "semantic_risk_score",
    }
    require(required_prediction_fields.issubset(predictions.columns), "Prompt, timing and operation-diagnostic fields are present", checks)
    require(predictions.groupby("method")["example_id"].nunique().nunique() == 1, "Every method was evaluated on the same fixtures", checks)
    require(predictions["method"].nunique() == 13 and len(predictions) == 156, "Thirteen methods produced 156 paired fixture records", checks)
    require(predictions.duplicated(["method", "example_id"]).sum() == 0, "No duplicate method/example pairs are present", checks)
    require(predictions["configuration_id"].nunique() == 1, "The validation run used one configuration", checks)
    require(counterfactual["passed"].astype(bool).all(), "All applicable candidate-mechanism counterfactual checks passed", checks)
    require(set(counterfactual["audit_layer"]) == {"candidate_mechanism"}, "Counterfactual output is labelled at the correct assurance layer", checks)
    require(counterfactual["perturbation"].nunique() == 5, "All five perturbation families were exercised", checks)
    require(predictions["candidate_count"].max() <= 241, "Total candidate bounds were respected", checks)
    require(predictions["direct_candidate_count"].max() <= 48, "Direct-candidate bounds were respected", checks)
    require(predictions["derived_candidate_count"].max() <= 192, "Derived-candidate bounds were respected", checks)

    forbidden_columns = {"gold_answer", "generated_answer", "synthetic_id", "feature_12"}
    require(not (forbidden_columns & set(predictions.columns)), "Legacy submission-facing columns are absent", checks)

    required_outputs = [
        ENGINEERING / "method_summary.csv",
        ENGINEERING / "paired_comparisons.csv",
        ENGINEERING / "research_tables.xlsx",
        ENGINEERING / "failure_review_blinded.xlsx",
        ENGINEERING / "failure_review_key.csv",
        ENGINEERING / "predictions.jsonl",
        VALIDATION / "engineering_check_figures/proof_operation_agreement.png",
        VALIDATION / "engineering_check_figures/method_latency.png",
        ROOT / "results/development/method_summary.csv",
        ROOT / "results/development/predictions.csv",
        ROOT / "results/development/semantic_calibration.json",
        ROOT / "models/candidate_selector.json",
        ROOT / "reproducibility/experiment_plan.json",
    ]
    require(all(path.exists() and path.stat().st_size > 0 for path in required_outputs), "Core implementation, result and assurance files are present", checks)

    first_jsonl = json.loads((ENGINEERING / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    require(isinstance(first_jsonl.get("proof_certificate"), dict), "JSONL stores proof certificates as nested objects", checks)

    stored = pd.read_csv(ROOT / "results/development/predictions.csv")
    require(len(stored) == 11323, "Stored development predictions contain 11,323 rows", checks)
    require(stored["method"].nunique() == 13 and stored["example_id"].nunique() == 871, "Stored development evidence covers 13 methods and 871 examples", checks)
    require(stored.duplicated(["method", "example_id"]).sum() == 0, "Stored development predictions contain no duplicate pairs", checks)
    require(stored.groupby("method")["example_id"].nunique().eq(871).all(), "Every stored method covers all 871 examples", checks)

    baseline = stored[stored["method"] == "baseline"]
    selector = stored[stored["method"] == "selector_guided_proof_lock"]
    fragment = stored[stored["method"] == "fragment_token_lock_ablation"]
    require(int(baseline["numeric_exact_match"].astype(bool).sum()) == 6, "Stored baseline correct-answer count is internally consistent", checks)
    require(int(baseline["unsupported_complete_answer"].astype(bool).sum()) == 46, "Stored baseline provenance-error count is internally consistent", checks)
    require(int(selector["numeric_exact_match"].astype(bool).sum()) == 14, "Stored selector-guided correct-answer count is internally consistent", checks)
    require(int(selector["unsupported_complete_answer"].astype(bool).sum()) == 0, "Stored selector-guided provenance-error count is internally consistent", checks)
    require(int(fragment["unsupported_complete_answer"].astype(bool).sum()) == 92, "Stored fragment-ablation escape count is internally consistent", checks)
    require(int(selector["candidate_recall"].astype(bool).sum()) == 424, "Stored candidate-recall count is internally consistent", checks)

    stored_summary = pd.read_csv(ROOT / "results/development/method_summary.csv").sort_values("method").reset_index(drop=True)
    regenerated_summary = summarise_records(stored.to_dict(orient="records"), seed=42).sort_values("method").reset_index(drop=True)
    summary_equal = stored_summary["method"].equals(regenerated_summary["method"])
    for column in stored_summary.columns:
        if column == "method":
            continue
        summary_equal = summary_equal and np.allclose(
            stored_summary[column].to_numpy(dtype=float),
            regenerated_summary[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
    require(summary_equal, "Stored method summary regenerates from the prediction records", checks)

    stored_comparisons = pd.read_csv(ROOT / "results/development/paired_comparisons.csv").sort_values(["method_a", "method_b", "outcome"]).reset_index(drop=True)
    regenerated_comparisons = _paired_comparisons(stored, 42).sort_values(["method_a", "method_b", "outcome"]).reset_index(drop=True)
    comparisons_equal = True
    for column in stored_comparisons.columns:
        if pd.api.types.is_numeric_dtype(stored_comparisons[column]):
            comparisons_equal = comparisons_equal and np.allclose(
                stored_comparisons[column].to_numpy(dtype=float),
                regenerated_comparisons[column].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )
        else:
            comparisons_equal = comparisons_equal and stored_comparisons[column].equals(
                regenerated_comparisons[column]
            )
    require(comparisons_equal, "Stored paired comparisons regenerate from the prediction records", checks)

    dev_manifest = json.loads((ROOT / "results/development/experiment_manifest.json").read_text(encoding="utf-8"))
    semantic = json.loads((ROOT / "results/development/semantic_calibration.json").read_text(encoding="utf-8"))
    provenance = json.loads((ROOT / "results/development/provenance_calibration.json").read_text(encoding="utf-8"))
    stored_predictions = ROOT / "results/development/predictions.csv"
    require(dev_manifest.get("selector_sha256") == sha256(ROOT / "models/candidate_selector.json"), "Stored selector hash matches the retained selector file", checks)
    require(semantic.get("source_predictions_sha256") == sha256(stored_predictions), "Semantic calibration hash matches stored predictions", checks)
    require(provenance.get("source_predictions_sha256") == sha256(stored_predictions), "Provenance calibration hash matches stored predictions", checks)
    require(semantic.get("feasible") is False and provenance.get("feasible") is True, "Stored semantic and provenance gate decisions are represented correctly", checks)

    require(not strict_json_files(), "All JSON files use strict JSON without NaN or Infinity", checks)
    require(not xlsx_metadata_failures(), "Excel files contain no document-property metadata", checks)
    require(not png_metadata_failures(), "PNG figures contain no ancillary metadata chunks", checks)
    require(not path_trace_failures(), "Stored artefacts contain no local filesystem traces", checks)
    require(not repository_hygiene_failures(), "Repository contains no caches, hidden system files or iteration-labelled artefacts", checks)
    require(notebook_is_clean(), "The Colab notebook contains no saved outputs, execution counts or author metadata", checks)
    require(not (ROOT / "configs").exists(), "Unused duplicate configuration files were removed", checks)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": "local technical, evidence-integrity and package-hygiene validation",
        "checks": checks,
        "summary": {
            "automated_tests": test_count(),
            "fixture_examples": 12,
            "paired_fixture_rows": len(predictions),
            "methods": predictions["method"].nunique(),
            "counterfactual_checks": len(counterfactual),
            "stored_development_rows": len(stored),
            "stored_development_examples": stored["example_id"].nunique(),
            "hard_lock_constraint_escapes": int(hard["unsupported_complete_answer"].astype(bool).sum()),
        },
    }
    VALIDATION.mkdir(parents=True, exist_ok=True)
    (VALIDATION / "validation_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("TECHNICAL AND PACKAGE VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
