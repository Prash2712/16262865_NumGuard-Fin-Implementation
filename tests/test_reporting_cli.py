import json

import pandas as pd

from numguard_fin.cli import main
from numguard_fin.pipeline import PipelineConfig, run_examples
from numguard_fin.reporting import build_failure_review_sample, generate_research_outputs


def test_research_outputs_are_created(tmp_path, examples):
    records = run_examples(examples[:2], generator=None, config=PipelineConfig(dataset_split="dev"))
    paths = generate_research_outputs(records, output_dir=tmp_path / "results", figures_dir=tmp_path / "figures")
    assert all(path.exists() for path in paths.values())
    assert (tmp_path / "figures" / "unsupported_answer_rate.png").exists()
    assert (tmp_path / "figures" / "numeric_answer_accuracy.png").exists()


def test_failure_review_requires_manual_confirmation(examples):
    frame = pd.DataFrame([record.to_dict() for record in run_examples(examples[:2], generator=None, config=PipelineConfig(dataset_split="dev"))])
    review = build_failure_review_sample(frame)
    assert {"manual_category", "manual_notes", "second_review_status"}.issubset(review.columns)
    assert (review["manual_category"] == "").all()


def test_cli_engineering_run_and_validation(tmp_path, fixture_path):
    result_dir = tmp_path / "run"
    figures_dir = tmp_path / "figures"
    code = main([
        "run", "--path", str(fixture_path), "--split", "dev", "--limit", "2",
        "--engineering-check", "--output-dir", str(result_dir), "--figures-dir", str(figures_dir),
    ])
    assert code == 0
    manifest = json.loads((result_dir / "experiment_manifest.json").read_text())
    assert manifest["run_type"] == "engineering_validation"
    assert main([
        "validate", "--predictions", str(result_dir / "predictions.csv"),
        "--allow-engineering",
    ]) == 0

    # A matching restart must reuse the complete per-example checkpoint rather than
    # duplicate method/example pairs.
    assert main([
        "run", "--path", str(fixture_path), "--split", "dev", "--limit", "2",
        "--engineering-check", "--resume",
        "--output-dir", str(result_dir), "--figures-dir", str(figures_dir),
    ]) == 0
    resumed = pd.read_csv(result_dir / "predictions.csv")
    assert not resumed.duplicated(["method", "example_id"]).any()


def test_cli_rejects_legacy_column(tmp_path):
    frame = pd.DataFrame({
        "example_id": ["x"], "method": ["baseline"], "question": ["q"],
        "reference_answer": ["1"], "model_response": ["1"],
        "valid_numeric_response": [True], "abstained": [False], "invalid_response": [False],
        "numeric_exact_match": [True], "candidate_recall": [True], "proof_valid": [True],
        "unsupported_complete_answer": [False], "proof_certificate": ["{}"],
        "candidate_count": [1], "configuration_id": ["abc"], "gold_answer": ["1"],
        "model_name": ["model"], "verification_seconds": [0.0],
        "total_method_seconds": [0.0],
    })
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    assert main(["validate", "--predictions", str(path)]) == 1


def test_cli_inspect_emits_v3_failure_stage_diagnostics(tmp_path, fixture_path):
    output = tmp_path / "audit.csv"
    assert main([
        "inspect", "--path", str(fixture_path), "--split", "dev",
        "--output", str(output),
    ]) == 0
    frame = pd.read_csv(output)
    required = {
        "alternative_operations",
        "intent_reference_operation_match",
        "reference_operand_count",
        "reference_operands_in_ledger_rate",
        "reference_operands_retrieved_rate",
        "candidate_recall_preselector",
        "candidate_recall_unpruned_same_retrieval",
        "expanded_ledger_candidate_recall_diagnostic",
        "failure_stage",
    }
    assert required.issubset(frame.columns)
    gate = json.loads((tmp_path / "candidate_quality_gate.json").read_text())
    assert {
        "candidate_recall",
        "candidate_recall_preselector",
        "candidate_recall_unpruned_same_retrieval",
        "expanded_ledger_candidate_recall_diagnostic",
        "failure_stage_counts",
        "reference_usage",
    }.issubset(gate)


def test_reference_program_diagnostics_do_not_count_constants_or_intermediates():
    from numguard_fin.cli import _reference_operands, _reference_operations

    program = "subtract(193.5, const_100), divide(#0, 25)"
    assert _reference_operations(program) == ("subtract", "divide")
    assert tuple(str(value) for value in _reference_operands(program)) == ("193.5", "25")
