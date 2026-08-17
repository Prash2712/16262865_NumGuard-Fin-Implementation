import pandas as pd

from numguard_fin.calibration import calibration_curve, clopper_pearson_upper, select_risk_threshold
from numguard_fin.counterfactual import (
    audit_counterfactual_predictions,
    audit_counterfactuals,
    generate_counterfactuals,
)
from numguard_fin.pipeline import PipelineConfig, run_example


def test_clopper_pearson_bounds():
    assert 0 < clopper_pearson_upper(0, 10) < 1
    assert clopper_pearson_upper(10, 10) == 1
    assert clopper_pearson_upper(0, 0) == 1


def test_calibration_selects_feasible_threshold():
    rows = [
        {"selected_candidate_confidence": 0.95, "valid_numeric_response": True, "numeric_exact_match": True},
        {"selected_candidate_confidence": 0.90, "valid_numeric_response": True, "numeric_exact_match": True},
        {"selected_candidate_confidence": 0.40, "valid_numeric_response": True, "numeric_exact_match": False},
        {"selected_candidate_confidence": None, "valid_numeric_response": False, "numeric_exact_match": False},
    ]
    decision = select_risk_threshold(rows, target_risk=0.80)
    assert decision.accepted_examples > 0
    assert 0 <= decision.coverage <= 1


def test_calibration_curve_is_monotonic_in_acceptance():
    rows = [
        {"selected_candidate_confidence": confidence, "valid_numeric_response": True, "numeric_exact_match": correct}
        for confidence, correct in [(0.9, True), (0.8, True), (0.4, False)]
    ]
    curve = calibration_curve(rows).sort_values("threshold")
    assert curve["accepted_examples"].is_monotonic_decreasing


def test_counterfactual_generation(example_by_suffix):
    cases = generate_counterfactuals(example_by_suffix("report_02.pdf-1"))
    names = {case.perturbation for case in cases}
    assert {"irrelevant_number_injection", "year_value_swap", "support_mask", "scale_flip"}.issubset(names)


def test_irrelevant_number_injection_is_invariant(example_by_suffix):
    rows = audit_counterfactuals(example_by_suffix("report_01.pdf-1"))
    injection = next(row for row in rows if row["perturbation"] == "irrelevant_number_injection")
    assert injection["passed"] is True


def test_counterfactual_audit_has_traceable_locations(example_by_suffix):
    rows = audit_counterfactuals(example_by_suffix("report_05.pdf-1"))
    assert rows
    assert all(row["changed_locations"] for row in rows)


def test_model_output_counterfactual_audit_uses_proof_certificates(example_by_suffix):
    config = PipelineConfig(
        dataset_split="dev",
        methods=("derivation_proof_lock",),
    )

    def predictor(example):
        return run_example(example, generator=None, config=config)[0]

    rows = audit_counterfactual_predictions(
        example_by_suffix("report_02.pdf-1"), predictor
    )
    assert rows
    assert {row["audit_layer"] for row in rows} == {"model_output"}
    assert all("original_proof" in row and "perturbed_proof" in row for row in rows)
