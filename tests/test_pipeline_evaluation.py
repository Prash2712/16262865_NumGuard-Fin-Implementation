from numguard_fin.evaluation import mcnemar_exact, paired_bootstrap_difference, summarise_records
from numguard_fin.pipeline import (
    DEFAULT_METHODS,
    PipelineConfig,
    infer_reference_operation,
    run_examples,
)


def test_engineering_pipeline_produces_paired_rows(examples):
    config = PipelineConfig(dataset_split="dev")
    records = run_examples(examples[:3], generator=None, config=config)
    assert len(records) == 3 * len(DEFAULT_METHODS)
    assert {(record.example_id, record.method) for record in records} == {
        (example.example_id, method) for example in examples[:3] for method in DEFAULT_METHODS
    }


def test_hard_locks_have_no_constraint_escape(examples):
    records = run_examples(examples, generator=None, config=PipelineConfig(dataset_split="dev"))
    hard = [record for record in records if record.method in {"direct_proof_lock", "derivation_proof_lock", "selector_guided_proof_lock", "risk_controlled_proof_lock"}]
    assert hard
    assert all(not record.unsupported_complete_answer for record in hard)
    assert all(record.proof_valid or record.abstained for record in hard)


def test_pipeline_uses_clean_submission_fields(examples):
    row = run_examples(examples[:1], generator=None, config=PipelineConfig(dataset_split="dev"))[0].to_dict()
    assert "reference_answer" in row and "model_response" in row
    assert "gold_answer" not in row and "generated_answer" not in row


def test_operation_diagnostic_is_not_claimed_as_exact_program_accuracy(examples):
    records = run_examples(examples[:1], generator=None, config=PipelineConfig(dataset_split="dev"))
    assert all(not hasattr(record, "program_execution_match") for record in records)
    assert all(record.reference_operation is not None for record in records)


def test_reference_operation_parser_handles_financial_program_families():
    assert infer_reference_operation(
        "divide(subtract(560,498),498)",
        question_operation="percentage_change",
    ) == "percentage_change"
    assert infer_reference_operation(
        "divide(1100,4200)", question_operation="margin"
    ) == "margin"
    assert infer_reference_operation("average(80,100)") == "average"
    assert infer_reference_operation("table_lookup(metric,2023)") == "direct_lookup"


def test_summary_does_not_reward_no_number_as_proof():
    rows = [
        {
            "method": "x", "example_id": "1", "numeric_exact_match": False,
            "valid_numeric_response": False, "abstained": True, "invalid_response": False,
            "unsupported_complete_answer": False, "proof_valid": False,
            "candidate_recall": False, "selected_candidate_confidence": None,
            "candidate_count": 1, "direct_candidate_count": 0,
            "derived_candidate_count": 0, "retrieval_count": 1, "inference_seconds": 0,
            "verification_seconds": 0, "total_method_seconds": 0,
            "proof_operation_match": float("nan"),
        }
    ]
    summary = summarise_records(rows)
    assert summary.loc[0, "valid_numeric_response_rate"] == 0
    assert summary.loc[0, "proof_validity_rate_given_numeric_response"] != 1
    assert summary.loc[0, "proof_operation_match_rate"] != summary.loc[0, "proof_operation_match_rate"]


def test_summary_contains_tradeoff_metrics(examples):
    records = run_examples(examples[:4], generator=None, config=PipelineConfig(dataset_split="dev"))
    summary = summarise_records(records)
    required = {
        "numeric_answer_accuracy", "valid_numeric_response_rate",
        "unsupported_complete_answer_rate", "candidate_recall",
        "risk_given_numeric_response", "area_under_risk_coverage",
        "proof_operation_match_rate", "mean_total_method_seconds",
    }
    assert required.issubset(summary.columns)


def test_mcnemar_and_bootstrap_are_paired(examples):
    records = run_examples(examples[:5], generator=None, config=PipelineConfig(dataset_split="dev"))
    test = mcnemar_exact(records, "baseline", "direct_proof_lock")
    difference = paired_bootstrap_difference(records, "baseline", "direct_proof_lock")
    assert test["paired_examples"] == 5
    assert -1 <= difference["difference"] <= 1


def test_candidate_prompt_is_bounded_and_compact(examples):
    from numguard_fin.candidates import build_candidate_lattice
    from numguard_fin.evidence import build_evidence_ledger
    from numguard_fin.intent import parse_question_intent
    from numguard_fin.model import build_evidence_prompt
    from numguard_fin.retrieval import retrieve_evidence

    example = examples[0]
    intent = parse_question_intent(example.question)
    evidence = build_evidence_ledger(example)
    retrieved = retrieve_evidence(evidence, example.question, intent, top_k=32)
    lattice = build_candidate_lattice(retrieved, intent, max_direct=32, max_derived=128)
    prompt = build_evidence_prompt(
        example.question,
        retrieved,
        lattice=lattice,
        include_candidates=True,
        candidate_display_limit=3,
    )
    assert prompt.count("support=") <= 3
    assert "expression=" not in prompt
    assert "INSUFFICIENT_EVIDENCE" in prompt
