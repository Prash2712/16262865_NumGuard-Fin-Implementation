from numguard_fin.candidates import build_candidate_lattice
from numguard_fin.dataset import coerce_example
from numguard_fin.evidence import build_evidence_ledger
from numguard_fin.intent import parse_question_intent
from numguard_fin.retrieval import retrieve_evidence


def _lattice(raw, *, top_k=32, max_direct=32, max_derived=128):
    example = coerce_example(raw)
    intent = parse_question_intent(example.question)
    ledger = build_evidence_ledger(example)
    retrieved = retrieve_evidence(ledger, example.question, intent, top_k=top_k)
    lattice = build_candidate_lattice(
        retrieved,
        intent,
        max_direct=max_direct,
        max_derived=max_derived,
    )
    return example, intent, ledger, retrieved, lattice


def test_column_oriented_table_average_is_certified():
    raw = {
        "id": "column-average",
        "pre_text": ["Amounts in millions."],
        "table": [
            ["Year", "Segment revenue"],
            ["2021", "6000"],
            ["2022", "6200"],
            ["2023", "6220"],
        ],
        "qa": {
            "question": "What was the average segment revenue for the three year period?",
            "answer": "6140",
            "program": "table_average(segment revenue, none)",
        },
    }
    _, intent, ledger, retrieved, lattice = _lattice(raw)
    assert intent.operation == "average"
    assert {item.period for item in ledger if item.column_label == "Segment revenue"} == {
        "2021", "2022", "2023"
    }
    assert lattice.contains_reference("6140") is True
    assert any(
        candidate.answer == "6140" and candidate.proof.operation == "table_average"
        for candidate in lattice.numeric_candidates()
    )


def test_row_sum_excludes_precomputed_total_cell():
    raw = {
        "id": "row-total-exclusion",
        "pre_text": ["Amounts in millions."],
        "table": [
            ["Commitment", "2024", "2025", "Thereafter", "Total"],
            ["Contractual commitments", "100", "200", "300", "600"],
        ],
        "qa": {
            "question": "What are the total contractual commitments, in millions?",
            "answer": "600",
            "program": "table_sum(contractual commitments, none)",
        },
    }
    _, intent, _, _, lattice = _lattice(raw)
    assert intent.operation == "sum"
    derived = [
        candidate for candidate in lattice.numeric_candidates()
        if candidate.answer == "600" and candidate.proof.proof_type == "derived"
    ]
    assert derived
    assert any(candidate.proof.operation == "table_sum" for candidate in derived)
    assert all("c4" not in candidate.evidence_ids for candidate in derived)
    assert not any(
        candidate.answer == "1200" and candidate.proof.proof_type == "derived"
        for candidate in lattice.numeric_candidates()
    )


def test_average_to_highest_composite_is_proof_locked():
    raw = {
        "id": "aggregate-composite",
        "table": [
            ["Metric", "2021", "2022", "2023"],
            ["Operating margin", "10%", "12%", "14%"],
        ],
        "qa": {
            "question": "What is the variation between the average and the highest operating margin?",
            "answer": "2%",
            "program": "table_average(operating margin, none), table_max(operating margin, none), subtract(#1, #0)",
        },
    }
    _, intent, _, _, lattice = _lattice(raw)
    assert intent.operation == "difference"
    assert "table_average" in intent.alternative_operations
    assert "table_max" in intent.alternative_operations
    candidate = lattice.find_by_answer("2%")
    assert candidate is not None
    assert candidate.proof.operation == "aggregate_difference"
    assert len(candidate.evidence_ids) == 3


def test_extreme_margin_wording_prefers_table_max():
    intent = parse_question_intent(
        "What was the greatest gross margin percentage in the three year period?"
    )
    assert intent.operation == "table_max"
    assert intent.expected_answer_type == "percentage"
    assert "margin" in intent.alternative_operations


def test_column_sum_respects_requested_periods():
    raw = {
        "id": "column-period-sum",
        "pre_text": ["Amounts in thousands."],
        "table": [
            ["Year", "Net undeveloped acres", "Other acres"],
            ["2024", "1000", "50"],
            ["2025", "2000", "60"],
            ["2026", "2808", "70"],
            ["2027", "9000", "80"],
        ],
        "qa": {
            "question": "What was total net undeveloped acres for the three year period from 2024 to 2026, in thousands?",
            "answer": "5808",
            "program": "table_sum(net undeveloped acres, none)",
        },
    }
    _, intent, _, retrieved, lattice = _lattice(raw, top_k=10)
    assert intent.operation == "sum"
    assert intent.periods == ("2024", "2025", "2026")
    target = [entry for entry in retrieved if entry.item.column_label == "Net undeveloped acres"]
    assert {entry.item.period for entry in target} >= {"2024", "2025", "2026"}
    assert lattice.contains_reference("5808") is True
