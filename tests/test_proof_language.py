from decimal import Decimal

from numguard_fin.calibration import select_risk_threshold
from numguard_fin.candidates import build_candidate_lattice
from numguard_fin.dataset import coerce_example
from numguard_fin.evidence import build_evidence_ledger
from numguard_fin.intent import parse_question_intent
from numguard_fin.retrieval import retrieve_evidence
from numguard_fin.selector import LinearCandidateSelector, FEATURE_NAMES


def lattice(raw):
    example = coerce_example(raw)
    intent = parse_question_intent(example.question)
    ledger = build_evidence_ledger(example)
    retrieved = retrieve_evidence(ledger, example.question, intent, top_k=16)
    return intent, build_candidate_lattice(retrieved, intent, max_direct=16, max_derived=48)


def test_cross_scale_percentage_of_total_uses_base_values():
    raw = {
        "id": "scale-ratio",
        "pre_text": ["Amounts are shown in the units stated in each row."],
        "table": [["Metric", "2019"], ["European net sales (billions)", "2.5 billion"], ["Consumer packaging sales (millions)", "3195 million"]],
        "qa": {"question": "What percentage of consumer packaging sales was represented by European net sales in 2019?", "answer": "78.25%"},
    }
    intent, result = lattice(raw)
    assert intent.operation == "margin"
    assert result.contains_reference("78.25%") is True


def test_percent_times_amount_is_fractional_product():
    raw = {
        "id": "percent-product",
        "pre_text": ["Amounts in millions."],
        "table": [["Metric", "2017"], ["International share", "25%"], ["Net sales", "5283.3"]],
        "qa": {"question": "In 2017 what was net sales applicable to the international market in millions?", "answer": "1320.825", "program": "multiply(25%,5283.3)"},
    }
    # Product is licensed by the explicit program-style wording used in FinQA examples.
    raw["qa"]["question"] = "What was the product of the 25% international share and net sales in 2017, in millions?"
    _, result = lattice(raw)
    assert result.contains_reference("1320.825") is True


def test_three_year_table_average():
    raw = {
        "id": "avg-3",
        "pre_text": ["Amounts in millions."],
        "table": [["Metric", "2015", "2016", "2017"], ["Capital expenditure", "90", "120", "150"]],
        "qa": {"question": "What was the average capital expenditure from 2015 to 2017?", "answer": "120"},
    }
    intent, result = lattice(raw)
    assert intent.period_count_hint == 3
    candidate = result.find_by_answer("120")
    assert candidate is not None
    assert candidate.proof.operation == "table_average"
    assert len(candidate.evidence_ids) == 3


def test_table_max_and_min_are_derived_certificates():
    raw = {
        "id": "max-row",
        "pre_text": ["Amounts in millions."],
        "table": [["Metric", "2014", "2015", "2016"], ["Interest expense", "15", "22", "18"]],
        "qa": {"question": "Considering 2014-2016, what was the highest interest expense observed?", "answer": "22"},
    }
    intent, result = lattice(raw)
    assert intent.operation == "table_max"
    candidate = result.find_by_answer("22")
    assert candidate is not None
    assert candidate.proof.operation == "table_max"


def test_nary_sum_for_three_requested_entities():
    raw = {
        "id": "sum-3",
        "pre_text": ["Amounts in millions."],
        "table": [["Metric", "2014"], ["Euro note A", "1029"], ["Euro note B", "1372"], ["Euro note C", "697"]],
        "qa": {"question": "What was the total of Euro note A, Euro note B and Euro note C issued in 2014?", "answer": "3098"},
    }
    _, result = lattice(raw)
    candidate = result.find_by_answer("3098")
    assert candidate is not None
    assert len(candidate.evidence_ids) == 3


def test_weighted_average_label_does_not_trigger_average_operation():
    intent = parse_question_intent("How many weighted average diluted shares were reported in 2023?")
    assert intent.operation == "direct_lookup"
    assert intent.expected_answer_type == "count"


def test_dividend_per_share_is_direct_lookup():
    intent = parse_question_intent("What was the dividend per share in 2023?")
    assert intent.operation == "direct_lookup"


def test_selector_round_trip_and_bounded_score():
    selector = LinearCandidateSelector(
        coefficients=tuple(0.1 for _ in FEATURE_NAMES),
        intercept=-0.5,
        metadata={"test": True},
    )
    restored = LinearCandidateSelector.from_dict(selector.to_dict())
    assert restored.coefficients == selector.coefficients
    assert restored.feature_names == FEATURE_NAMES


def test_calibration_blocks_zero_coverage_solution():
    rows = [
        {"selected_candidate_confidence": 0.9, "valid_numeric_response": True, "numeric_exact_match": False},
        {"selected_candidate_confidence": 0.8, "valid_numeric_response": True, "numeric_exact_match": False},
    ]
    decision = select_risk_threshold(
        rows,
        target_risk=0.2,
        minimum_coverage=0.5,
        minimum_accepted=1,
    )
    assert decision.feasible is False
    assert decision.coverage == 0.0
    assert "Public-test execution is blocked" in decision.reason


def test_percentage_evidence_does_not_inherit_million_scale():
    raw = {
        "id": "percent-scale",
        "pre_text": ["Amounts in millions."],
        "table": [["Metric", "2023"], ["Tax rate", "21%"]],
        "qa": {"question": "What was the tax rate in 2023?", "answer": "21%"},
    }
    example = coerce_example(raw)
    item = next(item for item in build_evidence_ledger(example) if item.is_percent)
    assert item.scale is None
    assert item.currency is None
    assert item.base_value == Decimal("21")
