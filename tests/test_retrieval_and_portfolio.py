from numguard_fin.candidates import build_candidate_lattice
from numguard_fin.dataset import coerce_example
from numguard_fin.evidence import build_evidence_ledger
from numguard_fin.intent import parse_question_intent
from numguard_fin.retrieval import retrieve_evidence


def _components(raw, *, top_k=24, max_direct=24, max_derived=96, scorer=None):
    example = coerce_example(raw)
    intent = parse_question_intent(example.question)
    ledger = build_evidence_ledger(example)
    retrieved = retrieve_evidence(ledger, example.question, intent, top_k=top_k)
    lattice = build_candidate_lattice(
        retrieved,
        intent,
        max_direct=max_direct,
        max_derived=max_derived,
        candidate_scorer=scorer,
    )
    return example, intent, ledger, retrieved, lattice


def test_per_transaction_wording_is_ratio_not_average():
    intent = parse_question_intent(
        "What is the average payment volume per transaction for American Express?"
    )
    assert intent.operation == "ratio"
    assert "average" in intent.alternative_operations


def test_percentage_of_increase_is_margin_not_difference():
    intent = parse_question_intent(
        "What were deferred fuel cost revisions as a percentage of the increase in fuel expense?"
    )
    assert intent.operation == "margin"
    assert "difference" in intent.alternative_operations


def test_total_across_two_periods_is_sum():
    intent = parse_question_intent(
        "What was the total residential mortgages balance for 2013 and 2012?"
    )
    assert intent.operation == "sum"
    assert intent.periods == ("2013", "2012")


def test_cumulative_indexed_return_candidate():
    raw = {
        "id": "indexed-return",
        "pre_text": ["The comparison assumes an initial investment of $100."],
        "table": [["Investment", "2017"], ["Citi common stock", "193.5"]],
        "qa": {
            "question": "What was the percentage cumulative total return for the five year period ended 31-Dec-2017 of Citi common stock?",
            "answer": "93.5%",
            "program": "subtract(193.5, const_100)",
        },
    }
    _, intent, _, _, lattice = _components(raw)
    assert intent.operation == "indexed_return"
    candidate = lattice.find_by_answer("93.5%")
    assert candidate is not None
    assert candidate.proof.operation == "indexed_return"


def test_generic_average_remains_average():
    intent = parse_question_intent(
        "What was the average net sales from 2020 to 2022?"
    )
    assert intent.operation == "average"
    assert intent.period_count_hint == 3


def test_retrieval_preserves_all_requested_period_cells_under_small_budget():
    raw = {
        "id": "period-protection",
        "pre_text": ["Amounts in millions."],
        "table": [
            ["Metric", "2021", "2022", "2023"],
            ["Residential mortgages balance", "50", "60", "70"],
            ["Residential mortgage applications", "900", "950", "1000"],
            ["Other consumer loans", "400", "420", "440"],
        ],
        "qa": {
            "question": "What was the total residential mortgages balance for 2022 and 2023?",
            "answer": "130",
            "program": "add(60, 70)",
        },
    }
    _, intent, _, retrieved, lattice = _components(raw, top_k=4)
    target = [
        entry for entry in retrieved
        if entry.item.row_label == "Residential mortgages balance"
        and entry.item.period in {"2022", "2023"}
    ]
    assert {entry.item.period for entry in target} == {"2022", "2023"}
    assert lattice.contains_reference("130") is True
    assert intent.operation == "sum"


def test_retrieval_guarantees_numerator_and_denominator_rows():
    raw = {
        "id": "role-protection",
        "pre_text": ["Amounts in millions."],
        "table": [
            ["Metric", "2021"],
            ["Total sales", "1000"],
            ["Europe sales", "240"],
            ["European employee count", "50000"],
            ["Total assets", "8000"],
            ["Asia sales", "300"],
        ],
        "qa": {
            "question": "What percentage of total sales came from Europe sales in 2021?",
            "answer": "24%",
            "program": "divide(240, 1000), multiply(#0, const_100)",
        },
    }
    _, intent, _, retrieved, lattice = _components(raw, top_k=4)
    row_labels = {entry.item.row_label for entry in retrieved}
    assert "Europe sales" in row_labels
    assert "Total sales" in row_labels
    assert lattice.contains_reference("24%") is True
    assert intent.numerator_terms
    assert intent.denominator_terms


def test_selector_rescores_before_direct_candidate_bound():
    raw = {
        "id": "selector-before-prune",
        "table": [["Metric", "2023"], ["Primary revenue", "10"], ["Secondary revenue", "20"]],
        "qa": {"question": "What was revenue in 2023?", "answer": "20"},
    }

    def scorer(candidate):
        return 1.0 if candidate.answer == "20" else 0.0

    _, _, _, _, lattice = _components(raw, max_direct=1, max_derived=0, scorer=scorer)
    numeric = lattice.numeric_candidates()
    assert len(numeric) == 1
    assert numeric[0].answer == "20"


def test_diverse_pruning_keeps_multiple_licensed_operations():
    raw = {
        "id": "operation-diversity",
        "pre_text": ["Amounts in millions."],
        "table": [["Metric", "2021", "2022", "2023"], ["Net sales", "10", "20", "30"]],
        "qa": {
            "question": "What were the total and average net sales from 2021 to 2023?",
            "answer": "60",
        },
    }
    _, _, _, _, lattice = _components(raw, max_direct=0, max_derived=2)
    operations = {candidate.proof.operation for candidate in lattice.numeric_candidates()}
    assert operations & {"addition", "table_sum"}
    assert operations & {"average", "table_average"}


def test_selector_bypasses_heuristic_floor_before_frozen_bound():
    raw = {
        "id": "selector-before-floor",
        "table": [["Metric", "2023"], ["Revenue", "20"]],
        "qa": {"question": "What was revenue in 2023?", "answer": "20"},
    }
    example = coerce_example(raw)
    intent = parse_question_intent(example.question)
    ledger = build_evidence_ledger(example)
    retrieved = retrieve_evidence(ledger, example.question, intent, top_k=2)
    lattice = build_candidate_lattice(
        retrieved,
        intent,
        max_direct=1,
        max_derived=0,
        minimum_direct_confidence=1.1,
        candidate_scorer=lambda candidate: 0.9,
    )
    assert lattice.contains_reference("20") is True
