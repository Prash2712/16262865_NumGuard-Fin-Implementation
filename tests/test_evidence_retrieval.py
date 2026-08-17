from decimal import Decimal

from numguard_fin.evidence import build_evidence_ledger, evidence_by_id
from numguard_fin.intent import parse_question_intent
from numguard_fin.retrieval import retrieve_evidence, score_evidence


def test_table_provenance_coordinates(example_by_suffix):
    example = example_by_suffix("report_01.pdf-1")
    ledger = build_evidence_ledger(example)
    target = next(item for item in ledger if item.value == Decimal("380"))
    assert target.evidence_id == "table:r1:c1:n0"
    assert target.row_label == "Interest expense"
    assert target.column_label == "2009"
    assert target.period == "2009"


def test_long_integer_preserved_in_ledger(example_by_suffix):
    ledger = build_evidence_ledger(example_by_suffix("report_11.pdf-1"))
    assert any(item.canonical == "41932" for item in ledger)
    assert not any(item.canonical in {"419", "32"} for item in ledger)


def test_negative_parentheses_preserved(example_by_suffix):
    ledger = build_evidence_ledger(example_by_suffix("report_10.pdf-1"))
    assert any(item.value == Decimal("-45") for item in ledger)


def test_scale_and_currency_propagation(example_by_suffix):
    ledger = build_evidence_ledger(example_by_suffix("report_04.pdf-1"))
    numeric = [item for item in ledger if not item.is_year]
    assert numeric
    assert all(item.scale == "million" for item in numeric)
    assert all(item.currency == "USD" for item in numeric)


def test_evidence_ids_are_unique(examples):
    for example in examples:
        ledger = build_evidence_ledger(example)
        assert len({item.evidence_id for item in ledger}) == len(ledger)


def test_evidence_map(example_by_suffix):
    ledger = build_evidence_ledger(example_by_suffix("report_01.pdf-1"))
    mapped = evidence_by_id(ledger)
    assert set(mapped) == {item.evidence_id for item in ledger}


def test_retrieval_prefers_requested_metric_and_period(example_by_suffix):
    example = example_by_suffix("report_01.pdf-1")
    intent = parse_question_intent(example.question)
    retrieved = retrieve_evidence(build_evidence_ledger(example), example.question, intent, top_k=4)
    assert retrieved[0].item.row_label == "Interest expense"
    assert retrieved[0].item.period == "2009"


def test_irrelevant_metric_is_penalised(example_by_suffix):
    example = example_by_suffix("report_01.pdf-1")
    intent = parse_question_intent(example.question)
    ledger = build_evidence_ledger(example)
    interest_item = next(item for item in ledger if item.row_label == "Interest expense" and item.period == "2009")
    debt_item = next(item for item in ledger if item.row_label == "Revenue" and item.period == "2009")
    interest = score_evidence(interest_item, example.question, intent)
    debt = score_evidence(debt_item, example.question, intent)
    assert interest.score > debt.score


def test_retrieval_is_deterministic(example_by_suffix):
    example = example_by_suffix("report_05.pdf-1")
    intent = parse_question_intent(example.question)
    ledger = build_evidence_ledger(example)
    one = [entry.item.evidence_id for entry in retrieve_evidence(ledger, example.question, intent)]
    two = [entry.item.evidence_id for entry in retrieve_evidence(ledger, example.question, intent)]
    assert one == two
