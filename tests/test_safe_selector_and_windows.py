from decimal import Decimal

from numguard_fin.candidates import build_candidate_lattice
from numguard_fin.dataset import coerce_example
from numguard_fin.evidence import build_evidence_ledger
from numguard_fin.intent import parse_question_intent
from numguard_fin.retrieval import retrieve_evidence
from numguard_fin.schemas import AnswerCandidate, CandidateProof
from numguard_fin.selector import FEATURE_NAMES, LinearCandidateSelector


def _lattice(raw, *, top_k=48, max_direct=48, max_derived=192):
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
    return intent, lattice


def test_three_year_average_window_is_retained_inside_longer_row():
    raw = {
        "id": "window-average",
        "pre_text": ["Amounts in millions."],
        "table": [
            ["Metric", "2019", "2020", "2021", "2022", "2023"],
            ["Effective tax rate", "20", "22", "25", "27", "28"],
        ],
        "qa": {
            "question": "What was the average effective tax rate for the three year period?",
            "answer": "26.67",
            "program": "table_average(effective tax rate, none)",
        },
    }
    intent, lattice = _lattice(raw)
    assert intent.period_count_hint == 3
    assert lattice.contains_reference("26.6667") is True
    candidate = lattice.find_by_answer("26.6667")
    assert candidate is not None
    assert candidate.proof.operation == "table_average"
    assert len(candidate.evidence_ids) == 3


def test_three_year_max_window_survives_extra_years():
    raw = {
        "id": "window-max",
        "table": [
            ["Metric", "2019", "2020", "2021", "2022", "2023"],
            ["Gross margin percentage", "24%", "25%", "26%", "27.9%", "27%"],
        ],
        "qa": {
            "question": "What was the greatest gross margin percentage in the three year period?",
            "answer": "27.9%",
            "program": "table_max(gross margin percentage, none)",
        },
    }
    intent, lattice = _lattice(raw)
    assert intent.operation == "table_max"
    assert lattice.contains_reference("27.9%") is True


def test_safe_selector_blend_can_fall_back_to_heuristic():
    candidate = AnswerCandidate(
        candidate_id="candidate",
        answer="10",
        value=Decimal("10"),
        answer_type="plain",
        proof=CandidateProof("direct", "lookup", ("e1",), "e1", True, True),
        evidence_ids=("e1",),
        retrieval_score=1.0,
        semantic_score=1.0,
        proof_score=1.0,
        confidence=0.83,
        explanation="fixture",
        feature_values={"direct": 1.0},
    )
    selector = LinearCandidateSelector(
        coefficients=tuple(0.0 for _ in FEATURE_NAMES),
        intercept=-20.0,
        blend_weight=0.0,
    )
    assert selector.learned_score(candidate) < 0.001
    assert selector.score(candidate) == 0.83
    restored = LinearCandidateSelector.from_dict(selector.to_dict())
    assert restored.blend_weight == 0.0
    assert restored.score(candidate) == 0.83


def test_monetary_gross_margin_is_not_mistyped_as_percentage():
    intent = parse_question_intent(
        "What was the greatest gross margin in millions for the three year period?"
    )
    assert intent.operation == "table_max"
    assert intent.target_scale == "million"
    assert intent.expected_answer_type == "plain"
