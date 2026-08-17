from decimal import Decimal

import pytest

from numguard_fin.candidates import ABSTAIN_RESPONSE, build_candidate_lattice
from numguard_fin.evidence import build_evidence_ledger
from numguard_fin.intent import parse_question_intent
from numguard_fin.proof import is_abstention, is_valid_numeric_response, verify_response
from numguard_fin.retrieval import retrieve_evidence


def lattice_for(example, include_derived=True):
    intent = parse_question_intent(example.question)
    ledger = build_evidence_ledger(example)
    retrieved = retrieve_evidence(ledger, example.question, intent, top_k=10)
    return ledger, build_candidate_lattice(retrieved, intent, include_derived=include_derived)


@pytest.mark.parametrize(
    "suffix",
    [
        "report_01.pdf-1",
        "report_02.pdf-1",
        "report_03.pdf-1",
        "report_04.pdf-1",
        "report_05.pdf-1",
        "report_06.pdf-1",
        "report_07.pdf-1",
        "report_08.pdf-1",
        "report_09.pdf-1",
        "report_10.pdf-1",
        "report_11.pdf-1",
    ],
)
def test_reference_is_in_question_conditioned_lattice(example_by_suffix, suffix):
    example = example_by_suffix(suffix)
    _, lattice = lattice_for(example)
    assert lattice.contains_reference(example.reference_answer) is True


def test_unsupported_reference_is_not_in_lattice(example_by_suffix):
    example = example_by_suffix("report_12.pdf-1")
    _, lattice = lattice_for(example)
    assert lattice.contains_reference(example.reference_answer) is False


def test_candidate_bound(examples):
    for example in examples:
        _, lattice = lattice_for(example)
        assert lattice.direct_count <= 48
        assert lattice.derived_count <= 192
        assert len(lattice.candidates) <= 241


def test_margin_has_derived_proof(example_by_suffix):
    example = example_by_suffix("report_04.pdf-1")
    _, lattice = lattice_for(example)
    candidate = lattice.find_by_answer("26.19%")
    assert candidate is not None
    assert candidate.proof.proof_type == "derived"
    assert candidate.proof.operation == "margin"


def test_sum_has_bounded_operands(example_by_suffix):
    example = example_by_suffix("report_05.pdf-1")
    _, lattice = lattice_for(example)
    candidate = lattice.find_by_answer("2000")
    assert candidate is not None
    assert len(candidate.evidence_ids) == 2
    assert candidate.proof.operation == "addition"


@pytest.mark.parametrize(
    "response,valid",
    [
        ("380", True),
        ("12.4%", True),
        ("$3.8 million", True),
        ("(45)", True),
        ("number of millions 12", False),
        ("The answer is 12", False),
        ("12 and 13", False),
        ("percentage", False),
        (".3", True),
    ],
)
def test_valid_numeric_response_is_complete(response, valid):
    assert is_valid_numeric_response(response) is valid


def test_abstention_recognition():
    assert is_abstention(ABSTAIN_RESPONSE)
    assert is_abstention("Insufficient evidence")
    assert not is_abstention("380")


def test_proof_accepts_certified_complete_answer(example_by_suffix):
    example = example_by_suffix("report_01.pdf-1")
    ledger, lattice = lattice_for(example)
    certificate = verify_response("380", lattice, ledger)
    assert certificate.decision == "accepted"
    assert certificate.evidence_ids == ("table:r1:c1:n0",)
    assert certificate.evidence_claims[0]["row_label"] == "Interest expense"
    assert certificate.evidence_claims[0]["period"] == "2009"
    assert certificate.candidate_confidence is not None


def test_proof_rejects_uncertified_complete_answer(example_by_suffix):
    example = example_by_suffix("report_01.pdf-1")
    ledger, lattice = lattice_for(example)
    certificate = verify_response("999", lattice, ledger)
    assert certificate.decision == "rejected"


def test_proof_rejects_prose_answer(example_by_suffix):
    example = example_by_suffix("report_01.pdf-1")
    ledger, lattice = lattice_for(example)
    assert verify_response("The answer is 380", lattice, ledger).decision == "rejected"


def test_proof_abstains_explicitly(example_by_suffix):
    example = example_by_suffix("report_01.pdf-1")
    ledger, lattice = lattice_for(example)
    assert verify_response(ABSTAIN_RESPONSE, lattice, ledger).decision == "abstained"
