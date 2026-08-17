from __future__ import annotations

from typing import Iterable, Optional

from .candidates import ABSTAIN_RESPONSE, CandidateLattice
from .numeric import extract_numbers
from .schemas import EvidenceItem, ProofCertificate


ABSTENTION_PHRASES = {
    ABSTAIN_RESPONSE.lower(),
    "insufficient evidence",
    "cannot determine",
    "not enough information",
    "unable to answer",
    "abstain",
}


def is_abstention(response: str) -> bool:
    normalised = " ".join(response.strip().lower().split())
    return normalised in ABSTENTION_PHRASES or any(
        phrase in normalised for phrase in ABSTENTION_PHRASES if phrase != ABSTAIN_RESPONSE.lower()
    )


def is_valid_numeric_response(response: str) -> bool:
    if is_abstention(response):
        return False
    stripped = response.strip()
    numbers = extract_numbers(stripped)
    if len(numbers) != 1:
        return False
    number = numbers[0]
    # A valid commitment is the complete numeric expression, not prose containing a
    # number. This rejects outputs such as "number of millions 12" while retaining
    # forms such as "$3.8 million", "(45)" and "12.4%".
    return (
        bool(stripped)
        and any(character.isdigit() for character in stripped)
        and number.start == 0
        and number.end == len(stripped)
    )


def verify_response(
    response: str,
    lattice: CandidateLattice,
    evidence: Iterable[EvidenceItem],
) -> ProofCertificate:
    cleaned = response.strip()
    if is_abstention(cleaned):
        return ProofCertificate(
            decision="abstained",
            response=cleaned,
            candidate_id="abstain",
            proof_type="abstain",
            operation="abstain",
            evidence_ids=(),
            evidence_claims=(),
            expression=None,
            answer_type="abstain",
            candidate_confidence=None,
            reason="The system withheld an answer.",
        )

    if not is_valid_numeric_response(cleaned):
        return ProofCertificate(
            decision="rejected",
            response=cleaned,
            candidate_id=None,
            proof_type=None,
            operation=None,
            evidence_ids=(),
            evidence_claims=(),
            expression=None,
            answer_type=None,
            candidate_confidence=None,
            reason="The response is not a complete single numeric answer.",
        )

    candidate = lattice.find_by_answer(cleaned)
    if candidate is None:
        return ProofCertificate(
            decision="rejected",
            response=cleaned,
            candidate_id=None,
            proof_type=None,
            operation=None,
            evidence_ids=(),
            evidence_claims=(),
            expression=None,
            answer_type=None,
            candidate_confidence=None,
            reason="The complete answer is absent from the question-conditioned proof lattice.",
        )

    evidence_map = {item.evidence_id: item for item in evidence}
    missing = [evidence_id for evidence_id in candidate.evidence_ids if evidence_id not in evidence_map]
    if missing:
        return ProofCertificate(
            decision="rejected",
            response=cleaned,
            candidate_id=candidate.candidate_id,
            proof_type=candidate.proof.proof_type,
            operation=candidate.proof.operation,
            evidence_ids=candidate.evidence_ids,
            evidence_claims=(),
            expression=candidate.proof.expression,
            answer_type=candidate.answer_type,
            candidate_confidence=candidate.confidence,
            reason=f"Proof references unavailable evidence: {', '.join(missing)}.",
        )
    if not candidate.proof.compatible_units:
        return ProofCertificate(
            decision="rejected",
            response=cleaned,
            candidate_id=candidate.candidate_id,
            proof_type=candidate.proof.proof_type,
            operation=candidate.proof.operation,
            evidence_ids=candidate.evidence_ids,
            evidence_claims=tuple(evidence_map[evidence_id].to_dict() for evidence_id in candidate.evidence_ids),
            expression=candidate.proof.expression,
            answer_type=candidate.answer_type,
            candidate_confidence=candidate.confidence,
            reason="The proof combines incompatible units or currencies.",
        )

    return ProofCertificate(
        decision="accepted",
        response=cleaned,
        candidate_id=candidate.candidate_id,
        proof_type=candidate.proof.proof_type,
        operation=candidate.proof.operation,
        evidence_ids=candidate.evidence_ids,
        evidence_claims=tuple(evidence_map[evidence_id].to_dict() for evidence_id in candidate.evidence_ids),
        expression=candidate.proof.expression,
        answer_type=candidate.answer_type,
        candidate_confidence=candidate.confidence,
        reason="The complete answer has a valid question-conditioned provenance proof.",
    )
