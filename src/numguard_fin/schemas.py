from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any, Literal, Optional


SourceType = Literal["table", "text"]
ProofType = Literal["direct", "derived", "abstain"]


@dataclass(frozen=True)
class FinQAExample:
    example_id: str
    question: str
    pre_text: tuple[str, ...]
    post_text: tuple[str, ...]
    table: tuple[tuple[str, ...], ...]
    reference_answer: Optional[str] = None
    reference_program: Optional[str] = None
    reference_explanation: Optional[str] = None
    reference_support: dict[str, Any] = field(default_factory=dict)
    filename: Optional[str] = None

    def to_dict(self, include_references: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_references:
            for key in (
                "reference_answer",
                "reference_program",
                "reference_explanation",
                "reference_support",
            ):
                payload.pop(key, None)
        return payload


@dataclass(frozen=True)
class ParsedNumber:
    raw: str
    value: Decimal
    canonical: str
    start: int
    end: int
    is_percent: bool = False
    is_year: bool = False
    currency: Optional[str] = None
    explicit_scale: Optional[str] = None
    effective_scale: Optional[str] = None
    base_value: Optional[Decimal] = None
    negative_parentheses: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("value", "base_value"):
            value = payload.get(key)
            if isinstance(value, Decimal):
                payload[key] = format(value, "f")
        return payload


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    source_type: SourceType
    raw_number: str
    value: Decimal
    canonical: str
    base_value: Decimal
    is_percent: bool
    is_year: bool
    currency: Optional[str]
    scale: Optional[str]
    text: str
    metric_text: str
    period: Optional[str]
    row_index: Optional[int] = None
    column_index: Optional[int] = None
    row_label: Optional[str] = None
    column_label: Optional[str] = None
    sentence_index: Optional[int] = None
    character_start: Optional[int] = None
    character_end: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("value", "base_value"):
            payload[key] = format(payload[key], "f")
        return payload


@dataclass(frozen=True)
class QuestionIntent:
    operation: str
    expected_answer_type: str
    metric_terms: tuple[str, ...]
    periods: tuple[str, ...]
    ordered_periods: tuple[str, ...]
    asks_for_change: bool
    asks_for_absolute_change: bool
    confidence: float
    aggregation: Optional[str] = None
    target_scale: Optional[str] = None
    period_count_hint: Optional[int] = None
    explicit_constants: tuple[Decimal, ...] = ()
    numerator_terms: tuple[str, ...] = ()
    denominator_terms: tuple[str, ...] = ()
    entity_terms: tuple[str, ...] = ()
    alternative_operations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["explicit_constants"] = [format(value, "f") for value in self.explicit_constants]
        return payload


@dataclass(frozen=True)
class RetrievedEvidence:
    item: EvidenceItem
    score: float
    lexical_score: float
    period_score: float
    metric_score: float
    type_score: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["item"] = self.item.to_dict()
        return payload


@dataclass(frozen=True)
class CandidateProof:
    proof_type: ProofType
    operation: str
    operand_ids: tuple[str, ...]
    expression: str
    compatible_units: bool
    question_conditioned: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerCandidate:
    candidate_id: str
    answer: str
    value: Optional[Decimal]
    answer_type: str
    proof: CandidateProof
    evidence_ids: tuple[str, ...]
    retrieval_score: float
    semantic_score: float
    proof_score: float
    confidence: float
    explanation: str
    feature_values: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["value"] = None if self.value is None else format(self.value, "f")
        payload["proof"] = self.proof.to_dict()
        return payload


@dataclass(frozen=True)
class ProofCertificate:
    decision: Literal["accepted", "abstained", "rejected"]
    response: str
    candidate_id: Optional[str]
    proof_type: Optional[str]
    operation: Optional[str]
    evidence_ids: tuple[str, ...]
    evidence_claims: tuple[dict[str, Any], ...]
    expression: Optional[str]
    answer_type: Optional[str]
    candidate_confidence: Optional[float]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionRecord:
    example_id: str
    method: str
    question: str
    reference_answer: Optional[str]
    model_response: str
    valid_numeric_response: bool
    abstained: bool
    invalid_response: bool
    numeric_exact_match: Optional[bool]
    candidate_recall: Optional[bool]
    proof_valid: bool
    unsupported_complete_answer: bool
    selected_candidate_id: Optional[str]
    selected_candidate_confidence: Optional[float]
    selected_candidate_margin: Optional[float]
    provenance_risk_score: Optional[float]
    semantic_risk_score: Optional[float]
    proof_certificate: dict[str, Any]
    candidate_count: int
    direct_candidate_count: int
    derived_candidate_count: int
    retrieval_count: int
    inference_seconds: float
    verification_seconds: float
    total_method_seconds: float
    input_token_count: int
    input_truncated: bool
    model_sequence_score: Optional[float]
    model_name: str
    dataset_split: str
    seed: int
    configuration_id: str
    reference_program: Optional[str] = None
    predicted_program: Optional[str] = None
    reference_operation: Optional[str] = None
    predicted_operation: Optional[str] = None
    proof_operation_match: Optional[bool] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
