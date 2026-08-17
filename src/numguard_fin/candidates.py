from __future__ import annotations

import hashlib
import itertools
import math
import re
from dataclasses import dataclass, replace
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import Callable, Iterable, Optional

from .numeric import SCALE_MULTIPLIERS, format_answer, numeric_equal, parse_single_number
from .retrieval import tokens
from .schemas import (
    AnswerCandidate,
    CandidateProof,
    EvidenceItem,
    QuestionIntent,
    RetrievedEvidence,
)


ABSTAIN_RESPONSE = "INSUFFICIENT_EVIDENCE"
DERIVATION_OPERATIONS = {
    "difference", "percentage_change", "sum", "average", "ratio", "margin",
    "product", "table_sum", "table_average", "table_min", "table_max", "indexed_return",
}

CandidateScorer = Callable[[AnswerCandidate], float]


def _operation_portfolio(intent: QuestionIntent) -> tuple[str, ...]:
    """Return a bounded set of question-licensed operation families.

    The primary intent remains the strongest signal. Alternatives come only from explicit
    lexical cues recorded by the intent parser; this is not unrestricted arithmetic search.
    """
    ordered: list[str] = []
    for operation in (intent.operation, *intent.alternative_operations):
        if operation not in ordered:
            ordered.append(operation)
    return tuple(ordered[:9])


@dataclass(frozen=True)
class CandidateLattice:
    candidates: tuple[AnswerCandidate, ...]
    direct_count: int
    derived_count: int
    retrieval_count: int

    @property
    def answer_strings(self) -> tuple[str, ...]:
        return tuple(candidate.answer for candidate in self.candidates)

    def numeric_candidates(self) -> tuple[AnswerCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.proof.proof_type != "abstain")

    def find_by_answer(self, response: str) -> Optional[AnswerCandidate]:
        cleaned = response.strip()
        exact = [candidate for candidate in self.candidates if candidate.answer == cleaned]
        if exact:
            return max(exact, key=lambda candidate: candidate.confidence)
        parsed_response = parse_single_number(cleaned)
        if parsed_response is None:
            return None
        matches: list[AnswerCandidate] = []
        for candidate in self.numeric_candidates():
            parsed_candidate = parse_single_number(candidate.answer)
            if parsed_candidate is None:
                continue
            strict_match = numeric_equal(
                parsed_response, parsed_candidate, require_percent_compatibility=True
            )
            # FinQA frequently omits the percent sign from the reference or generated
            # surface form even when the question and certified proof are percentage-typed.
            # Accept only a missing marker for an explicitly percentage-typed candidate;
            # the numeric magnitude must still match under the normal tolerances.
            marker_omission_match = (
                candidate.answer_type == "percentage"
                and numeric_equal(
                    parsed_response, parsed_candidate, require_percent_compatibility=False
                )
            )
            if strict_match or marker_omission_match:
                matches.append(candidate)
        return max(matches, key=lambda candidate: candidate.confidence) if matches else None

    def contains_reference(self, reference_answer: Optional[str]) -> Optional[bool]:
        if reference_answer is None:
            return None
        parsed_reference = parse_single_number(reference_answer)
        if parsed_reference is None:
            return False
        return any(
            (parsed := parse_single_number(candidate.answer)) is not None
            and (
                numeric_equal(parsed_reference, parsed, require_percent_compatibility=True)
                or (
                    candidate.answer_type == "percentage"
                    and numeric_equal(
                        parsed_reference, parsed, require_percent_compatibility=False
                    )
                )
            )
            for candidate in self.numeric_candidates()
        )

    def confidence_margin(self, response: str) -> Optional[float]:
        selected = self.find_by_answer(response)
        if selected is None:
            return None
        scores = sorted(
            (candidate.confidence for candidate in self.numeric_candidates()), reverse=True
        )
        if not scores:
            return None
        alternatives = [candidate.confidence for candidate in self.numeric_candidates() if candidate.candidate_id != selected.candidate_id]
        return selected.confidence - max(alternatives) if alternatives else selected.confidence


def _stable_id(prefix: str, payload: str) -> str:
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _candidate_answer(value: Decimal, answer_type: str) -> str:
    return format_answer(value, answer_type, places=4)


def _normalised_metric(item: EvidenceItem) -> str:
    return " ".join((item.row_label or item.metric_text or "").strip().lower().split())


def _same_metric(left: EvidenceItem, right: EvidenceItem) -> bool:
    a, b = _normalised_metric(left), _normalised_metric(right)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    ta, tb = set(tokens(a)), set(tokens(b))
    return bool(ta and tb and len(ta & tb) / max(1, min(len(ta), len(tb))) >= 0.67)


def _term_overlap(terms: Iterable[str], item: EvidenceItem) -> float:
    terms = set(tokens(" ".join(terms)))
    if not terms:
        return 0.0
    item_terms = set(tokens(f"{item.row_label or ''} {item.metric_text} {item.text}"))
    return len(terms & item_terms) / len(terms)


def _scale_multiplier(scale: Optional[str]) -> Decimal:
    return SCALE_MULTIPLIERS.get(scale or "", Decimal("1"))


def _value_in_scale(item: EvidenceItem, scale: Optional[str]) -> Decimal:
    if scale:
        return item.base_value / _scale_multiplier(scale)
    return item.value


def _common_output_scale(intent: QuestionIntent, operands: Iterable[EvidenceItem]) -> Optional[str]:
    if intent.target_scale:
        return intent.target_scale
    scales = {item.scale for item in operands if item.scale}
    return next(iter(scales)) if len(scales) == 1 else None


def _safe_divide(numerator: Decimal, denominator: Decimal) -> Optional[Decimal]:
    try:
        return None if denominator == 0 else numerator / denominator
    except (DivisionByZero, InvalidOperation, ZeroDivisionError):
        return None


def _compatible_units(items: Iterable[EvidenceItem], operation: str) -> bool:
    items = list(items)
    currencies = {item.currency for item in items if item.currency}
    if len(currencies) > 1:
        return False
    if operation in {
        "addition", "subtraction", "aggregate_difference", "average",
        "table_sum", "table_average", "table_min", "table_max",
    }:
        # Different scales are convertible because every EvidenceItem stores base_value.
        return all(not item.is_percent for item in items) or all(item.is_percent for item in items)
    return True


def _answer_type_for(operation: str, intent: QuestionIntent, operands: Iterable[EvidenceItem]) -> str:
    operands = list(operands)
    if operation in {"percentage_change", "margin", "percentage_of_total"}:
        return "percentage"
    if intent.expected_answer_type == "percentage":
        return "percentage"
    if operation in {
        "table_min", "table_max", "table_average", "average", "addition",
        "subtraction", "aggregate_difference",
    } and operands and all(item.is_percent for item in operands):
        return "percentage"
    return intent.expected_answer_type if intent.expected_answer_type in {"plain", "ratio", "count", "year"} else "plain"


def _feature_values(
    *,
    operation: str,
    operands: list[RetrievedEvidence],
    intent: QuestionIntent,
    compatible: bool,
    depth: int,
    role_fit: float,
    period_fit: float,
) -> dict[str, float]:
    retrieval = sum(_sigmoid(entry.score) for entry in operands) / max(1, len(operands))
    metric = sum(max(0.0, entry.metric_score) for entry in operands) / max(1, len(operands))
    type_fit = sum(max(0.0, entry.type_score) for entry in operands) / max(1, len(operands))
    aliases = {
        "difference": {"difference", "subtraction", "aggregate_difference"},
        "sum": {"sum", "addition", "table_sum"},
        "table_sum": {"sum", "addition", "table_sum"},
        "average": {"average", "table_average"},
        "table_average": {"average", "table_average"},
        "ratio": {"ratio", "division"},
        "margin": {"margin", "percentage_of_total", "division"},
        "percentage_change": {"percentage_change", "division"},
        "product": {"product", "multiplication"},
        "table_min": {"table_min"},
        "table_max": {"table_max"},
        "indexed_return": {"indexed_return", "subtraction"},
        "direct_lookup": {"lookup"},
    }
    licensed = set()
    for family in _operation_portfolio(intent):
        licensed.update(aliases.get(family, {family}))
    operation_match = float(operation in licensed)
    return {
        "retrieval": retrieval,
        "metric_fit": metric,
        "type_fit": type_fit,
        "period_fit": _clamp(period_fit),
        "role_fit": _clamp(role_fit),
        "proof_valid": float(compatible),
        "operation_match": operation_match,
        "proof_depth": float(depth),
        "operand_count": float(len(operands)),
        "direct": 0.0,
        "percentage_answer": float(intent.expected_answer_type == "percentage"),
    }


def _base_confidence(features: dict[str, float]) -> float:
    score = (
        0.18 * features["retrieval"]
        + 0.24 * features["metric_fit"]
        + 0.08 * features["type_fit"]
        + 0.17 * features["period_fit"]
        + 0.18 * features["role_fit"]
        + 0.10 * features["proof_valid"]
        + 0.14 * features["operation_match"]
        - 0.015 * max(0.0, features["proof_depth"] - 1.0)
        - 0.005 * max(0.0, features["operand_count"] - 2.0)
    )
    return _clamp(score)


def _direct_candidate(entry: RetrievedEvidence, intent: QuestionIntent) -> AnswerCandidate:
    item = entry.item
    value = _value_in_scale(item, intent.target_scale) if intent.target_scale and item.scale else item.value
    answer_type = "percentage" if item.is_percent else (
        "year" if item.is_year and intent.expected_answer_type == "year" else intent.expected_answer_type
    )
    if answer_type not in {"percentage", "year", "count", "ratio", "plain"}:
        answer_type = "plain"
    metric_fit = entry.metric_score if intent.metric_terms else max(0.0, entry.lexical_score)
    period_fit = max(0.0, entry.period_score) if intent.periods else 1.0
    type_fit = 1.0 if not item.is_year or answer_type == "year" else 0.0
    semantic = _clamp(0.50 * metric_fit + 0.30 * period_fit + 0.20 * type_fit)
    direct_match = 1.0 if intent.operation == "direct_lookup" else (
        0.65 if metric_fit >= 0.50 and period_fit >= 0.70 else 0.30
    )
    features = {
        "retrieval": _sigmoid(entry.score), "metric_fit": metric_fit, "type_fit": type_fit,
        "period_fit": period_fit, "role_fit": max(_term_overlap(intent.numerator_terms, item), _term_overlap(intent.denominator_terms, item)),
        "proof_valid": 1.0, "operation_match": direct_match,
        "proof_depth": 0.0, "operand_count": 1.0, "direct": 1.0,
        "percentage_answer": float(answer_type == "percentage"),
    }
    confidence = _clamp(0.20 * features["retrieval"] + 0.46 * semantic + 0.19 * features["operation_match"] + 0.15)
    if intent.metric_terms and entry.metric_score == 0.0:
        confidence *= 0.42
    if intent.operation in DERIVATION_OPERATIONS:
        # Direct values remain legitimate for questions whose report already states the
        # requested ratio, margin or change. Over-penalising them can prune correct
        # direct percentages from the final lattice.
        confidence *= 0.76
    answer = _candidate_answer(value, answer_type)
    proof = CandidateProof("direct", "lookup", (item.evidence_id,), item.evidence_id, True, True)
    return AnswerCandidate(
        candidate_id=_stable_id("direct", f"{item.evidence_id}|{answer}"), answer=answer,
        value=value, answer_type=answer_type, proof=proof, evidence_ids=(item.evidence_id,),
        retrieval_score=entry.score, semantic_score=semantic, proof_score=1.0,
        confidence=confidence, explanation=f"Direct value from {item.evidence_id}: {item.text}",
        feature_values=features,
    )


def _role_fit(operation: str, operands: list[RetrievedEvidence], intent: QuestionIntent) -> float:
    if not operands:
        return 0.0
    if operation in {"subtraction", "aggregate_difference", "percentage_change"}:
        return 1.0 if len(operands) >= 2 and _same_metric(operands[0].item, operands[1].item) else 0.45
    if operation in {"addition", "average", "table_sum", "table_average", "table_min", "table_max"}:
        same_row = len({_normalised_metric(entry.item) for entry in operands}) == 1
        entity_hits = sum(_term_overlap(intent.entity_terms, entry.item) > 0 for entry in operands)
        return max(float(same_row), min(1.0, entity_hits / max(1, len(operands))))
    if operation in {"division", "margin", "percentage_of_total"} and len(operands) >= 2:
        num = _term_overlap(intent.numerator_terms, operands[0].item)
        den = _term_overlap(intent.denominator_terms, operands[1].item)
        if intent.numerator_terms or intent.denominator_terms:
            return (num + den) / 2.0
        if operation == "margin":
            left = _normalised_metric(operands[0].item)
            right = _normalised_metric(operands[1].item)
            n = any(token in left for token in ("income", "profit", "earnings", "component", "segment"))
            d = any(token in right for token in ("revenue", "sales", "total", "obligation", "commitment"))
            return 1.0 if n and d else 0.45
        return 0.75
    if operation == "multiplication":
        return 1.0 if any(entry.item.is_percent for entry in operands) else 0.70
    return 0.65


def _period_fit(operands: list[RetrievedEvidence], intent: QuestionIntent) -> float:
    if not intent.periods:
        return 1.0
    represented = {entry.item.period for entry in operands if entry.item.period}
    return len(set(intent.periods) & represented) / max(1, len(intent.periods))


def _derived_candidate(
    operation: str,
    operands: list[RetrievedEvidence],
    result: Decimal,
    intent: QuestionIntent,
    expression: str,
    *,
    depth: int = 1,
    answer_type: Optional[str] = None,
) -> Optional[AnswerCandidate]:
    if not result.is_finite() or abs(result) > Decimal("1e18"):
        return None
    compatible = _compatible_units((entry.item for entry in operands), operation)
    if not compatible:
        return None
    role_fit = _role_fit(operation, operands, intent)
    period_fit = _period_fit(operands, intent)
    features = _feature_values(
        operation=operation, operands=operands, intent=intent, compatible=compatible,
        depth=depth, role_fit=role_fit, period_fit=period_fit,
    )
    semantic = _clamp(
        0.28 * features["metric_fit"] + 0.22 * period_fit + 0.30 * role_fit
        + 0.20 * features["operation_match"]
    )
    confidence = _base_confidence(features)
    if role_fit == 0.0:
        confidence *= 0.25
    answer_type = answer_type or _answer_type_for(operation, intent, (entry.item for entry in operands))
    answer = _candidate_answer(result, answer_type)
    ids = tuple(entry.item.evidence_id for entry in operands)
    proof = CandidateProof("derived", operation, ids, expression, compatible, True)
    return AnswerCandidate(
        candidate_id=_stable_id("derived", f"{operation}|{'|'.join(ids)}|{answer}|{expression}"),
        answer=answer, value=result, answer_type=answer_type, proof=proof, evidence_ids=ids,
        retrieval_score=sum(entry.score for entry in operands) / max(1, len(operands)),
        semantic_score=semantic, proof_score=1.0, confidence=confidence,
        explanation=f"{operation}: {expression}", feature_values=features,
    )


SUMMARY_LABEL_RE = re.compile(
    r"\b(?:grand\s+total|total|subtotal|average|mean|maximum|max|minimum|min|highest|lowest)\b",
    flags=re.IGNORECASE,
)


def _row_groups(retrieved: list[RetrievedEvidence]) -> list[list[RetrievedEvidence]]:
    groups: dict[tuple[int | None, str], list[RetrievedEvidence]] = {}
    for entry in retrieved:
        if entry.item.source_type != "table" or entry.item.is_year:
            continue
        key = (entry.item.row_index, _normalised_metric(entry.item))
        groups.setdefault(key, []).append(entry)
    return [sorted(group, key=lambda e: (e.item.column_index or 0, e.item.evidence_id)) for group in groups.values()]


def _column_groups(retrieved: list[RetrievedEvidence]) -> list[list[RetrievedEvidence]]:
    """Group numeric cells by table column.

    FinQA's table operators can target either a row label or a column label. Aggregating
    only by row makes column-oriented tables invisible to table_sum,
    table_average, table_min and table_max.
    """
    groups: dict[tuple[int | None, str], list[RetrievedEvidence]] = {}
    for entry in retrieved:
        item = entry.item
        if item.source_type != "table" or item.is_year:
            continue
        label = " ".join((item.column_label or "").strip().lower().split())
        if not label:
            continue
        key = (item.column_index, label)
        groups.setdefault(key, []).append(entry)
    return [
        sorted(group, key=lambda e: (e.item.row_index or 0, e.item.evidence_id))
        for group in groups.values()
    ]


def _table_aggregate_groups(
    retrieved: list[RetrievedEvidence],
) -> list[tuple[str, list[RetrievedEvidence]]]:
    groups: list[tuple[str, list[RetrievedEvidence]]] = []
    groups.extend(("row", group) for group in _row_groups(retrieved))
    groups.extend(("column", group) for group in _column_groups(retrieved))
    return groups


def _axis_summary_label(entry: RetrievedEvidence, axis: str) -> str:
    return (
        entry.item.column_label or ""
        if axis == "row"
        else entry.item.row_label or ""
    )


def _requested_group_values(
    group: list[RetrievedEvidence], intent: QuestionIntent, axis: str
) -> list[RetrievedEvidence]:
    requested = [entry for entry in group if entry.item.period in intent.periods]
    values = requested or group
    # Do not double-count a pre-computed Total/Average/Maximum cell when the table
    # operator asks us to calculate the aggregate from the atomic cells. A direct
    # lookup of that summary cell remains available as a separate direct candidate.
    atomic = [
        entry for entry in values
        if not SUMMARY_LABEL_RE.search(_axis_summary_label(entry, axis))
    ]
    if len(atomic) >= 2:
        values = atomic
    if intent.period_count_hint and len(values) > intent.period_count_hint:
        values = sorted(values, key=lambda e: (-e.score, e.item.evidence_id))[: intent.period_count_hint]
    return values


def _aggregate_set_key(values: Iterable[RetrievedEvidence]) -> tuple[str, ...]:
    return tuple(sorted(entry.item.evidence_id for entry in values))


def _axis_order(entry: RetrievedEvidence, axis: str) -> tuple[int, int, str]:
    period = entry.item.period or ""
    period_value = int(period) if period.isdigit() else 999999
    position = (
        entry.item.column_index if axis == "row" else entry.item.row_index
    )
    return (period_value, position if position is not None else 999999, entry.item.evidence_id)


def _plausible_aggregate_value_sets(
    group: list[RetrievedEvidence], intent: QuestionIntent, axis: str
) -> list[list[RetrievedEvidence]]:
    """Return bounded question-conditioned subsets for table aggregation.

    FinQA often asks for an aggregate over a three- or five-period window while the
    table exposes a longer row or column. Selecting only one top-scoring subset can miss
    the correct window even when every operand is present. The implementation retains the
    primary deterministic subset, then adds contiguous and bounded combinatorial
    windows licensed by the question's period-count hint. Gold answers/programs are
    never consulted.
    """
    non_year = [entry for entry in group if not entry.item.is_year]
    atomic = [entry for entry in non_year if not SUMMARY_LABEL_RE.search(_axis_summary_label(entry, axis))]
    if len(atomic) < 2:
        atomic = non_year

    value_sets: list[list[RetrievedEvidence]] = []
    seen: set[tuple[str, ...]] = set()

    def add(values: Iterable[RetrievedEvidence]) -> None:
        materialised = list(values)
        if not materialised:
            return
        key = _aggregate_set_key(materialised)
        if key in seen:
            return
        seen.add(key)
        value_sets.append(materialised)

    add(_requested_group_values(group, intent, axis))

    # Explicit requested periods receive one highest-scoring cell per period.
    if intent.periods:
        by_period: list[RetrievedEvidence] = []
        for period in intent.ordered_periods or intent.periods:
            matches = [entry for entry in atomic if entry.item.period == period]
            if matches:
                by_period.append(max(matches, key=lambda entry: (entry.score, entry.item.evidence_id)))
        if len(by_period) >= 2:
            add(by_period)

    ordered = sorted(atomic, key=lambda entry: _axis_order(entry, axis))
    count = intent.period_count_hint
    if count and 2 <= count <= 6 and len(ordered) >= count:
        # Every contiguous table window is cheap and preserves table order.
        for start in range(0, len(ordered) - count + 1):
            add(ordered[start : start + count])

        # When the group is small, enumerate all count-sized subsets. This is bounded
        # and recovers irregular layouts where a summary/spacer column breaks contiguity.
        if len(ordered) <= 10:
            for index, values in enumerate(itertools.combinations(ordered, count)):
                if index >= 120:
                    break
                add(values)

        add(sorted(ordered, key=lambda entry: (-entry.score, entry.item.evidence_id))[:count])
    elif 2 <= len(ordered) <= 8:
        add(ordered)

    return value_sets[:128]


def _semantic_aggregate_groups(
    retrieved: list[RetrievedEvidence], intent: QuestionIntent
) -> list[tuple[str, list[RetrievedEvidence]]]:
    """Create bounded cross-axis pools from metric/entity matches.

    Some FinQA tables use multi-row headers or sparse labels that prevent a correct
    aggregate from appearing in a single structural row/column group. This fallback
    groups only cells that match the question's metric/entity terms, and is therefore
    still question-conditioned rather than an unrestricted arithmetic closure.
    """
    table_entries = [
        entry for entry in retrieved
        if entry.item.source_type == "table" and not entry.item.is_year
    ]
    if len(table_entries) < 2:
        return []

    scored: list[tuple[float, RetrievedEvidence]] = []
    for entry in table_entries:
        metric = _term_overlap(intent.metric_terms, entry.item)
        entity = _term_overlap(intent.entity_terms, entry.item)
        role = max(
            _term_overlap(intent.numerator_terms, entry.item),
            _term_overlap(intent.denominator_terms, entry.item),
        )
        match = max(metric, entity, role)
        if match > 0.0:
            scored.append((match, entry))
    if len(scored) < 2:
        return []

    scored.sort(key=lambda pair: (-pair[0], -pair[1].score, pair[1].item.evidence_id))
    best_match = scored[0][0]
    candidates = [entry for match, entry in scored if match >= max(0.34, best_match * 0.60)]

    groups: list[tuple[str, list[RetrievedEvidence]]] = []
    if intent.periods:
        selected: list[RetrievedEvidence] = []
        for period in intent.ordered_periods or intent.periods:
            period_entries = [entry for entry in candidates if entry.item.period == period]
            if period_entries:
                selected.append(max(period_entries, key=lambda entry: (entry.score, entry.item.evidence_id)))
        if len(selected) >= 2:
            groups.append(("semantic_period", selected))

    hint = intent.period_count_hint
    if hint and len(candidates) >= hint:
        groups.append(("semantic_hint", candidates[: min(len(candidates), max(hint, hint * 2))]))
    elif 2 <= len(candidates) <= 10:
        groups.append(("semantic_metric", candidates))
    return groups[:3]


def _summary_axis_label(entry: RetrievedEvidence) -> bool:
    return bool(
        SUMMARY_LABEL_RE.search(entry.item.row_label or "")
        or SUMMARY_LABEL_RE.search(entry.item.column_label or "")
    )


def _combination_double_counts_summary(operands: Iterable[RetrievedEvidence]) -> bool:
    """Detect atomic cells combined with a pre-computed summary from the same axis.

    A row such as ``100, 200, 300, Total=600`` must never yield a certified
    ``1200`` proof by adding all four cells. Summary values can still be returned as
    direct candidates and can still participate in operations across independent rows.
    """
    values = list(operands)
    for summary in values:
        if not _summary_axis_label(summary):
            continue
        for other in values:
            if other.item.evidence_id == summary.item.evidence_id:
                continue
            same_row = (
                summary.item.row_index is not None
                and summary.item.row_index == other.item.row_index
            )
            same_column = (
                summary.item.column_index is not None
                and summary.item.column_index == other.item.column_index
            )
            if same_row or same_column:
                return True
    return False


def _aggregate_candidates(retrieved: list[RetrievedEvidence], intent: QuestionIntent) -> list[AnswerCandidate]:
    output: list[AnswerCandidate] = []
    aggregate_ops = [
        operation for operation in _operation_portfolio(intent)
        if operation in {"sum", "average", "table_sum", "table_average", "table_min", "table_max"}
    ]
    if not aggregate_ops:
        return output

    structural = _table_aggregate_groups(retrieved)
    semantic = _semantic_aggregate_groups(retrieved, intent)
    for axis, group in [*structural, *semantic]:
        value_sets = (
            _plausible_aggregate_value_sets(group, intent, axis)
            if axis in {"row", "column"}
            else _plausible_aggregate_value_sets(group, intent, "row")
        )
        for values in value_sets:
            for operation in aggregate_ops:
                minimum = 1 if operation in {"table_min", "table_max"} else 2
                if len(values) < minimum:
                    continue
                scale = _common_output_scale(intent, (entry.item for entry in values))
                numbers = [_value_in_scale(entry.item, scale) for entry in values]
                if operation in {"sum", "table_sum"}:
                    result = sum(numbers, Decimal("0"))
                    op = "table_sum" if len(values) > 2 else "addition"
                    expression = " + ".join(entry.item.evidence_id for entry in values)
                elif operation in {"average", "table_average"}:
                    result = sum(numbers, Decimal("0")) / Decimal(len(numbers))
                    op = "table_average" if len(values) > 2 else "average"
                    expression = f"({' + '.join(entry.item.evidence_id for entry in values)}) / {len(values)}"
                elif operation == "table_min":
                    result, op = min(numbers), "table_min"
                    expression = f"min({','.join(entry.item.evidence_id for entry in values)})"
                else:
                    result, op = max(numbers), "table_max"
                    expression = f"max({','.join(entry.item.evidence_id for entry in values)})"
                candidate = _derived_candidate(
                    op, values, result, intent, expression, depth=max(1, len(values) - 1)
                )
                if candidate:
                    output.append(candidate)
                if len(output) >= 900:
                    return output
    return output

def _aggregate_composite_candidates(
    aggregate_candidates: list[AnswerCandidate],
    retrieved: list[RetrievedEvidence],
    intent: QuestionIntent,
) -> list[AnswerCandidate]:
    """Compose question-licensed aggregate summaries.

    FinQA contains questions such as "the variation between the average and the
    highest operating margin". Producing the average and maximum separately is not
    enough; the certified lattice must also contain max-average. The composition is
    restricted to aggregates computed from the exact same evidence set.
    """
    portfolio = set(_operation_portfolio(intent))
    if "difference" not in portfolio:
        return []

    by_group: dict[tuple[str, ...], dict[str, AnswerCandidate]] = {}
    for candidate in aggregate_candidates:
        if candidate.value is None:
            continue
        key = tuple(sorted(candidate.evidence_ids))
        by_group.setdefault(key, {})[candidate.proof.operation] = candidate

    entry_by_id = {entry.item.evidence_id: entry for entry in retrieved}
    output: list[AnswerCandidate] = []
    for evidence_ids, operations in by_group.items():
        operands = [entry_by_id[evidence_id] for evidence_id in evidence_ids if evidence_id in entry_by_id]
        if not operands:
            continue

        pairs: list[tuple[AnswerCandidate, AnswerCandidate, str]] = []
        maximum = operations.get("table_max")
        minimum = operations.get("table_min")
        average = operations.get("table_average") or operations.get("average")
        if maximum is not None and average is not None:
            pairs.append((maximum, average, "max_minus_average"))
        if average is not None and minimum is not None:
            pairs.append((average, minimum, "average_minus_min"))
        if maximum is not None and minimum is not None:
            pairs.append((maximum, minimum, "max_minus_min"))

        for left, right, label in pairs:
            result = left.value - right.value
            expression = f"({left.proof.expression}) - ({right.proof.expression})"
            candidate = _derived_candidate(
                "aggregate_difference",
                operands,
                result,
                intent,
                expression,
                depth=max(2, len(operands)),
                answer_type=left.answer_type if left.answer_type == right.answer_type else None,
            )
            if candidate is not None:
                candidate = replace(
                    candidate,
                    explanation=f"{label}: {expression}",
                )
                output.append(candidate)
    return output

def _ordered_period_pairs(retrieved: list[RetrievedEvidence], intent: QuestionIntent) -> list[tuple[RetrievedEvidence, RetrievedEvidence]]:
    if len(intent.ordered_periods) < 2:
        return []
    earlier, later = intent.ordered_periods[:2]
    old_items = [e for e in retrieved if e.item.period == earlier and not e.item.is_year]
    new_items = [e for e in retrieved if e.item.period == later and not e.item.is_year]
    same = [(old, new) for old in old_items for new in new_items if _same_metric(old.item, new.item)]
    return same or [(old, new) for old in old_items for new in new_items]


def _top_numeric(retrieved: list[RetrievedEvidence], limit: int = 12) -> list[RetrievedEvidence]:
    return [entry for entry in retrieved if not entry.item.is_year][:limit]


def _pair_candidates(retrieved: list[RetrievedEvidence], intent: QuestionIntent) -> list[AnswerCandidate]:
    output: list[AnswerCandidate] = []
    operations = [
        operation for operation in _operation_portfolio(intent)
        if operation in {"difference", "percentage_change", "ratio", "margin", "sum", "average", "product"}
    ]
    if not operations:
        return output

    period_pairs = _ordered_period_pairs(retrieved, intent)
    top = _top_numeric(retrieved, limit=16)

    for operation in operations:
        if operation in {"difference", "percentage_change"} and period_pairs:
            pairs = period_pairs
        elif operation in {"sum", "average"}:
            pairs = list(itertools.combinations(top, 2))
        else:
            pairs = [
                (a, b) for a in top for b in top
                if a.item.evidence_id != b.item.evidence_id
            ]

        emitted = 0
        for first, second in pairs:
            if emitted >= 220:
                break
            if operation in {"sum", "average"} and _combination_double_counts_summary((first, second)):
                continue
            a_item, b_item = first.item, second.item
            candidate: Optional[AnswerCandidate]
            if operation == "difference":
                scale = _common_output_scale(intent, (a_item, b_item))
                a, b = _value_in_scale(a_item, scale), _value_in_scale(b_item, scale)
                if period_pairs:
                    result, operands = b - a, [second, first]
                    expression = f"{b_item.evidence_id} - {a_item.evidence_id}"
                else:
                    result, operands = a - b, [first, second]
                    expression = f"{a_item.evidence_id} - {b_item.evidence_id}"
                if intent.asks_for_absolute_change:
                    result, expression = abs(result), f"abs({expression})"
                candidate = _derived_candidate("subtraction", operands, result, intent, expression)
            elif operation == "percentage_change":
                old, new = (first, second) if period_pairs else (second, first)
                ratio = _safe_divide(new.item.base_value - old.item.base_value, old.item.base_value)
                candidate = None if ratio is None else _derived_candidate(
                    "percentage_change", [new, old], ratio * Decimal("100"), intent,
                    f"(({new.item.evidence_id} - {old.item.evidence_id}) / {old.item.evidence_id}) * 100",
                    depth=3, answer_type="percentage",
                )
            elif operation in {"ratio", "margin"}:
                ratio = _safe_divide(a_item.base_value, b_item.base_value)
                if ratio is None:
                    candidate = None
                else:
                    percent = intent.expected_answer_type == "percentage" or operation == "margin"
                    result = ratio * Decimal("100") if percent else ratio
                    op = "margin" if operation == "margin" else "division"
                    expression = f"({a_item.evidence_id} / {b_item.evidence_id})" + (" * 100" if percent else "")
                    candidate = _derived_candidate(
                        op, [first, second], result, intent, expression,
                        depth=2 if percent else 1,
                        answer_type="percentage" if percent else None,
                    )
            elif operation == "sum":
                scale = _common_output_scale(intent, (a_item, b_item))
                candidate = _derived_candidate(
                    "addition", [first, second],
                    _value_in_scale(a_item, scale) + _value_in_scale(b_item, scale),
                    intent, f"{a_item.evidence_id} + {b_item.evidence_id}",
                )
            elif operation == "average":
                scale = _common_output_scale(intent, (a_item, b_item))
                candidate = _derived_candidate(
                    "average", [first, second],
                    (_value_in_scale(a_item, scale) + _value_in_scale(b_item, scale)) / Decimal("2"),
                    intent, f"({a_item.evidence_id} + {b_item.evidence_id}) / 2", depth=2,
                )
            else:  # product
                if a_item.is_percent and not b_item.is_percent:
                    result = (a_item.value / Decimal("100")) * _value_in_scale(b_item, intent.target_scale or b_item.scale)
                elif b_item.is_percent and not a_item.is_percent:
                    result = (b_item.value / Decimal("100")) * _value_in_scale(a_item, intent.target_scale or a_item.scale)
                else:
                    result = a_item.value * b_item.value
                candidate = _derived_candidate(
                    "multiplication", [first, second], result, intent,
                    f"{a_item.evidence_id} * {b_item.evidence_id}",
                )
            if candidate:
                output.append(candidate)
                emitted += 1
    return output

def _constant_candidates(retrieved: list[RetrievedEvidence], intent: QuestionIntent) -> list[AnswerCandidate]:
    output: list[AnswerCandidate] = []
    operations = set(_operation_portfolio(intent))
    constants = list(intent.explicit_constants)
    if "indexed_return" in operations and Decimal("100") not in constants:
        constants.append(Decimal("100"))
    if not constants:
        return output

    top = _top_numeric(retrieved, limit=12)
    for entry in top:
        for constant in constants:
            if constant == 0:
                continue
            pseudo_expression = f"const_{constant}"
            candidate: Optional[AnswerCandidate] = None
            if "indexed_return" in operations and constant == Decimal("100"):
                candidate = _derived_candidate(
                    "indexed_return", [entry], entry.item.value - constant, intent,
                    f"{entry.item.evidence_id} - {pseudo_expression}",
                    depth=1, answer_type="percentage",
                )
            elif "average" in operations:
                candidate = _derived_candidate(
                    "average", [entry], entry.item.value / constant, intent,
                    f"{entry.item.evidence_id} / {pseudo_expression}", depth=1,
                )
            elif "ratio" in operations:
                candidate = _derived_candidate(
                    "division", [entry], entry.item.value / constant, intent,
                    f"{entry.item.evidence_id} / {pseudo_expression}", depth=1,
                )
            elif "product" in operations:
                candidate = _derived_candidate(
                    "multiplication", [entry], entry.item.value * constant, intent,
                    f"{entry.item.evidence_id} * {pseudo_expression}", depth=1,
                )
            if candidate:
                output.append(candidate)
    return output

def _nary_cross_row_candidates(retrieved: list[RetrievedEvidence], intent: QuestionIntent) -> list[AnswerCandidate]:
    operations = [operation for operation in _operation_portfolio(intent) if operation in {"sum", "average"}]
    if not operations:
        return []
    top = _top_numeric(retrieved, 14)
    if intent.periods:
        matching = [entry for entry in top if entry.item.period in intent.periods]
        if matching:
            top = matching
    output: list[AnswerCandidate] = []
    max_size = min(6, intent.period_count_hint or 5, len(top))
    for size in range(3, max_size + 1):
        for operands in itertools.combinations(top, size):
            if _combination_double_counts_summary(operands):
                continue
            row_names = {_normalised_metric(entry.item) for entry in operands}
            entity_hits = sum(_term_overlap(intent.entity_terms, entry.item) > 0 for entry in operands)
            if len(row_names) > 1 and intent.entity_terms and entity_hits < min(2, size):
                continue
            scale = _common_output_scale(intent, (entry.item for entry in operands))
            values = [_value_in_scale(entry.item, scale) for entry in operands]
            for operation in operations:
                if operation == "sum":
                    result, op = sum(values, Decimal("0")), "addition"
                    expression = " + ".join(entry.item.evidence_id for entry in operands)
                else:
                    result, op = sum(values, Decimal("0")) / Decimal(size), "average"
                    expression = f"({' + '.join(entry.item.evidence_id for entry in operands)}) / {size}"
                candidate = _derived_candidate(op, list(operands), result, intent, expression, depth=size)
                if candidate:
                    output.append(candidate)
            if len(output) >= 160:
                return output
    return output


def _composed_ratio_candidates(
    retrieved: list[RetrievedEvidence], intent: QuestionIntent
) -> list[AnswerCandidate]:
    """Generate a small, question-licensed set of two-step arithmetic proofs.

    Many missed references use programs such as
    ``subtract -> divide`` or ``add -> divide``. This routine does not enumerate an
    unrestricted arithmetic closure. It is enabled only for ratio/margin/change/
    average questions, uses the highest-ranked eight evidence values, and stops at a
    fixed candidate budget.
    """
    portfolio = set(_operation_portfolio(intent))
    if not portfolio & {"ratio", "margin", "percentage_change", "average", "difference"}:
        return []

    top = _top_numeric(retrieved, limit=8)
    if len(top) < 2:
        return []

    output: list[AnswerCandidate] = []
    budget = 220

    def add_candidate(
        operation: str,
        operands: list[RetrievedEvidence],
        result: Optional[Decimal],
        expression: str,
        *,
        depth: int,
        answer_type: Optional[str] = None,
    ) -> None:
        if result is None or len(output) >= budget:
            return
        candidate = _derived_candidate(
            operation,
            operands,
            result,
            intent,
            expression,
            depth=depth,
            answer_type=answer_type,
        )
        if candidate is not None:
            output.append(candidate)

    # Relative-difference variants: (a-b)/b and (a-b)/a. These are restricted to
    # change/ratio families and preserve percent semantics when the question asks for it.
    if portfolio & {"percentage_change", "ratio", "margin", "difference"}:
        for first, second in itertools.permutations(top, 2):
            if len(output) >= budget:
                break
            numerator = first.item.base_value - second.item.base_value
            for denominator, denominator_entry in (
                (second.item.base_value, second),
                (first.item.base_value, first),
            ):
                ratio = _safe_divide(numerator, denominator)
                if ratio is None:
                    continue
                percentage = intent.expected_answer_type == "percentage" or "percentage_change" in portfolio or "margin" in portfolio
                result = ratio * Decimal("100") if percentage else ratio
                operation = "percentage_change" if percentage else "division"
                expression = (
                    f"(({first.item.evidence_id} - {second.item.evidence_id}) / "
                    f"{denominator_entry.item.evidence_id})"
                    + (" * 100" if percentage else "")
                )
                add_candidate(
                    operation,
                    [first, second, denominator_entry],
                    result,
                    expression,
                    depth=3 if percentage else 2,
                    answer_type="percentage" if percentage else None,
                )

    # Three-operand compositions. Keep only forms observed in FinQA's disclosed
    # program grammar: (a+b)/c, (a-b)/c, a/(b+c), and a/(b-c).
    if portfolio & {"ratio", "margin", "average", "percentage_change"}:
        for a, b, c in itertools.permutations(top, 3):
            if len(output) >= budget:
                break
            forms = (
                (a.item.base_value + b.item.base_value, c.item.base_value,
                 f"({a.item.evidence_id} + {b.item.evidence_id}) / {c.item.evidence_id}"),
                (a.item.base_value - b.item.base_value, c.item.base_value,
                 f"({a.item.evidence_id} - {b.item.evidence_id}) / {c.item.evidence_id}"),
                (a.item.base_value, b.item.base_value + c.item.base_value,
                 f"{a.item.evidence_id} / ({b.item.evidence_id} + {c.item.evidence_id})"),
                (a.item.base_value, b.item.base_value - c.item.base_value,
                 f"{a.item.evidence_id} / ({b.item.evidence_id} - {c.item.evidence_id})"),
            )
            for numerator, denominator, expression in forms:
                ratio = _safe_divide(numerator, denominator)
                if ratio is None:
                    continue
                percentage = intent.expected_answer_type == "percentage" or "margin" in portfolio or "percentage_change" in portfolio
                result = ratio * Decimal("100") if percentage else ratio
                add_candidate(
                    "margin" if percentage else "division",
                    [a, b, c],
                    result,
                    expression + (" * 100" if percentage else ""),
                    depth=3 if percentage else 2,
                    answer_type="percentage" if percentage else None,
                )
                if len(output) >= budget:
                    break
    return output

def _deduplicate(candidates: Iterable[AnswerCandidate]) -> list[AnswerCandidate]:
    # Keep the strongest proof for each complete answer. Proof type is intentionally not
    # part of the key: the verifier should return the best certificate for that answer.
    best: dict[str, AnswerCandidate] = {}
    for candidate in candidates:
        previous = best.get(candidate.answer)
        specific_operations = {
            "aggregate_difference", "table_sum", "table_average",
            "table_min", "table_max", "percentage_change",
            "percentage_of_total", "indexed_return",
        }
        candidate_rank = (
            candidate.feature_values.get("operation_match", 0.0),
            float(candidate.proof.operation in specific_operations),
            candidate.proof_score,
            candidate.confidence,
            candidate.semantic_score,
            len(candidate.evidence_ids),
        )
        previous_rank = None if previous is None else (
            previous.feature_values.get("operation_match", 0.0),
            float(previous.proof.operation in specific_operations),
            previous.proof_score,
            previous.confidence,
            previous.semantic_score,
            len(previous.evidence_ids),
        )
        if previous is None or candidate_rank > previous_rank:
            best[candidate.answer] = candidate
    return sorted(best.values(), key=lambda candidate: (-candidate.confidence, candidate.candidate_id))


def _diverse_prune(candidates: Iterable[AnswerCandidate], limit: int) -> list[AnswerCandidate]:
    """Prune without allowing one prolific operation family to consume the lattice.

    Generating many ratio or percentage candidates can remove the sole table-average or
    indexed-return answer. A small round-robin allocation is therefore reserved per proof
    operation before remaining slots are filled by global score.
    """
    ranked = _deduplicate(candidates)
    if len(ranked) <= limit:
        return ranked
    groups: dict[str, list[AnswerCandidate]] = {}
    for candidate in ranked:
        groups.setdefault(candidate.proof.operation, []).append(candidate)
    selected: list[AnswerCandidate] = []
    seen: set[str] = set()
    reserve = max(2, min(8, limit // max(1, len(groups))))
    for index in range(reserve):
        for operation in sorted(groups):
            group = groups[operation]
            if index < len(group) and len(selected) < limit:
                candidate = group[index]
                if candidate.candidate_id not in seen:
                    selected.append(candidate)
                    seen.add(candidate.candidate_id)
    for candidate in ranked:
        if len(selected) >= limit:
            break
        if candidate.candidate_id not in seen:
            selected.append(candidate)
            seen.add(candidate.candidate_id)
    return sorted(selected, key=lambda candidate: (-candidate.confidence, candidate.candidate_id))


def build_candidate_lattice(
    retrieved: Iterable[RetrievedEvidence],
    intent: QuestionIntent,
    *,
    include_derived: bool = True,
    max_direct: int = 48,
    max_derived: int = 192,
    minimum_direct_confidence: float = 0.02,
    minimum_derived_confidence: float = 0.08,
    candidate_scorer: Optional[CandidateScorer] = None,
) -> CandidateLattice:
    retrieved_list = list(retrieved)
    direct_all = _deduplicate([
        _direct_candidate(entry, intent)
        for entry in retrieved_list
        if not entry.item.is_year or intent.expected_answer_type == "year"
    ])

    derived_all: list[AnswerCandidate] = []
    if include_derived:
        aggregate_candidates = _aggregate_candidates(retrieved_list, intent)
        derived_all.extend(aggregate_candidates)
        derived_all.extend(
            _aggregate_composite_candidates(
                aggregate_candidates, retrieved_list, intent
            )
        )
        derived_all.extend(_pair_candidates(retrieved_list, intent))
        derived_all.extend(_nary_cross_row_candidates(retrieved_list, intent))
        derived_all.extend(_constant_candidates(retrieved_list, intent))
        derived_all.extend(_composed_ratio_candidates(retrieved_list, intent))
        derived_all = _deduplicate(derived_all)

    # Learned rescoring is applied before every deployment pruning decision, including the
    # heuristic confidence floor. With a selector, the frozen lattice is bounded only after
    # learned rescoring. Without a selector, the disclosed heuristic floors remain active.
    if candidate_scorer is not None:
        direct_all = [
            replace(candidate, confidence=_clamp(float(candidate_scorer(candidate))))
            for candidate in direct_all
        ]
        derived_all = [
            replace(candidate, confidence=_clamp(float(candidate_scorer(candidate))))
            for candidate in derived_all
        ]
    else:
        direct_all = [
            candidate for candidate in direct_all
            if candidate.confidence >= minimum_direct_confidence
        ]
        derived_all = [
            candidate for candidate in derived_all
            if candidate.confidence >= minimum_derived_confidence
        ]

    direct = sorted(direct_all, key=lambda candidate: (-candidate.confidence, candidate.candidate_id))[:max_direct]
    derived = _diverse_prune(derived_all, max_derived)
    numeric = _deduplicate([*direct, *derived])

    abstain = AnswerCandidate(
        candidate_id="abstain", answer=ABSTAIN_RESPONSE, value=None, answer_type="abstain",
        proof=CandidateProof("abstain", "abstain", (), "", True, True), evidence_ids=(),
        retrieval_score=0.0, semantic_score=1.0, proof_score=1.0, confidence=0.0,
        explanation="Explicit abstention when no certified answer is sufficiently supported.",
        feature_values={"direct": 0.0},
    )
    return CandidateLattice(
        candidates=tuple([*numeric, abstain]),
        direct_count=sum(candidate.proof.proof_type == "direct" for candidate in numeric),
        derived_count=sum(candidate.proof.proof_type == "derived" for candidate in numeric),
        retrieval_count=len(retrieved_list),
    )
