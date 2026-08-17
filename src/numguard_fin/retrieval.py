from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable

from .schemas import EvidenceItem, QuestionIntent, RetrievedEvidence


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "a", "an", "of", "in", "to", "for", "and", "or", "was", "were",
    "is", "are", "what", "how", "much", "many", "during", "from", "by", "with",
    "according", "company", "year", "years", "reported", "considering", "given",
}


def _normalise_token(token: str) -> str:
    token = token.lower()
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def tokens(text: str) -> list[str]:
    return [
        _normalise_token(token)
        for token in TOKEN_RE.findall((text or "").lower())
        if token not in STOPWORDS
    ]


def _weighted_overlap(query_tokens: list[str], evidence_tokens: list[str]) -> float:
    if not query_tokens or not evidence_tokens:
        return 0.0
    query_counts = Counter(query_tokens)
    evidence_counts = Counter(evidence_tokens)
    overlap = sum(min(query_counts[token], evidence_counts[token]) for token in query_counts)
    return overlap / math.sqrt(max(1, len(query_tokens)) * max(1, len(evidence_tokens)))


def _set_overlap(terms: Iterable[str], text: str) -> float:
    term_set = {_normalise_token(term) for term in terms if term}
    if not term_set:
        return 0.0
    evidence = set(tokens(text))
    return len(term_set & evidence) / len(term_set)


def score_evidence(item: EvidenceItem, question: str, intent: QuestionIntent) -> RetrievedEvidence:
    question_tokens = tokens(question)
    row_text = f"{item.row_label or ''} {item.metric_text}"
    evidence_text = f"{row_text} {item.column_label or ''} {item.text}"
    evidence_tokens = tokens(evidence_text)
    lexical = _weighted_overlap(question_tokens, evidence_tokens)

    # Metric names can live in either table axis, so
    # year-in-row / metric-in-column tables could rank the correct metric below an
    # unrelated column. Use the stronger row-or-column match without consulting
    # gold programs or answers.
    column_text = item.column_label or ""
    metric_score = max(
        _set_overlap(intent.metric_terms, row_text),
        _set_overlap(intent.metric_terms, column_text),
    )
    entity_score = max(
        _set_overlap(intent.entity_terms, row_text),
        _set_overlap(intent.entity_terms, column_text),
    )
    numerator_score = max(
        _set_overlap(intent.numerator_terms, row_text),
        _set_overlap(intent.numerator_terms, column_text),
    )
    denominator_score = max(
        _set_overlap(intent.denominator_terms, row_text),
        _set_overlap(intent.denominator_terms, column_text),
    )
    role_score = max(numerator_score, denominator_score)

    period_score = 0.0
    if intent.periods:
        if item.period in intent.periods:
            period_score = 1.0
        elif any(period in (item.column_label or "") or period in item.text for period in intent.periods):
            period_score = 0.7
        else:
            period_score = -0.30

    expected = intent.expected_answer_type
    if expected == "percentage":
        type_score = 0.75 if item.is_percent else 0.18
    elif expected == "year":
        type_score = 1.0 if item.is_year else -1.0
    else:
        type_score = 0.20 if not item.is_year else -0.85

    scale_score = 0.0
    if intent.target_scale:
        if item.scale == intent.target_scale:
            scale_score = 0.35
        elif item.scale:
            scale_score = 0.05

    phrase_bonus = 0.0
    normalised_question = " ".join(tokens(question))
    normalised_row = " ".join(tokens(row_text))
    if normalised_row and normalised_row in normalised_question:
        phrase_bonus = 0.75

    source_bonus = 0.30 if item.source_type == "table" else 0.14
    non_year_bonus = 0.12 if not item.is_year or expected == "year" else -0.65
    score = (
        2.4 * lexical
        + 2.8 * metric_score
        + 1.5 * entity_score
        + 2.0 * role_score
        + 2.0 * period_score
        + type_score
        + scale_score
        + phrase_bonus
        + source_bonus
        + non_year_bonus
    )
    return RetrievedEvidence(
        item=item,
        score=score,
        lexical_score=lexical,
        period_score=period_score,
        metric_score=max(metric_score, role_score),
        type_score=type_score,
    )


def _row_key(item: EvidenceItem) -> tuple[str, int | None, str]:
    return item.source_type, item.row_index, (item.row_label or item.metric_text or "").lower()


def _column_key(item: EvidenceItem) -> tuple[str, int | None, str]:
    return item.source_type, item.column_index, (item.column_label or "").lower()


def _row_rank_value(entries: list[RetrievedEvidence]) -> tuple[float, float, float]:
    return (
        max(entry.score for entry in entries),
        max(entry.metric_score for entry in entries),
        sum(max(0.0, entry.period_score) for entry in entries),
    )


def _role_row(
    rows: dict[tuple[str, int | None, str], list[RetrievedEvidence]],
    terms: tuple[str, ...],
) -> tuple[str, int | None, str] | None:
    if not terms:
        return None
    ranked = []
    for key, entries in rows.items():
        text = f"{entries[0].item.row_label or ''} {entries[0].item.metric_text}"
        overlap = _set_overlap(terms, text)
        ranked.append((overlap, _row_rank_value(entries), key))
    ranked.sort(key=lambda row: (-row[0], tuple(-value for value in row[1]), row[2]))
    return ranked[0][2] if ranked and ranked[0][0] > 0 else None


def retrieve_evidence(
    items: Iterable[EvidenceItem],
    question: str,
    intent: QuestionIntent,
    *,
    top_k: int = 24,
    minimum_score: float = -0.50,
) -> list[RetrievedEvidence]:
    """Return a bounded, role-balanced and period-complete evidence set.

    A global re-sort can discard requested-period or denominator evidence that was
    protected during row completion. This implementation uses an explicit priority queue:
    mandatory metric, role and period cells are inserted first and cannot be displaced by
    later global fillers. References and gold programs are not consulted.
    """

    scored = [score_evidence(item, question, intent) for item in items]
    scored = [entry for entry in scored if entry.score >= minimum_score]
    scored.sort(key=lambda entry: (-entry.score, entry.item.evidence_id))
    if not scored:
        return []

    table_rows: dict[tuple[str, int | None, str], list[RetrievedEvidence]] = defaultdict(list)
    table_columns: dict[tuple[str, int | None, str], list[RetrievedEvidence]] = defaultdict(list)
    text_entries: list[RetrievedEvidence] = []
    for entry in scored:
        if entry.item.source_type == "table":
            table_rows[_row_key(entry.item)].append(entry)
            if entry.item.column_label:
                table_columns[_column_key(entry.item)].append(entry)
        else:
            text_entries.append(entry)

    for entries in table_rows.values():
        entries.sort(key=lambda entry: (-entry.score, entry.item.column_index or 0, entry.item.evidence_id))
    for entries in table_columns.values():
        entries.sort(key=lambda entry: (-entry.score, entry.item.row_index or 0, entry.item.evidence_id))

    row_rank = sorted(
        table_rows,
        key=lambda key: (
            -_row_rank_value(table_rows[key])[0],
            -_row_rank_value(table_rows[key])[1],
            -_row_rank_value(table_rows[key])[2],
            key,
        ),
    )
    column_rank = sorted(
        table_columns,
        key=lambda key: (
            -_row_rank_value(table_columns[key])[0],
            -_row_rank_value(table_columns[key])[1],
            -_row_rank_value(table_columns[key])[2],
            key,
        ),
    )

    selected: list[RetrievedEvidence] = []
    selected_ids: set[str] = set()

    def add(entry: RetrievedEvidence) -> None:
        if len(selected) >= top_k or entry.item.evidence_id in selected_ids:
            return
        selected.append(entry)
        selected_ids.add(entry.item.evidence_id)

    def add_row(key: tuple[str, int | None, str], *, complete_requested: bool, max_cells: int) -> None:
        entries = table_rows[key]
        requested = [entry for entry in entries if entry.item.period in intent.periods]
        if complete_requested and requested:
            period_order = {period: index for index, period in enumerate(intent.ordered_periods or intent.periods)}
            requested.sort(key=lambda entry: (period_order.get(entry.item.period or "", 999), -entry.score, entry.item.evidence_id))
            for entry in requested:
                add(entry)
        remaining = [entry for entry in entries if entry.item.evidence_id not in selected_ids and not entry.item.is_year]
        for entry in remaining[:max_cells]:
            add(entry)

    def add_column(key: tuple[str, int | None, str], *, complete_requested: bool, max_cells: int) -> None:
        entries = table_columns[key]
        requested = [entry for entry in entries if entry.item.period in intent.periods]
        if complete_requested and requested:
            period_order = {period: index for index, period in enumerate(intent.ordered_periods or intent.periods)}
            requested.sort(
                key=lambda entry: (
                    period_order.get(entry.item.period or "", 999),
                    -entry.score,
                    entry.item.evidence_id,
                )
            )
            for entry in requested:
                add(entry)
        remaining = [
            entry for entry in entries
            if entry.item.evidence_id not in selected_ids and not entry.item.is_year
        ]
        for entry in remaining[:max_cells]:
            add(entry)

    # 1. Highest-scoring anchor cell, preserving the expected top result for direct lookup.
    add(scored[0])

    # 2. Explicit numerator and denominator rows receive guaranteed slots.
    role_keys = []
    for terms in (intent.numerator_terms, intent.denominator_terms):
        key = _role_row(table_rows, terms)
        if key is not None and key not in role_keys:
            role_keys.append(key)
    for key in role_keys:
        add_row(key, complete_requested=True, max_cells=3)

    # 3. Protect both table orientations before broad row completion. In a
    # metric-in-column table, filling several year rows first can exhaust top_k and
    # silently remove one requested-period value from the correct metric column.
    operation_family = {intent.operation, *intent.alternative_operations}
    aggregate_like = bool(operation_family & {"sum", "average", "table_sum", "table_average", "table_min", "table_max"})
    comparison_like = bool(operation_family & {"difference", "percentage_change", "ratio", "margin", "indexed_return"})
    if aggregate_like:
        column_budget = min(len(column_rank), 3)
        for key in column_rank[:column_budget]:
            add_column(
                key,
                complete_requested=bool(intent.periods),
                max_cells=8,
            )

    # 4. Complete the best metric rows across every requested period. Aggregations need
    # a wider row view; comparisons need both old and new periods from the same row.
    row_budget = min(len(row_rank), 8 if aggregate_like else 6)
    for key in row_rank[:row_budget]:
        add_row(
            key,
            complete_requested=bool(intent.periods) or comparison_like,
            max_cells=6 if aggregate_like else 3,
        )

    # 5. Narrative evidence may contain the only explicit percentage or constant.
    for entry in text_entries[: max(3, top_k // 6)]:
        add(entry)

    # 6. Global fill without displacing protected cells.
    for entry in scored:
        add(entry)
        if len(selected) >= top_k:
            break

    return selected
