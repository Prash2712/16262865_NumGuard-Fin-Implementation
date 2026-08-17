from __future__ import annotations

import re
from decimal import Decimal
from typing import Iterable, Optional

from .numeric import detect_context_scale, extract_numbers, normalise_scale
from .schemas import EvidenceItem, FinQAExample


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
CURRENCY_RE = re.compile(r"[$£€¥]|\b(?:USD|GBP|EUR|JPY|dollars?|pounds?|euros?|yen)\b", flags=re.IGNORECASE)


def _currency_from_text(text: str) -> Optional[str]:
    match = CURRENCY_RE.search(text or "")
    if not match:
        return None
    value = match.group(0).upper()
    mapping = {
        "$": "USD", "£": "GBP", "€": "EUR", "¥": "JPY",
        "DOLLAR": "USD", "DOLLARS": "USD",
        "POUND": "GBP", "POUNDS": "GBP",
        "EURO": "EUR", "EUROS": "EUR",
        "YEN": "JPY",
    }
    return mapping.get(value, value)


def _period_from_text(text: str) -> Optional[str]:
    match = YEAR_RE.search(text or "")
    return match.group(0) if match else None




def _table_number_is_year(
    *,
    parsed_is_year: bool,
    cell: str,
    row_label: str,
    column_label: str,
    column_index: int,
) -> bool:
    """Disambiguate calendar years from financial values in the 1900--2099 range.

    Numeric parsers necessarily flag values such as ``2000`` as possible years. In a
    financial table, however, a value of 2,000 under a metric column is usually an
    amount, not a period. A table number is treated as a year only when the table
    structure assigns it a period role (for example a ``Year`` column or the first
    column of a year-oriented table). Year-valued metric cells remain ordinary
    numeric evidence while inheriting their period from the row or column label.
    """
    if not parsed_is_year:
        return False
    stripped = (cell or "").strip()
    if not re.fullmatch(r"(?:19|20)\d{2}", stripped):
        return False
    header = " ".join((column_label or "").lower().split())
    if re.search(r"\b(?:year|fiscal\s+year|fy|period|date)\b", header):
        return True
    # In year-in-rows tables the first cell is the row's period label. Restricting
    # this fallback to the first column prevents values such as 2000 in a metric
    # column from being discarded as dates.
    return column_index == 0 and stripped == (row_label or "").strip()


def _table_global_context(example: FinQAExample) -> str:
    header = " ".join(example.table[0]) if example.table else ""
    surrounding = " ".join((*example.pre_text[-3:], *example.post_text[:3]))
    return f"{header} {surrounding}"


def _row_label(row: tuple[str, ...]) -> str:
    for cell in row:
        if cell and not extract_numbers(cell):
            return cell.strip()
    return row[0].strip() if row else ""


def build_evidence_ledger(example: FinQAExample) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    global_context = _table_global_context(example)
    global_scale = detect_context_scale(global_context)
    global_currency = _currency_from_text(global_context)

    if example.table:
        header = example.table[0]
        for row_index, row in enumerate(example.table[1:], start=1):
            row_label = _row_label(row)
            row_scale = detect_context_scale(" ".join(row)) or global_scale
            row_currency = _currency_from_text(" ".join(row)) or global_currency
            for column_index, cell in enumerate(row):
                if not cell.strip():
                    continue
                column_label = header[column_index].strip() if column_index < len(header) else ""
                inherited_scale = detect_context_scale(f"{column_label} {row_label}") or row_scale
                cell_currency = _currency_from_text(f"{column_label} {cell}") or row_currency
                parsed_numbers = extract_numbers(cell, inherited_scale=inherited_scale)
                for number_index, parsed in enumerate(parsed_numbers):
                    # Percentages are dimensionless. Inheriting a table-wide monetary
                    # scale (for example "amounts in millions") would incorrectly turn
                    # 25% into 25 million.
                    scale = None if parsed.is_percent else (parsed.effective_scale or inherited_scale)
                    currency = None if parsed.is_percent else (parsed.currency or cell_currency)
                    # FinQA tables use both common orientations:
                    #   * metrics in rows and years in columns; and
                    #   * years in rows and metrics in columns.
                    # Periods must be inherited from both table axes; otherwise
                    # column-wise aggregates and comparisons lose their year identity.
                    table_is_year = _table_number_is_year(
                        parsed_is_year=parsed.is_year,
                        cell=cell,
                        row_label=row_label,
                        column_label=column_label,
                        column_index=column_index,
                    )
                    period = _period_from_text(column_label) or _period_from_text(row_label)
                    if period is None and table_is_year:
                        period = parsed.canonical
                    evidence_id = f"table:r{row_index}:c{column_index}:n{number_index}"
                    text = f"{row_label} | {column_label} | {cell}".strip(" |")
                    items.append(
                        EvidenceItem(
                            evidence_id=evidence_id,
                            source_type="table",
                            raw_number=parsed.raw,
                            value=parsed.value,
                            canonical=parsed.canonical,
                            base_value=parsed.value if parsed.is_percent else (parsed.base_value or parsed.value),
                            is_percent=parsed.is_percent,
                            is_year=table_is_year,
                            currency=currency,
                            scale=normalise_scale(scale),
                            text=text,
                            metric_text=row_label,
                            period=period,
                            row_index=row_index,
                            column_index=column_index,
                            row_label=row_label or None,
                            column_label=column_label or None,
                        )
                    )

    sentences: list[str] = []
    for paragraph in (*example.pre_text, *example.post_text):
        sentences.extend(part.strip() for part in SENTENCE_SPLIT_RE.split(paragraph) if part.strip())
    for sentence_index, sentence in enumerate(sentences):
        inherited_scale = detect_context_scale(sentence)
        sentence_currency = _currency_from_text(sentence)
        period = _period_from_text(sentence)
        for number_index, parsed in enumerate(extract_numbers(sentence, inherited_scale=inherited_scale)):
            evidence_id = f"text:s{sentence_index}:n{number_index}"
            items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    source_type="text",
                    raw_number=parsed.raw,
                    value=parsed.value,
                    canonical=parsed.canonical,
                    base_value=parsed.value if parsed.is_percent else (parsed.base_value or parsed.value),
                    is_percent=parsed.is_percent,
                    is_year=parsed.is_year,
                    currency=None if parsed.is_percent else (parsed.currency or sentence_currency),
                    scale=None if parsed.is_percent else normalise_scale(parsed.effective_scale or inherited_scale),
                    text=sentence,
                    metric_text=sentence,
                    period=parsed.canonical if parsed.is_year else period,
                    sentence_index=sentence_index,
                    character_start=parsed.start,
                    character_end=parsed.end,
                )
            )

    # Preserve source order while removing exact duplicates from the same location.
    deduplicated: list[EvidenceItem] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (item.evidence_id, item.canonical, item.raw_number)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)
    return deduplicated


def evidence_by_id(items: Iterable[EvidenceItem]) -> dict[str, EvidenceItem]:
    return {item.evidence_id: item for item in items}
