from __future__ import annotations

import re
from decimal import Decimal
from typing import Iterable, Optional

from .numeric import normalise_scale
from .schemas import QuestionIntent


YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
TOKEN_RE = re.compile(r"[a-z0-9]+")
NUMBER_RE = re.compile(r"(?<![\w.])(?:\d+(?:\.\d+)?|\.\d+)(?![\w.])")

STOPWORDS = {
    "what", "was", "were", "is", "are", "the", "a", "an", "of", "in", "for",
    "to", "from", "during", "how", "much", "many", "did", "does", "do", "by",
    "and", "or", "between", "compared", "with", "according", "reported", "company",
    "year", "years", "amount", "approximately", "respectively", "given", "data",
    "considering", "observed", "ended", "ending", "period", "as", "at", "on",
}

GENERIC_FINANCE_TERMS = {
    "revenue", "sales", "income", "expense", "expenses", "cost", "costs", "profit",
    "loss", "assets", "liabilities", "debt", "cash", "equity", "shares", "tax",
    "interest", "margin", "capital", "inventory", "receivables", "payables", "dividend",
    "operating", "net", "gross", "diluted", "basic", "earnings", "compensation",
    "depreciation", "amortization", "goodwill", "investment", "investments", "lease",
    "payments", "commitments", "obligations", "stock", "price", "return", "volume",
    "mortgages", "balance", "value", "rate", "benefits", "facilities", "securities",
}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twelve": 12, "twenty": 20, "twenty-four": 24, "thirty-six": 36,
}


def _matches(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _year_range(question: str) -> Optional[tuple[str, str]]:
    match = re.search(
        r"\b(?:from|between|during|considering(?: the)? years?)\s+"
        r"((?:19|20)\d{2})\s*(?:-|to|and|through)\s*((?:19|20)\d{2})\b",
        question,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1), match.group(2)
    match = re.search(r"\b((?:19|20)\d{2})\s*-\s*((?:19|20)\d{2})\b", question)
    return (match.group(1), match.group(2)) if match else None


def _ordered_periods(question: str, periods: tuple[str, ...]) -> tuple[str, ...]:
    year_range = _year_range(question)
    if year_range:
        return year_range
    return periods


def _period_count_hint(question: str, periods: tuple[str, ...]) -> Optional[int]:
    year_range = _year_range(question)
    if year_range:
        first, last = map(int, year_range)
        if 0 < abs(last - first) <= 20:
            return abs(last - first) + 1
    if len(periods) >= 2:
        return len(periods)
    q = question.lower().replace("-", " ")
    for word, number in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word.replace('-', ' '))}\s+(?:year|month|quarter|day)s?\b", q):
            return number
    numeric = re.search(r"\b(\d{1,3})\s+(?:year|month|quarter|day)s?\b", q)
    return int(numeric.group(1)) if numeric else None


def _target_scale(question: str) -> Optional[str]:
    match = re.search(r"\b(?:in|into)\s+(thousands?|millions?|billions?|trillions?)\b", question, re.I)
    if not match:
        return None
    return normalise_scale(match.group(1).lower().rstrip("s"))


def _meaningful_tokens(text: str, periods: tuple[str, ...]) -> list[str]:
    period_set = set(periods)
    output = []
    for token in TOKEN_RE.findall(text.lower()):
        if token in STOPWORDS or token in period_set or token.isdigit():
            continue
        output.append(token)
    return output


def _metric_terms(question: str, periods: tuple[str, ...]) -> tuple[str, ...]:
    meaningful = _meaningful_tokens(question, periods)
    finance_positions = [i for i, token in enumerate(meaningful) if token in GENERIC_FINANCE_TERMS]
    selected: list[str] = []
    if finance_positions:
        for position in finance_positions:
            selected.extend(meaningful[max(0, position - 2): position + 3])
    else:
        selected = meaningful
    seen: set[str] = set()
    ordered: list[str] = []
    for term in selected:
        if term not in seen:
            seen.add(term)
            ordered.append(term)
    return tuple(ordered[:18])


def _phrase_terms(text: str, periods: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_meaningful_tokens(text, periods)))[:12]


def _role_terms(question: str, periods: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    q = " ".join(question.lower().split())

    # Denominator-first wording: "what percentage of total X comes from Y".
    denominator_first = [
        r"what\s+percent(?:age)?\s+of\s+(.+?)\s+(?:comes?|came|is|was|were|are)\s+from\s+(.+)",
        r"what\s+percent(?:age)?\s+of\s+(.+?)\s+(?:was|were|is|are)\s+(.+)",
    ]
    for pattern in denominator_first:
        match = re.search(pattern, q)
        if match:
            return _phrase_terms(match.group(2), periods), _phrase_terms(match.group(1), periods)

    numerator_first = [
        r"(.+?)\s+as\s+a\s+percentage\s+of\s+(.+)",
        r"(.+?)\s+as\s+percent(?:age)?\s+of\s+(.+)",
        r"ratio\s+of\s+(.+?)\s+to\s+(.+)",
        r"(.+?)\s+relative\s+to\s+(.+)",
        r"(.+?)\s+divided\s+by\s+(.+)",
        r"what\s+(?:percentage|percent|portion)\s+of\s+(.+?)\s+(?:is|was|were|are)\s+(.+)",
    ]
    for pattern in numerator_first:
        match = re.search(pattern, q)
        if match:
            return _phrase_terms(match.group(1), periods), _phrase_terms(match.group(2), periods)
    return (), ()


def _explicit_constants(question: str, periods: tuple[str, ...]) -> tuple[Decimal, ...]:
    years = set(periods)
    values: list[Decimal] = []
    for raw in NUMBER_RE.findall(question):
        if raw in years:
            continue
        try:
            value = Decimal(raw)
        except Exception:
            continue
        if re.search(rf"\b{re.escape(raw)}\s*(?:day|month|quarter|year|share|vote|transaction)s?\b", question, re.I):
            values.append(value)
    q = question.lower().replace("-", " ")
    for word, number in NUMBER_WORDS.items():
        normalised = word.replace("-", " ")
        if re.search(rf"\b{re.escape(normalised)}\s+(?:day|month|quarter|year)s?\b", q):
            values.append(Decimal(number))
    return tuple(dict.fromkeys(values))


def _operation_and_aggregation(q_lower: str) -> tuple[str, Optional[str]]:
    """Infer a primary operation with precedence chosen for FinQA-style wording.

    Ambiguous phrases such as "per transaction" and "as a percentage of the increase"
    require explicit relational parsing before generic words such as average, increase
    and total.
    """
    weighted_average_lookup = bool(re.search(r"weighted\s+average\s+(?:diluted|basic)?\s*shares?", q_lower))
    per_share_lookup = bool(re.search(r"(?:dividend|earnings|income|book value)\s+per\s+share", q_lower))

    if _matches(q_lower, (r"cumulative\s+(?:total\s+)?return", r"cumulative shareholder return")):
        return "indexed_return", None
    # Extrema and aggregate-comparison constructions must be resolved before the
    # generic financial term "margin". The parser treats "greatest gross margin" as
    # a ratio/margin question and therefore never licensed the table maximum.
    if _matches(q_lower, (
        r"(?:variation|difference|gap|spread)\s+between\s+(?:the\s+)?average\s+and\s+(?:the\s+)?(?:highest|greatest|maximum)",
        r"(?:variation|difference|gap|spread)\s+between\s+(?:the\s+)?(?:highest|greatest|maximum)\s+and\s+(?:the\s+)?average",
        r"(?:variation|difference|gap|spread)\s+between\s+(?:the\s+)?average\s+and\s+(?:the\s+)?(?:lowest|least|minimum)",
        r"(?:range|difference|gap|spread)\s+between\s+(?:the\s+)?(?:highest|greatest|maximum)\s+and\s+(?:the\s+)?(?:lowest|least|minimum)",
    )):
        return "difference", None
    if _matches(q_lower, (r"\b(?:highest|greatest|maximum|max)\b",)):
        return "table_max", "max"
    if _matches(q_lower, (r"\b(?:lowest|least|minimum|min)\b",)):
        return "table_min", "min"
    if _matches(q_lower, (
        r"\bmargin\b", r"as a percentage of", r"as percent(?:age)? of",
        r"what percent(?:age)? of", r"what portion of", r"represented by",
        r"represented .* percentage", r"percent of the total", r"percentage of the total",
    )):
        return "margin", None
    if _matches(q_lower, (
        r"percentage\s+(?:change|increase|decrease|growth|reduction)",
        r"percent\s+(?:change|increase|decrease|higher|lower|growth|reduction)",
        r"growth\s+rate", r"\broi\b", r"change.*percent", r"percentual\s+(?:increase|decrease|reduction)",
    )):
        return "percentage_change", None
    if not per_share_lookup and _matches(q_lower, (
        r"\bratio\b", r"divided by", r"relative to", r"times as",
        r"(?:cost|amount|volume|revenue|price|payment|sales|income)\s+per\s+(?:transaction|car|tower|gwh|unit|employee|customer)",
    )):
        return "ratio", None
    if not weighted_average_lookup and _matches(q_lower, (
        r"\baverage\b", r"\bmean of\b", r"\bannual average\b",
    )):
        return "average", "average"
    if _matches(q_lower, (
        r"\bsum\b", r"\bcombined\b", r"\btogether\b", r"\btotal of\b",
        r"what (?:is|are|was|were) the total(?:\s+(?:amount|value|in))?",
        r"for .* and .* what (?:was|were) the total",
        r"\btotal\b.*\b(?:and|through|to)\b.*(?:19|20)\d{2}",
        r"\btotal\b.*\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:year|month|quarter|day)s?\s+period\b",
    )):
        return "sum", "sum"
    if _matches(q_lower, (
        r"\bdifference\b", r"how much (?:more|less|higher|lower|bigger)",
        r"\bchange in\b", r"\bincrease in\b", r"\bdecrease in\b",
        r"\bnet change\b", r"(?:more|less) than", r"percentage point",
        r"increase .* between", r"decrease .* between",
    )):
        return "difference", None
    if _matches(q_lower, (r"\bproduct\b", r"multiplied by", r"market capitalization", r"total votes")):
        return "product", None
    return "direct_lookup", None


def _alternative_operations(q_lower: str, primary: str, periods: tuple[str, ...]) -> tuple[str, ...]:
    alternatives: list[str] = []

    def add(*ops: str) -> None:
        for op in ops:
            if op != primary and op not in alternatives:
                alternatives.append(op)

    if "average" in q_lower and not re.search(r"weighted\s+average\s+(?:diluted|basic)?\s*shares?", q_lower):
        add("average", "table_average")
    if re.search(r"\b(?:total|combined|sum|together)\b", q_lower):
        add("sum", "table_sum")
    if re.search(r"\b(?:highest|greatest|maximum|max)\b", q_lower):
        add("table_max")
    if re.search(r"\b(?:lowest|least|minimum|min)\b", q_lower):
        add("table_min")
    if re.search(r"\b(?:difference|change|increase|decrease)\b", q_lower):
        add("difference")
        if re.search(r"percent|percentage|rate", q_lower):
            add("percentage_change")
    if re.search(r"percent|percentage|portion|margin", q_lower):
        add("margin", "ratio")
    if re.search(r"(?:variation|difference|gap|spread|range).*\b(?:average|highest|greatest|maximum|lowest|least|minimum)\b", q_lower):
        add("difference", "average", "table_average", "table_max", "table_min")
    if re.search(r"\bper\s+(?:transaction|car|tower|gwh|unit|employee|customer)\b", q_lower):
        add("ratio")
    if len(periods) >= 2 and primary == "direct_lookup":
        add("sum", "average", "difference")
    if primary == "indexed_return":
        add("difference")
    return tuple(alternatives[:8])


def parse_question_intent(question: str) -> QuestionIntent:
    q = " ".join(question.strip().split())
    q_lower = q.lower()
    periods = tuple(dict.fromkeys(YEAR_RE.findall(q)))
    range_bounds = _year_range(q)
    if range_bounds:
        first, last = map(int, range_bounds)
        step = 1 if last >= first else -1
        periods = tuple(str(year) for year in range(first, last + step, step))

    operation, aggregation = _operation_and_aggregation(q_lower)
    target_scale = _target_scale(q)

    answer_type = "plain"
    if _matches(q_lower, (r"percent", r"percentage", r"\bmargin\b", r"growth rate", r"tax rate", r"\broi\b", r"rate of return", r"cumulative.*return")):
        answer_type = "percentage"
    elif _matches(q_lower, (r"what year", r"which year", r"in which year")):
        answer_type = "year"
    elif _matches(q_lower, (r"\bratio\b", r"times as")):
        answer_type = "ratio"
    elif _matches(q_lower, (r"how many", r"number of", r"count of")):
        answer_type = "count"

    if operation in {"percentage_change", "margin", "indexed_return"}:
        answer_type = "percentage"

    # A financial metric name may contain the word "margin" while the question asks
    # for an amount in millions/thousands rather than a percentage. Explicit percent
    # wording remains percentage-typed; otherwise a requested monetary scale wins.
    explicit_percent_cue = bool(re.search(
        r"percent|percentage|tax rate|growth rate|rate of return|\broi\b",
        q_lower,
    ))
    if target_scale and not explicit_percent_cue:
        answer_type = "plain"

    metric_terms = _metric_terms(q, periods)
    numerator_terms, denominator_terms = _role_terms(q, periods)
    entity_terms = tuple(
        term for term in _meaningful_tokens(q, periods)
        if term not in GENERIC_FINANCE_TERMS and term not in {"total", "average", "highest", "lowest"}
    )[:14]
    signal_count = sum((bool(metric_terms), bool(periods), operation != "direct_lookup", bool(numerator_terms or denominator_terms)))
    confidence = min(1.0, 0.40 + 0.15 * signal_count)

    return QuestionIntent(
        operation=operation,
        expected_answer_type=answer_type,
        metric_terms=metric_terms,
        periods=periods,
        ordered_periods=_ordered_periods(q, periods),
        asks_for_change=operation in {"difference", "percentage_change", "indexed_return"},
        asks_for_absolute_change=bool(re.search(r"absolute|amount of (?:the )?change", q_lower)),
        confidence=confidence,
        aggregation=aggregation,
        target_scale=target_scale,
        period_count_hint=_period_count_hint(q, periods),
        explicit_constants=_explicit_constants(q, periods),
        numerator_terms=numerator_terms,
        denominator_terms=denominator_terms,
        entity_terms=entity_terms,
        alternative_operations=_alternative_operations(q_lower, operation, periods),
    )
