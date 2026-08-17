from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Iterable, Optional

from .schemas import ParsedNumber


SCALE_MULTIPLIERS: dict[str, Decimal] = {
    "thousand": Decimal("1000"),
    "k": Decimal("1000"),
    "million": Decimal("1000000"),
    "m": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "bn": Decimal("1000000000"),
    "trillion": Decimal("1000000000000"),
    "tn": Decimal("1000000000000"),
}

SCALE_CANONICAL = {
    "k": "thousand",
    "thousand": "thousand",
    "m": "million",
    "million": "million",
    "bn": "billion",
    "billion": "billion",
    "tn": "trillion",
    "trillion": "trillion",
}

CURRENCY_SYMBOLS = {"$": "USD", "£": "GBP", "€": "EUR", "¥": "JPY"}

# The alternatives are ordered so a comma-grouped number or long integer is consumed
# as one token. Boundary assertions prevent the historical 2018 -> 201 + 8 defect.
NUMBER_RE = re.compile(
    r"""
    (?<![\w.])
    (?P<open>\()?
    (?P<sign>[+-])?
    (?P<currency>[$£€¥])?\s*
    (?P<number>
        (?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?
        |
        \.\d+
    )
    \s*(?P<scale>thousand|million|billion|trillion|bn|tn|k|m)?
    \s*(?P<percent>%|percent(?:age)?)?
    (?P<close>\))?
    (?![\w.]|,\d)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
SCALE_CONTEXT_RE = re.compile(
    r"(?:in|figures?\s+in|amounts?\s+in|dollars?\s+in|\$\s*in)\s+"
    r"(thousands?|millions?|billions?|trillions?)",
    flags=re.IGNORECASE,
)


def canonical_decimal(value: Decimal | int | float | str, max_places: int = 10) -> str:
    value = value if isinstance(value, Decimal) else Decimal(str(value))
    if value == 0:
        return "0"
    quant = Decimal(1).scaleb(-max_places)
    try:
        value = value.quantize(quant).normalize()
    except InvalidOperation:
        value = value.normalize()
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def normalise_scale(scale: Optional[str]) -> Optional[str]:
    if not scale:
        return None
    return SCALE_CANONICAL.get(scale.lower().rstrip("s"))


def detect_context_scale(text: str) -> Optional[str]:
    match = SCALE_CONTEXT_RE.search(text or "")
    if not match:
        return None
    return normalise_scale(match.group(1).lower().rstrip("s"))


def _parse_decimal(number_text: str) -> Decimal:
    cleaned = number_text.replace(",", "")
    if cleaned.startswith("."):
        cleaned = "0" + cleaned
    return Decimal(cleaned)


def extract_numbers(text: str, inherited_scale: Optional[str] = None) -> list[ParsedNumber]:
    results: list[ParsedNumber] = []
    inherited_scale = normalise_scale(inherited_scale)
    for match in NUMBER_RE.finditer(text or ""):
        groups = match.groupdict()
        try:
            value = _parse_decimal(groups["number"])
        except InvalidOperation:
            continue

        negative_parentheses = bool(groups["open"] and groups["close"])
        if groups["sign"] == "-" or negative_parentheses:
            value = -value

        explicit_scale = normalise_scale(groups["scale"])
        effective_scale = explicit_scale or inherited_scale
        multiplier = SCALE_MULTIPLIERS.get(effective_scale or "", Decimal("1"))
        base_value = value * multiplier
        is_percent = bool(groups["percent"])
        currency = CURRENCY_SYMBOLS.get(groups["currency"] or "")
        number_without_commas = groups["number"].replace(",", "")
        is_year = bool(
            YEAR_RE.fullmatch(number_without_commas)
            and not is_percent
            and not currency
            and not explicit_scale
            and not negative_parentheses
            and groups["sign"] not in {"-", "+"}
        )
        results.append(
            ParsedNumber(
                raw=match.group(0).strip(),
                value=value,
                canonical=canonical_decimal(value),
                start=match.start(),
                end=match.end(),
                is_percent=is_percent,
                is_year=is_year,
                currency=currency,
                explicit_scale=explicit_scale,
                effective_scale=effective_scale,
                base_value=base_value,
                negative_parentheses=negative_parentheses,
            )
        )
    return results


def parse_single_number(text: str, inherited_scale: Optional[str] = None) -> Optional[ParsedNumber]:
    numbers = extract_numbers(text, inherited_scale=inherited_scale)
    if len(numbers) != 1:
        return None
    return numbers[0]


def numeric_equal(
    left: ParsedNumber | Decimal | str,
    right: ParsedNumber | Decimal | str,
    *,
    absolute_tolerance: Decimal = Decimal("0.01"),
    relative_tolerance: Decimal = Decimal("0.005"),
    compare_base_values: bool = False,
    require_percent_compatibility: bool = True,
) -> bool:
    def coerce(value: ParsedNumber | Decimal | str) -> tuple[Decimal, Optional[bool]]:
        if isinstance(value, ParsedNumber):
            numeric = value.base_value if compare_base_values and value.base_value is not None else value.value
            return numeric, value.is_percent
        if isinstance(value, Decimal):
            return value, None
        parsed = parse_single_number(str(value))
        if parsed is None:
            raise ValueError(f"Expected exactly one number: {value!r}")
        numeric = parsed.base_value if compare_base_values and parsed.base_value is not None else parsed.value
        return numeric, parsed.is_percent

    try:
        left_value, left_percent = coerce(left)
        right_value, right_percent = coerce(right)
    except ValueError:
        return False
    if require_percent_compatibility and left_percent is not None and right_percent is not None:
        if left_percent != right_percent:
            return False
    difference = abs(left_value - right_value)
    if difference <= absolute_tolerance:
        return True
    denominator = max(abs(left_value), abs(right_value), Decimal("1"))
    return difference / denominator <= relative_tolerance


def format_answer(value: Decimal, answer_type: str, places: int = 4) -> str:
    if answer_type == "percentage":
        rounded = value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
        return f"{canonical_decimal(rounded, max_places=places)}%"
    rounded = value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return canonical_decimal(rounded, max_places=places)


def first_numeric_value(text: str) -> Optional[Decimal]:
    numbers = extract_numbers(text)
    return numbers[0].value if numbers else None


def contains_valid_number(text: str) -> bool:
    return bool(extract_numbers(text))


def unique_numeric_strings(values: Iterable[Decimal], answer_type: str) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        candidate = format_answer(value, answer_type)
        if candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    return output
