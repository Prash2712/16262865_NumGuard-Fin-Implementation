from decimal import Decimal

import pytest

from numguard_fin.numeric import (
    canonical_decimal,
    detect_context_scale,
    extract_numbers,
    format_answer,
    numeric_equal,
    parse_single_number,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2018", ["2018"]),
        ("41932", ["41932"]),
        (".3", ["0.3"]),
        ("0.3", ["0.3"]),
        ("1,234,567", ["1234567"]),
        ("-45", ["-45"]),
        ("(45)", ["-45"]),
        ("+12.5", ["12.5"]),
        ("12.45%", ["12.45"]),
        ("$3.8 million", ["3.8"]),
        ("2021 and 2022", ["2021", "2022"]),
        ("Revenue was 41932 in 2018", ["41932", "2018"]),
    ],
)
def test_whole_number_parsing(text, expected):
    assert [item.canonical for item in extract_numbers(text)] == expected


def test_year_is_not_fragmented():
    values = extract_numbers("Years 2009, 2018 and 2023")
    assert [item.canonical for item in values] == ["2009", "2018", "2023"]
    assert all(item.is_year for item in values)


def test_long_integer_is_not_fragmented():
    value = parse_single_number("41932")
    assert value is not None and value.value == Decimal("41932")


def test_leading_decimal_is_not_rewritten_as_integer():
    value = parse_single_number(".5")
    assert value is not None and value.value == Decimal("0.5")
    assert not numeric_equal(".5", "5")


def test_scale_propagation():
    value = parse_single_number("3.8", inherited_scale="million")
    assert value is not None
    assert value.effective_scale == "million"
    assert value.base_value == Decimal("3800000")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Amounts in millions", "million"),
        ("Figures in thousands", "thousand"),
        ("$ in billions", "billion"),
        ("No scale", None),
    ],
)
def test_detect_context_scale(text, expected):
    assert detect_context_scale(text) == expected


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("12.4500%", "12.45%", True),
        ("12.45", "12.45%", False),
        ("41932", "41932.00", True),
        ("100", "100.6", False),
        ("-45", "(45)", True),
    ],
)
def test_numeric_equal_semantics(left, right, expected):
    assert numeric_equal(left, right) is expected


def test_format_answer():
    assert format_answer(Decimal("12.449799"), "percentage") == "12.4498%"
    assert format_answer(Decimal("90.0000"), "plain") == "90"


@pytest.mark.parametrize(
    "value,expected",
    [(Decimal("0"), "0"), (Decimal("12.340000"), "12.34"), (Decimal("-0.5000"), "-0.5")],
)
def test_canonical_decimal(value, expected):
    assert canonical_decimal(value) == expected
