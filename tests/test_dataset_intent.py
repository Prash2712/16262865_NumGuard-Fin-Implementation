import json

import pytest

from numguard_fin.dataset import coerce_example, load_examples
from numguard_fin.intent import parse_question_intent


def test_loads_structured_fixture(fixture_path):
    examples = load_examples(fixture_path)
    assert len(examples) == 12
    assert examples[0].table[0][1] == "2009"


def test_coerce_official_shape():
    raw = {
        "id": "x-1",
        "pre_text": ["A"],
        "post_text": ["B"],
        "table_ori": [["", "2023"], ["Revenue", "100"]],
        "qa": {"question": "What was revenue in 2023?", "answer": "100", "program": ["table_1"]},
    }
    example = coerce_example(raw)
    assert example.example_id == "x-1"
    assert example.reference_answer == "100"
    assert example.reference_program == "table_1"
    assert example.table[1] == ("Revenue", "100")


def test_blank_reference_is_filtered(tmp_path):
    payload = [
        {"id": "ok", "table": [["Metric", "2023"], ["Revenue", "100"]], "qa": {"question": "Q", "answer": "100"}},
        {"id": "bad", "table": [["Metric", "2023"], ["Revenue", "100"]], "qa": {"question": "Q", "answer": ""}},
    ]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert [example.example_id for example in load_examples(path)] == ["ok"]


@pytest.mark.parametrize(
    "question,operation,answer_type",
    [
        ("What was the interest expense in 2009?", "direct_lookup", "plain"),
        ("What was the percentage change in revenue from 2021 to 2022?", "percentage_change", "percentage"),
        ("What was the increase in expenses from 2022 to 2023?", "difference", "plain"),
        ("What was the net income margin in 2023?", "margin", "percentage"),
        ("What was the combined revenue from Europe and Asia?", "sum", "plain"),
        ("What was the ratio of 2023 revenue to 2022 revenue?", "ratio", "ratio"),
        ("What was the average capital expenditure in 2022 and 2023?", "average", "plain"),
        ("How many weighted average diluted shares were reported in 2023?", "direct_lookup", "count"),
        ("What was the dividend per share in 2023?", "direct_lookup", "plain"),
    ],
)
def test_question_intent(question, operation, answer_type):
    intent = parse_question_intent(question)
    assert intent.operation == operation
    assert intent.expected_answer_type == answer_type


def test_period_order_from_to():
    intent = parse_question_intent("What was the change from 2021 to 2023?")
    assert intent.ordered_periods == ("2021", "2023")


def test_metric_terms_preserve_financial_meaning():
    intent = parse_question_intent("What was the interest expense in 2009?")
    assert "interest" in intent.metric_terms
    assert "expense" in intent.metric_terms
