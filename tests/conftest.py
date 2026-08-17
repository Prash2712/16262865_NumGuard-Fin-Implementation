from pathlib import Path

import pytest

from numguard_fin.dataset import load_examples


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "fixtures" / "finqa_structured_fixture.json"


@pytest.fixture(scope="session")
def examples(fixture_path):
    return load_examples(fixture_path)


@pytest.fixture
def example_by_suffix(examples):
    def get(suffix: str):
        return next(example for example in examples if example.example_id.endswith(suffix))
    return get
