import pytest

from src.experiments.part1 import run
from src.metrics import accuracy
from src.models.mock_runner import MockRunner
from src.prompts import build_base_prompt
from src.schemas import load_cases


def oracle_answers(cases):
    return ['{"category": "%s"}' % c.expected.category for c in cases]


def test_oracle_mock_reaches_full_accuracy():
    cases = load_cases()
    records = run(MockRunner(oracle_answers(cases)), cases)
    assert len(records) == 10
    assert accuracy(records) == 1.0


def test_constant_mock_matches_only_its_category():
    cases = load_cases()
    records = run(MockRunner(['{"category": "billing"}']), cases)
    expected_hits = sum(c.expected.category == "billing" for c in cases)
    assert sum(r["correct"] for r in records) == expected_hits
    assert accuracy(records) == expected_hits / len(cases)


def test_unparseable_mock_scores_zero_and_is_flagged():
    cases = load_cases()
    records = run(MockRunner(["billing"]), cases)
    assert accuracy(records) == 0.0
    assert not any(r["parse_ok"] for r in records)


def test_records_keep_case_order_and_ids():
    cases = load_cases()
    records = run(MockRunner(['{"category": "account"}']), cases)
    assert [r["case_id"] for r in records] == [c.id for c in cases]


def test_mock_result_has_normalized_fields():
    result = MockRunner(["x"]).generate("prompt")
    assert set(result) == {
        "text", "input_tokens", "output_tokens", "reasoning_tokens", "latency_seconds", "raw_response",
    }
    assert result["input_tokens"] is None


def test_mock_cycles_through_responses():
    runner = MockRunner(["a", "b"])
    assert [runner.generate("p")["text"] for _ in range(3)] == ["a", "b", "a"]


def test_base_prompt_contains_ticket_and_all_categories():
    prompt = build_base_prompt("my ticket text")
    assert prompt.endswith("my ticket text")
    for category in ("billing", "technical", "account"):
        assert category in prompt


def test_accuracy_on_empty_records_raises():
    with pytest.raises(ZeroDivisionError):
        accuracy([])
