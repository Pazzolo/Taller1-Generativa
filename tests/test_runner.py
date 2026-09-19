import json

import pytest

from src.metrics import mean_latency, total_cost
from src.models.mock_runner import MockRunner
from src.models.openai_runner import OpenAIRunner
from src.pricing import cost_usd
from src.results import append_result, read_results
from src.runner import run_case
from src.schemas import load_cases
from tests.fakes import FakeHTTPError, FakeOpenAIClient

CASE = load_cases()[0]  # case_01, billing
KEY = "propietario_economico"

REQUIRED_OK_FIELDS = {
    "timestamp", "part", "experiment", "model_id", "provider", "model_name", "case_id", "run",
    "temperature", "top_p", "top_k", "effort", "prompt_variant", "status", "input_tokens",
    "output_tokens", "reasoning_tokens", "latency_seconds", "raw_output", "predicted", "expected",
    "parse_ok", "valid_schema", "correct", "cost_usd",
}


def openai_runner(**client_kwargs):
    return OpenAIRunner("gpt-4o-mini", client=FakeOpenAIClient(**client_kwargs))


def test_successful_call_writes_a_complete_row(tmp_path):
    path = tmp_path / "results.jsonl"
    record = run_case(openai_runner(), KEY, CASE, part="1", experiment="t", results_path=path)

    assert REQUIRED_OK_FIELDS <= set(record)
    assert record["status"] == "ok"
    assert (record["predicted"], record["expected"], record["correct"]) == ("billing", "billing", True)
    assert record["provider"] == "openai" and record["model_name"] == "gpt-4o-mini"
    assert record["cost_usd"] == pytest.approx(cost_usd(KEY, 100, 10))
    assert read_results(path) == [record]


def test_wrong_answer_is_recorded_as_incorrect_not_error(tmp_path):
    runner = openai_runner(content='{"category": "technical"}')
    record = run_case(runner, KEY, CASE, part="1", experiment="t", results_path=tmp_path / "r.jsonl")
    assert (record["status"], record["correct"], record["predicted"]) == ("ok", False, "technical")


def test_unparseable_answer_is_recorded_with_parse_ok_false(tmp_path):
    runner = openai_runner(content="billing")
    record = run_case(runner, KEY, CASE, part="1", experiment="t", results_path=tmp_path / "r.jsonl")
    assert (record["status"], record["parse_ok"], record["correct"]) == ("ok", False, False)


def test_api_error_is_logged_with_literal_message_and_http_status(tmp_path):
    path = tmp_path / "r.jsonl"
    runner = openai_runner(error=FakeHTTPError("Unsupported parameter: 'top_k'", 400))
    record = run_case(runner, KEY, CASE, part="2a", experiment="t", results_path=path)

    assert record["status"] == "error"
    assert record["error_type"] == "FakeHTTPError"
    assert record["error_message"] == "Unsupported parameter: 'top_k'"
    assert record["http_status"] == 400
    assert record["correct"] is False
    assert read_results(path) == [record]


def test_error_without_status_code_has_none_http_status(tmp_path):
    runner = openai_runner(error=TimeoutError("timed out"))
    record = run_case(runner, KEY, CASE, part="1", experiment="t", results_path=tmp_path / "r.jsonl")
    assert (record["status"], record["http_status"]) == ("error", None)


def test_programming_errors_are_not_swallowed(tmp_path):
    with pytest.raises(NotImplementedError):
        run_case(openai_runner(), KEY, CASE, part="1", experiment="t", top_k=5, results_path=tmp_path / "r.jsonl")
    assert not (tmp_path / "r.jsonl").exists()


def test_sampling_parameters_reach_the_provider_and_the_row(tmp_path):
    client = FakeOpenAIClient()
    record = run_case(
        OpenAIRunner("gpt-4o-mini", client=client), KEY, CASE,
        part="2b", experiment="t", temperature=0.7, top_p=0.9, run=3, results_path=tmp_path / "r.jsonl",
    )
    assert client.calls[0]["temperature"] == 0.7 and client.calls[0]["top_p"] == 0.9
    assert (record["temperature"], record["top_p"], record["run"]) == (0.7, 0.9, 3)


def test_mock_model_has_no_price_so_cost_is_none(tmp_path):
    record = run_case(MockRunner(['{"category": "billing"}']), "mock", CASE, part="1", experiment="t", results_path=None)
    assert record["cost_usd"] is None and record["provider"] is None


def test_results_path_none_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_case(MockRunner(["x"]), "mock", CASE, part="1", experiment="t", results_path=None)
    assert list(tmp_path.iterdir()) == []


def test_jsonl_appends_one_line_per_call_and_never_overwrites(tmp_path):
    path = tmp_path / "raw" / "results.jsonl"
    for _ in range(3):
        run_case(openai_runner(), KEY, CASE, part="1", experiment="t", results_path=path)
    append_result({"manual": True}, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert all(isinstance(json.loads(line), dict) for line in lines)
    assert json.loads(lines[-1]) == {"manual": True}


def test_read_results_on_missing_file_is_empty(tmp_path):
    assert read_results(tmp_path / "nope.jsonl") == []


def test_jsonl_keeps_non_ascii_text_readable(tmp_path):
    path = tmp_path / "r.jsonl"
    append_result({"text": "reembolso — año"}, path)
    assert "reembolso — año" in path.read_text(encoding="utf-8")


def test_metrics_over_real_shaped_rows(tmp_path):
    path = tmp_path / "r.jsonl"
    ok = run_case(openai_runner(), KEY, CASE, part="1", experiment="t", results_path=path)
    err = run_case(openai_runner(error=TimeoutError("x")), KEY, CASE, part="1", experiment="t", results_path=path)
    rows = read_results(path)
    assert total_cost(rows) == pytest.approx(ok["cost_usd"])
    assert mean_latency(rows) == pytest.approx(ok["latency_seconds"])
    assert err["status"] == "error"
