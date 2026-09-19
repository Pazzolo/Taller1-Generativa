import pytest

from src.planning import estimate, key_of, plan
from src.models.mock_runner import MockRunner
from src.models.openai_runner import OpenAIRunner
from src.results import call_key, completed_keys, read_results
from src.runner import run_case
from src.schemas import load_cases
from src.sweeps import DECODING_EXPERIMENT, MAX_OUTPUT_TOKENS, decoding_cells, topk_cells
from tests.fakes import FakeOpenAIClient

KEY = "propietario_economico"


def test_decoding_plan_is_750_calls_and_topk_plan_is_200():
    cases = load_cases()
    assert len(plan(KEY, DECODING_EXPERIMENT, decoding_cells(), cases, 5)) == 750
    assert len(plan("open_weight_pequeno", "top_k", topk_cells(), cases, 5)) == 200


def test_plan_has_no_duplicate_calls():
    calls = plan(KEY, DECODING_EXPERIMENT, decoding_cells(), load_cases(), 5)
    assert len({key_of(c) for c in calls}) == len(calls)


def test_planned_keys_match_the_rows_run_case_writes(tmp_path):
    """Si no coincidieran, reanudar repetiría (y cobraría) llamadas ya hechas."""
    path = tmp_path / "r.jsonl"
    calls = plan(KEY, DECODING_EXPERIMENT, decoding_cells()[:2], load_cases()[:2], 2)
    runner = OpenAIRunner("gpt-4o-mini", client=FakeOpenAIClient())
    for c in calls:
        run_case(runner, KEY, c["case"], part="2b", experiment=c["experiment"], run=c["run"],
                 temperature=c["temperature"], top_p=c["top_p"], top_k=c["top_k"], results_path=path)
    done = completed_keys(read_results(path))
    assert [key_of(c) in done for c in calls] == [True] * len(calls)


def test_errors_are_not_treated_as_done_so_they_get_retried(tmp_path):
    path = tmp_path / "r.jsonl"
    (call,) = plan(KEY, DECODING_EXPERIMENT, decoding_cells()[:1], load_cases()[:1], 1)
    runner = OpenAIRunner("gpt-4o-mini", client=FakeOpenAIClient(error=TimeoutError("x")))
    run_case(runner, KEY, call["case"], part="2b", experiment=call["experiment"], run=1,
             temperature=call["temperature"], top_p=call["top_p"], results_path=path)
    assert key_of(call) not in completed_keys(read_results(path))


def test_call_key_distinguishes_every_varied_field():
    base = dict(model_id="m", experiment="e", case_id="c", run=1, temperature=0.0, top_p=1.0, top_k=None, effort=None, prompt_variant="base")
    keys = {call_key(base)}
    for field, value in [("run", 2), ("temperature", 0.3), ("top_p", 0.5), ("top_k", 5), ("case_id", "d"), ("model_id", "n")]:
        keys.add(call_key({**base, field: value}))
    assert len(keys) == 7


def test_output_cap_is_forwarded_to_openai_and_ollama(monkeypatch):
    client = FakeOpenAIClient()
    OpenAIRunner("m", client=client, max_output_tokens=MAX_OUTPUT_TOKENS).generate("p")
    assert client.calls[0]["max_completion_tokens"] == MAX_OUTPUT_TOKENS
    uncapped = FakeOpenAIClient()
    OpenAIRunner("m", client=uncapped).generate("p")
    assert "max_completion_tokens" not in uncapped.calls[0]

    from src.models import ollama_runner
    from tests.fakes import ollama_ok

    sent = {}
    monkeypatch.setattr(ollama_runner.requests, "post", lambda url, json=None, timeout=None: sent.update(json) or ollama_ok())
    ollama_runner.OllamaRunner("m", max_output_tokens=MAX_OUTPUT_TOKENS).generate("p")
    assert sent["options"]["num_predict"] == MAX_OUTPUT_TOKENS


def test_estimate_without_part1_rows_says_so():
    assert "n/a" in estimate(KEY, 10, [])


def test_estimate_scales_with_calls():
    rows = [{"part": "1", "model_id": KEY, "status": "ok", "case_id": "c1", "input_tokens": 100, "output_tokens": 10}]
    text = estimate(KEY, 10, rows)
    assert "estimated input tokens: 1,000" in text and "estimated output tokens: 100" in text
