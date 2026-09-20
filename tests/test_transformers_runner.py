import logging

import pytest

torch = pytest.importorskip("torch")

from src.models.transformers_runner import TransformersRunner  # noqa: E402
from src.runner import run_case  # noqa: E402
from src.schemas import load_cases  # noqa: E402


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, prompt, return_tensors=None):
        return {"input_ids": torch.tensor([[5, 6, 7]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=False):
        return "|".join(str(int(i)) for i in ids)


class FakeModel:
    def __init__(self, log_message=None):
        self.calls = []
        self.log_message = log_message

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=0, pad_token_id=None, **kwargs):
        self.calls.append({"max_new_tokens": max_new_tokens, **kwargs})
        if self.log_message:
            logging.getLogger("transformers").warning(self.log_message)
        return torch.cat([input_ids, torch.full((1, max_new_tokens), 9)], dim=1)


def runner(**options):
    model = FakeModel(options.pop("log_message", None))
    return TransformersRunner(FakeTokenizer(), model, **options), model


def test_generate_returns_the_normalized_fields_with_real_token_counts():
    r, _ = runner(max_new_tokens=4)
    out = r.generate("hello")
    assert out["text"] == "9|9|9|9"
    assert (out["input_tokens"], out["output_tokens"]) == (3, 4)
    assert out["reasoning_tokens"] is None and out["latency_seconds"] >= 0


def test_greedy_mode_forwards_temperature_so_the_library_can_report_it_is_ignored():
    r, model = runner(do_sample=False)
    r.generate("hello", temperature=1.5)
    assert model.calls[0]["do_sample"] is False and model.calls[0]["temperature"] == 1.5


def test_sampling_mode_forwards_only_the_parameters_that_are_set():
    r, model = runner(do_sample=True)
    r.generate("hello", top_k=1)
    assert model.calls[0] == {"max_new_tokens": 60, "do_sample": True, "top_k": 1}


def test_seed_is_applied_before_generating():
    r, _ = runner(do_sample=True, seed=1234)
    r.generate("hello")
    assert torch.initial_seed() == 1234


def test_unsupported_parameters_fail_loudly():
    r, model = runner()
    for kwargs in ({"effort": "low"}, {"structured_schema": {}}):
        with pytest.raises(NotImplementedError):
            r.generate("hello", **kwargs)
    assert model.calls == []


def test_clone_shares_the_loaded_model_and_overrides_options():
    r, model = runner(do_sample=False, max_new_tokens=10)
    c = r.clone(do_sample=True, seed=7)
    assert c.model is model and c.tokenizer is r.tokenizer
    assert (c.do_sample, c.seed, c.max_new_tokens) == (True, 7, 10)
    assert (r.do_sample, r.seed) == (False, None)


def test_library_notices_are_captured():
    r, _ = runner(log_message="flags are not valid")
    assert r.generate("hello")["notices"] == ["flags are not valid"]


def test_a_local_generation_becomes_a_free_row_in_the_same_format(tmp_path):
    r, _ = runner(max_new_tokens=5, log_message="temperature ignored")
    path = tmp_path / "results.jsonl"
    record = run_case(r, "gpt2_base", load_cases()[0], part="0", experiment="base_model_classification", results_path=path)
    assert record["cost_usd"] == 0.0
    assert (record["part"], record["model_id"], record["provider"], record["model_name"]) == ("0", "gpt2_base", "transformers", "openai-community/gpt2")
    assert (record["input_tokens"], record["output_tokens"]) == (3, 5)
    assert record["notices"] == ["temperature ignored"]
    assert record["parse_ok"] is False and record["correct"] is False
