from types import SimpleNamespace

import pytest

from src.models.anthropic_runner import AnthropicRunner
from tests.fakes import FakeAnthropicClient


def test_generate_returns_normalized_fields():
    client = FakeAnthropicClient(input_tokens=90, output_tokens=7)
    result = AnthropicRunner("claude-opus-4-8", client=client).generate("hello")
    assert result["text"] == '{"category": "billing"}'
    assert (result["input_tokens"], result["output_tokens"]) == (90, 7)
    assert result["reasoning_tokens"] is None
    assert result["latency_seconds"] >= 0
    assert result["raw_response"] == {"fake": True}


def test_request_sets_model_max_tokens_and_omits_unset_sampling_params():
    client = FakeAnthropicClient()
    AnthropicRunner("claude-opus-4-8", client=client, max_tokens=256).generate("hello")
    (call,) = client.calls
    assert call["model"] == "claude-opus-4-8"
    assert call["max_tokens"] == 256
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert "temperature" not in call and "top_p" not in call


def test_temperature_and_top_p_are_forwarded_when_set():
    client = FakeAnthropicClient()
    AnthropicRunner("m", client=client).generate("hello", temperature=0.0, top_p=0.9)
    assert client.calls[0]["temperature"] == 0.0 and client.calls[0]["top_p"] == 0.9


def test_only_text_blocks_are_joined():
    blocks = [
        SimpleNamespace(type="thinking", thinking="hmm"),
        SimpleNamespace(type="text", text='{"category": '),
        SimpleNamespace(type="text", text='"account"}'),
    ]
    result = AnthropicRunner("m", client=FakeAnthropicClient(blocks=blocks)).generate("hello")
    assert result["text"] == '{"category": "account"}'


def test_no_content_blocks_gives_empty_text():
    assert AnthropicRunner("m", client=FakeAnthropicClient(blocks=[])).generate("hello")["text"] == ""


@pytest.mark.parametrize("kwargs", [{"top_k": 5}, {"effort": "low"}, {"structured_schema": {}}])
def test_unsupported_parameters_fail_loudly(kwargs):
    client = FakeAnthropicClient()
    with pytest.raises(NotImplementedError):
        AnthropicRunner("m", client=client).generate("hello", **kwargs)
    assert client.calls == []


def test_missing_api_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicRunner("m")
