import pytest

from src.models.openai_runner import OpenAIRunner
from tests.fakes import FakeOpenAIClient


def test_generate_returns_normalized_fields():
    client = FakeOpenAIClient(prompt_tokens=120, completion_tokens=8)
    result = OpenAIRunner("gpt-4o-mini", client=client).generate("hello")
    assert result["text"] == '{"category": "billing"}'
    assert result["input_tokens"] == 120
    assert result["output_tokens"] == 8
    assert result["reasoning_tokens"] == 0
    assert result["latency_seconds"] >= 0
    assert result["raw_response"] == {"fake": True, "content": '{"category": "billing"}'}


def test_request_uses_model_and_prompt_and_omits_unset_sampling_params():
    client = FakeOpenAIClient()
    OpenAIRunner("gpt-4o-mini", client=client).generate("hello")
    (call,) = client.calls
    assert call["model"] == "gpt-4o-mini"
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert "temperature" not in call and "top_p" not in call


def test_temperature_and_top_p_are_forwarded_when_set():
    client = FakeOpenAIClient()
    OpenAIRunner("gpt-4o-mini", client=client).generate("hello", temperature=0.0, top_p=0.9)
    (call,) = client.calls
    assert call["temperature"] == 0.0
    assert call["top_p"] == 0.9


@pytest.mark.parametrize("kwargs", [{"top_k": 5}, {"effort": "low"}, {"structured_schema": {}}])
def test_unsupported_parameters_fail_loudly_instead_of_being_dropped(kwargs):
    client = FakeOpenAIClient()
    with pytest.raises(NotImplementedError):
        OpenAIRunner("gpt-4o-mini", client=client).generate("hello", **kwargs)
    assert client.calls == []


def test_empty_content_becomes_empty_text():
    result = OpenAIRunner("gpt-4o-mini", client=FakeOpenAIClient(content=None)).generate("hello")
    assert result["text"] == ""


def test_missing_api_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIRunner("gpt-4o-mini")
