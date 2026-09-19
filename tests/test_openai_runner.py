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


@pytest.mark.parametrize("kwargs", [{"effort": "low"}])
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


def test_top_k_is_sent_in_extra_body_so_the_api_can_reject_it():
    client = FakeOpenAIClient()
    OpenAIRunner("gpt-4o-mini", client=client).generate("hello", top_k=5)
    (call,) = client.calls
    assert call["extra_body"] == {"top_k": 5}
    assert "top_k" not in call


def test_structured_schema_is_sent_as_strict_json_schema_response_format():
    from src.schemas import CATEGORY_SCHEMA

    client = FakeOpenAIClient()
    OpenAIRunner("gpt-4o-mini", client=client).generate("hello", structured_schema=CATEGORY_SCHEMA)
    (call,) = client.calls
    assert call["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "ticket_category", "strict": True, "schema": CATEGORY_SCHEMA},
    }


def test_no_response_format_without_a_schema():
    client = FakeOpenAIClient()
    OpenAIRunner("gpt-4o-mini", client=client).generate("hello")
    assert "response_format" not in client.calls[0]
