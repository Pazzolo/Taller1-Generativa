import pytest
import requests

from src.models import ollama_runner
from src.models.ollama_runner import OllamaError, OllamaRunner
from tests.fakes import FakeHTTPResponse, ollama_ok


@pytest.fixture
def posts(monkeypatch):
    calls = []

    def install(response=None, error=None):
        def fake_post(url, json=None, timeout=None):
            calls.append({"url": url, "json": json, "timeout": timeout})
            if error:
                raise error
            return response

        monkeypatch.setattr(ollama_runner.requests, "post", fake_post)
        return calls

    return install


def test_generate_returns_normalized_fields(posts):
    posts(ollama_ok(prompt_eval_count=60, eval_count=9))
    result = OllamaRunner("qwen3:1.7b").generate("hello")
    assert result["text"] == '{"category": "billing"}'
    assert (result["input_tokens"], result["output_tokens"]) == (60, 9)
    assert result["reasoning_tokens"] is None
    assert result["latency_seconds"] >= 0
    assert result["raw_response"]["load_duration"] == 123


def test_request_shape_and_thinking_disabled_by_default(posts):
    calls = posts(ollama_ok())
    OllamaRunner("qwen3:1.7b", host="http://example:1234/").generate("hello")
    (call,) = calls
    assert call["url"] == "http://example:1234/api/chat"
    assert call["json"]["model"] == "qwen3:1.7b"
    assert call["json"]["messages"] == [{"role": "user", "content": "hello"}]
    assert call["json"]["stream"] is False
    assert call["json"]["think"] is False
    assert call["json"]["options"] == {}


def test_temperature_and_top_p_go_in_options(posts):
    calls = posts(ollama_ok())
    OllamaRunner("m").generate("hello", temperature=0.0, top_p=0.9)
    assert calls[0]["json"]["options"] == {"temperature": 0.0, "top_p": 0.9}


def test_host_comes_from_env_then_default(posts, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://from-env:1")
    assert OllamaRunner("m").host == "http://from-env:1"
    monkeypatch.delenv("OLLAMA_HOST")
    assert OllamaRunner("m").host == "http://localhost:11434"


def test_missing_token_counts_stay_none(posts):
    posts(FakeHTTPResponse(200, {"message": {"content": "x"}, "done": True}))
    result = OllamaRunner("m").generate("hello")
    assert result["input_tokens"] is None and result["output_tokens"] is None


def test_http_error_keeps_literal_message_and_status(posts):
    posts(FakeHTTPResponse(404, {"error": "model 'qwen3:1.7b' not found"}))
    with pytest.raises(OllamaError) as exc:
        OllamaRunner("qwen3:1.7b").generate("hello")
    assert exc.value.status_code == 404
    assert str(exc.value) == "model 'qwen3:1.7b' not found"


def test_http_error_with_non_json_body_falls_back_to_text(posts):
    posts(FakeHTTPResponse(500, None, text="Internal Server Error"))
    with pytest.raises(OllamaError, match="Internal Server Error"):
        OllamaRunner("m").generate("hello")


def test_connection_error_propagates_for_the_runner_to_log(posts):
    posts(error=requests.ConnectionError("refused"))
    with pytest.raises(requests.ConnectionError):
        OllamaRunner("m").generate("hello")


@pytest.mark.parametrize("kwargs", [{"top_k": 5}, {"effort": "low"}, {"structured_schema": {}}])
def test_unsupported_parameters_fail_loudly(posts, kwargs):
    calls = posts(ollama_ok())
    with pytest.raises(NotImplementedError):
        OllamaRunner("m").generate("hello", **kwargs)
    assert calls == []
