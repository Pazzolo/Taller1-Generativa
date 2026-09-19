import pytest

from src.models.anthropic_runner import AnthropicRunner
from src.models.ollama_runner import OllamaRunner
from src.models.openai_runner import OpenAIRunner
from src.models.registry import RUNNERS, get_runner
from src.pricing import PRICES


def test_every_price_row_has_a_registered_provider():
    assert {row["provider"] for row in PRICES.values()} <= set(RUNNERS)


def test_get_runner_builds_the_right_class_and_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    expected = {
        "propietario_economico": (OpenAIRunner, "gpt-4o-mini"),
        "propietario_grande": (OpenAIRunner, "gpt-5.5"),
        "propietario_balanceado": (AnthropicRunner, "claude-sonnet-4-6"),
        "open_weight_pequeno": (OllamaRunner, "qwen3:1.7b"),
    }
    for key, (cls, model_name) in expected.items():
        runner = get_runner(key)
        assert isinstance(runner, cls) and runner.model_name == model_name


def test_unknown_model_key_raises():
    with pytest.raises(KeyError):
        get_runner("nope")
