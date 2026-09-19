from src.models.anthropic_runner import AnthropicRunner
from src.models.base import ModelRunner
from src.models.ollama_runner import OllamaRunner
from src.models.openai_runner import OpenAIRunner
from src.pricing import PRICES

RUNNERS = {
    "openai": OpenAIRunner,
    "anthropic": AnthropicRunner,
    "ollama": OllamaRunner,
}


def get_runner(model_key: str) -> ModelRunner:
    price = PRICES[model_key]
    return RUNNERS[price["provider"]](price["model_id"])
