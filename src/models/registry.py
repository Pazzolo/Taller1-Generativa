from src.models.base import ModelRunner
from src.models.openai_runner import OpenAIRunner
from src.pricing import PRICES


def get_runner(model_key: str) -> ModelRunner:
    price = PRICES[model_key]
    if price["provider"] == "openai":
        return OpenAIRunner(price["model_id"])
    raise NotImplementedError(f"Proveedor '{price['provider']}' aún no implementado.")
