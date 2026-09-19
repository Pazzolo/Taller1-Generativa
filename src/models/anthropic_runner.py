import os
import time

from src.models.base import ModelRunner


class AnthropicRunner(ModelRunner):
    def __init__(self, model_name: str, client=None, max_tokens: int = 1024):
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY no está definida; agregarla a .env.")
            from anthropic import Anthropic

            client = Anthropic()
        self._client = client
        self.model_name = model_name
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        effort: str | None = None,
        structured_schema: dict | None = None,
    ) -> dict:
        if top_k is not None or effort is not None or structured_schema is not None:
            raise NotImplementedError("AnthropicRunner solo soporta temperature y top_p por ahora.")

        params = {}
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p

        start = time.perf_counter()
        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **params,
        )
        latency = time.perf_counter() - start

        usage = response.usage
        return {
            "text": "".join(block.text for block in response.content if block.type == "text"),
            "input_tokens": usage.input_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
            "reasoning_tokens": None,
            "latency_seconds": latency,
            "raw_response": response.model_dump(),
        }
