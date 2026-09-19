import os
import time

from src.models.base import ModelRunner


class OpenAIRunner(ModelRunner):
    def __init__(self, model_name: str, client=None):
        if client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY no está definida; copiar .env.example a .env y completarla.")
            from openai import OpenAI

            client = OpenAI()
        self._client = client
        self.model_name = model_name

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
            raise NotImplementedError("OpenAIRunner solo soporta temperature y top_p por ahora.")

        params = {}
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **params,
        )
        latency = time.perf_counter() - start

        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        return {
            "text": response.choices[0].message.content or "",
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
            "reasoning_tokens": getattr(details, "reasoning_tokens", None),
            "latency_seconds": latency,
            "raw_response": response.model_dump(),
        }
