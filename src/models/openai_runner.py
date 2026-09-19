import os
import time

from src.models.base import ModelRunner


class OpenAIRunner(ModelRunner):
    def __init__(self, model_name: str, client=None, max_output_tokens: int | None = None):
        if client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY no está definida; copiar .env.example a .env y completarla.")
            from openai import OpenAI

            client = OpenAI()
        self._client = client
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        effort: str | None = None,
        structured_schema: dict | None = None,
    ) -> dict:
        params = {}
        if effort is not None:
            params["reasoning_effort"] = effort
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if top_k is not None:
            # El SDK no acepta top_k; va en el cuerpo para que la API decida si lo rechaza.
            params["extra_body"] = {"top_k": top_k}

        if structured_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "ticket_category", "strict": True, "schema": structured_schema},
            }
        if self.max_output_tokens is not None:
            params["max_completion_tokens"] = self.max_output_tokens

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
