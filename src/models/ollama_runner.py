import os
import time

import requests

from src.models.base import ModelRunner

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class OllamaRunner(ModelRunner):
    def __init__(self, model_name: str, host: str | None = None, think: bool = False, timeout: float = 300):
        self.model_name = model_name
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self.think = think
        self.timeout = timeout

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
            raise NotImplementedError("OllamaRunner solo soporta temperature y top_p por ahora.")

        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": self.think,
            "options": options,
        }
        start = time.perf_counter()
        response = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        latency = time.perf_counter() - start

        if not response.ok:
            try:
                message = response.json().get("error", response.text)
            except ValueError:
                message = response.text
            raise OllamaError(message, response.status_code)

        data = response.json()
        return {
            "text": data["message"]["content"],
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "reasoning_tokens": None,
            "latency_seconds": latency,
            "raw_response": data,
        }
