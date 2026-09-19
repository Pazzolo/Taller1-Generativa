from itertools import cycle

from src.models.base import ModelRunner


class MockRunner(ModelRunner):
    """Runner determinista sin red: devuelve las respuestas dadas, en ciclo."""

    def __init__(self, responses: list[str]):
        self._responses = cycle(responses)

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        effort: str | None = None,
        structured_schema: dict | None = None,
    ) -> dict:
        return {
            "text": next(self._responses),
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "latency_seconds": 0.0,
            "raw_response": {},
        }
