from types import SimpleNamespace


class FakeOpenAIClient:
    """Imita client.chat.completions.create sin red; guarda los kwargs recibidos."""

    def __init__(self, content='{"category": "billing"}', prompt_tokens=100, completion_tokens=10, error=None):
        self.calls = []
        self._content = content
        self._usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        )
        self._error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=self._usage,
            model_dump=lambda: {"fake": True, "content": self._content},
        )


class FakeHTTPError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code
