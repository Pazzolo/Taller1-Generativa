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


class FakeAnthropicClient:
    """Imita client.messages.create sin red; guarda los kwargs recibidos."""

    def __init__(self, blocks=None, input_tokens=90, output_tokens=7, error=None):
        self.calls = []
        self._blocks = blocks if blocks is not None else [SimpleNamespace(type="text", text='{"category": "billing"}')]
        self._usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
        self._error = error
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return SimpleNamespace(content=self._blocks, usage=self._usage, model_dump=lambda: {"fake": True})


class FakeHTTPResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


def ollama_ok(content='{"category": "billing"}', prompt_eval_count=60, eval_count=9):
    return FakeHTTPResponse(
        200,
        {
            "model": "qwen3:1.7b",
            "message": {"role": "assistant", "content": content},
            "done": True,
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "load_duration": 123,
        },
    )
