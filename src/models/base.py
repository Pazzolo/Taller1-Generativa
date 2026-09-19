class ModelRunner:
    """Interfaz común. generate() devuelve un dict normalizado:

    text, input_tokens, output_tokens, reasoning_tokens, latency_seconds, raw_response.
    Un campo que el proveedor no expone va como None; nunca se inventa.
    """

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        effort: str | None = None,
        structured_schema: dict | None = None,
    ) -> dict:
        raise NotImplementedError
