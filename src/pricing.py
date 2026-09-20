LOCAL = "n/a (local, sin precio por token)"

# Las nueve filas de la tabla semestral del curso (anexo del enunciado, generado el 2026-09-18), con la fecha
# de verificación de cada fila. `extra_gpt55` NO es una fila del curso: se usó en una exploración inicial.
PRICES = {
    "propietario_grande": {"provider": "anthropic", "model_id": "claude-opus-4-8", "input_per_million": 5.0, "output_per_million": 25.0, "verified_at": "2026-07-10"},
    "propietario_balanceado": {"provider": "anthropic", "model_id": "claude-sonnet-4-6", "input_per_million": 3.0, "output_per_million": 15.0, "verified_at": "2026-07-10"},
    "propietario_economico": {"provider": "openai", "model_id": "gpt-4o-mini", "input_per_million": 0.15, "output_per_million": 0.60, "verified_at": "2026-08-26"},
    "openai_razonamiento": {"provider": "openai", "model_id": "gpt-5.6-luna", "input_per_million": 0.20, "output_per_million": 1.20, "verified_at": "2026-09-18"},
    "juez_economico": {"provider": "anthropic", "model_id": "claude-haiku-4-5", "input_per_million": 1.0, "output_per_million": 5.0, "verified_at": "2026-07-10"},
    "open_weight_local": {"provider": "ollama", "model_id": "llama3.1:8b", "input_per_million": 0.0, "output_per_million": 0.0, "verified_at": LOCAL},
    "base_local": {"provider": "transformers", "model_id": "openai-community/gpt2", "input_per_million": 0.0, "output_per_million": 0.0, "verified_at": LOCAL},
    "open_weight_pequeno": {"provider": "ollama", "model_id": "qwen3:1.7b", "input_per_million": 0.0, "output_per_million": 0.0, "verified_at": LOCAL},
    "open_weight_razonamiento": {"provider": "ollama", "model_id": "gpt-oss:20b", "input_per_million": 0.0, "output_per_million": 0.0, "verified_at": LOCAL},
    # Fuera de la tabla del curso: precio de developers.openai.com/api/docs/pricing (<272K contexto), 2026-09-19.
    "extra_gpt55": {"provider": "openai", "model_id": "gpt-5.5", "input_per_million": 5.0, "output_per_million": 30.0, "verified_at": "2026-09-19 (fuera de la tabla del curso)"},
}


def cost_breakdown(model_key: str, input_tokens: int | None, output_tokens: int | None) -> tuple[float, float] | None:
    if input_tokens is None or output_tokens is None:
        return None
    price = PRICES[model_key]
    return (
        input_tokens / 1_000_000 * price["input_per_million"],
        output_tokens / 1_000_000 * price["output_per_million"],
    )


def cost_usd(model_key: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """output_tokens debe ser el total del proveedor (razonamiento + visibles)."""
    if input_tokens is None or output_tokens is None:
        return None
    price = PRICES[model_key]
    return (
        input_tokens / 1_000_000 * price["input_per_million"]
        + output_tokens / 1_000_000 * price["output_per_million"]
    )
