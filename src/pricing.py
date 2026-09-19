PRICES = {
    # Sustituye a claude-opus-4-8 de la tabla del curso; precio de developers.openai.com/api/docs/pricing (<272K contexto), no de la tabla del curso.
    "propietario_grande": {
        "provider": "openai",
        "model_id": "gpt-5.5",
        "input_per_million": 5.0,
        "output_per_million": 30.0,
        "verified_at": "2026-09-19",
    },
    "propietario_balanceado": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "input_per_million": 3.0,
        "output_per_million": 15.0,
        "verified_at": "2026-07-10",
    },
    "propietario_economico": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "input_per_million": 0.15,
        "output_per_million": 0.60,
        "verified_at": "2026-08-26",
    },
    "open_weight_pequeno": {
        "provider": "ollama",
        "model_id": "qwen3:1.7b",
        "input_per_million": 0.0,
        "output_per_million": 0.0,
        "verified_at": "n/a (local, sin precio por token)",
    },
    "openai_razonamiento": {
        "provider": "openai",
        "model_id": "gpt-5.6-luna",
        "input_per_million": 0.20,
        "output_per_million": 1.20,
        "verified_at": "2026-09-18",
    },
}


def cost_usd(model_key: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """output_tokens debe ser el total del proveedor (razonamiento + visibles)."""
    if input_tokens is None or output_tokens is None:
        return None
    price = PRICES[model_key]
    return (
        input_tokens / 1_000_000 * price["input_per_million"]
        + output_tokens / 1_000_000 * price["output_per_million"]
    )
