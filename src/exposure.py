from src.aggregation import write_csv
from src.config import COURSE_MODELS

PARAMETERS = {
    "temperature": (0.0, 2.0),
    "top_p": (0.01, 1.0),
    "top_k": (1, 100),
}

# "Declarado" = lo que el proveedor documenta. None de esto sustituye a la observación.
# Campo parametros_expuestos de la tabla semestral (anexo del enunciado, 2026-09-18), con su vocabulario:
# si, no, no_en_esta_fila, solo_valor_por_defecto, sin_verificar.
DECLARED = {
    "propietario_economico": {"temperature": "si", "top_p": "si", "top_k": "no"},
    "openai_razonamiento": {"temperature": "no_en_esta_fila", "top_p": "no_en_esta_fila", "top_k": "no"},
    "open_weight_pequeno": {"temperature": "si", "top_p": "si", "top_k": "si"},
    "extra_gpt55": {"temperature": "no documentado", "top_p": "no documentado", "top_k": "no documentado"},
}

REJECTION_STATUSES = {400, 422}

PART2A_COLUMNS = [
    "model",
    "parameter",
    "declared",
    "observed_state",
    "http_status",
    "error_message",
    "verified_at",
    "low_value",
    "high_value",
    "distinct_low",
    "distinct_high",
    "runs_low",
    "runs_high",
]


def latest_batch(rows: list[dict]) -> list[dict]:
    """Filas desde la última corrida con run == 1 (una repetición del experimento reemplaza a la anterior)."""
    start = max((i for i, r in enumerate(rows) if r["run"] == 1), default=0)
    return rows[start:]


def distinct_outputs(rows: list[dict]) -> int:
    return len({r["raw_output"].strip() for r in rows if r["status"] == "ok"})


def assess(low_rows: list[dict], high_rows: list[dict]) -> dict:
    errors = [r for r in low_rows + high_rows if r["status"] == "error"]
    rejections = [r for r in errors if r.get("http_status") in REJECTION_STATUSES]
    distinct_low, distinct_high = distinct_outputs(low_rows), distinct_outputs(high_rows)

    if rejections:
        state, cause = "rejected", rejections[0]
    elif errors:
        state, cause = "inconclusive", errors[0]
    elif distinct_high > distinct_low:
        state, cause = "accepted_and_acts", None
    else:
        state, cause = "accepted_and_does_not_act", None

    return {
        "observed_state": state,
        "http_status": cause["http_status"] if cause else None,
        "error_message": cause["error_message"] if cause else "",
        "distinct_low": distinct_low,
        "distinct_high": distinct_high,
        "runs_low": len(low_rows),
        "runs_high": len(high_rows),
    }


def part2a_table(rows: list[dict], models: tuple[str, ...] = COURSE_MODELS) -> list[dict]:
    rows = [r for r in rows if r.get("part") == "2a"]
    table = []
    for model_key in models:
        for parameter, (low, high) in PARAMETERS.items():
            def setting(value):
                return latest_batch(
                    [r for r in rows if r["model_id"] == model_key and r.get(parameter) == value]
                )

            low_rows, high_rows = setting(low), setting(high)
            if not low_rows and not high_rows:
                continue
            verified = max(r["timestamp"] for r in low_rows + high_rows)[:10]
            table.append(
                {
                    "model": model_key,
                    "parameter": parameter,
                    "declared": DECLARED.get(model_key, {}).get(parameter, "no documentado"),
                    **assess(low_rows, high_rows),
                    "verified_at": verified,
                    "low_value": low,
                    "high_value": high,
                }
            )
    return table


def write_part2a_csv(table: list[dict], path) -> None:
    write_csv(table, path, PART2A_COLUMNS)
