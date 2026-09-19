from src.metrics import accuracy, mean_latency, mean_output_tokens, total_cost
from src.pricing import PRICES

EXPERIMENT = "reasoning_effort"
CONTROL_EXPERIMENT = "reasoning_control"

EFFORT_COLUMNS = [
    "model", "effort", "calls", "errors", "accuracy", "reasoning_tokens_mean", "visible_output_tokens_mean",
    "total_output_tokens_mean", "latency_mean", "cost_usd_total", "cost_usd_per_call", "error_message",
]
DELTA_COLUMNS = [
    "from_effort", "to_effort", "accuracy_delta", "reasoning_tokens_delta", "cost_delta_usd", "cost_per_accuracy_point_usd",
]
CONTROL_COLUMNS = ["model", "effort", "calls", "errors", "reasoning_tokens_mean", "total_output_tokens_mean", "answer_correct_rate"]


def latest_rows(rows: list[dict], model: str, experiment: str, effort: str) -> list[dict]:
    """Filas de un nivel; si una llamada se repitió (caso, corrida) gana la última."""
    latest = {}
    for r in rows:
        if r.get("part") == "4a" and r["model_id"] == model and r["experiment"] == experiment and r["effort"] == effort:
            latest[(r["case_id"], r["run"])] = r
    return list(latest.values())


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def effort_table(rows: list[dict], model: str, levels: tuple[str, ...], experiment: str = EXPERIMENT) -> list[dict]:
    table = []
    for level in levels:
        records = latest_rows(rows, model, experiment, level)
        if not records:
            continue
        ok = [r for r in records if r["status"] == "ok"]
        errors = [r for r in records if r["status"] == "error"]
        reasoning = [r["reasoning_tokens"] for r in ok]
        total = [r["output_tokens"] for r in ok]
        visible = [t - r_ for t, r_ in zip(total, reasoning) if t is not None and r_ is not None]
        table.append(
            {
                "model": model,
                "effort": level,
                "calls": len(records),
                "errors": len(errors),
                "accuracy": accuracy(ok) if ok else None,
                "reasoning_tokens_mean": _mean(reasoning),
                "visible_output_tokens_mean": _mean(visible),
                "total_output_tokens_mean": _mean(total),
                "latency_mean": mean_latency(ok) if ok else None,
                "cost_usd_total": total_cost(ok) if ok else None,
                "cost_usd_per_call": (total_cost(ok) / len(ok)) if ok else None,
                "error_message": errors[0]["error_message"] if errors else "",
            }
        )
    return table


def deltas_table(table: list[dict]) -> list[dict]:
    """Saltos entre niveles consecutivos con datos; el costo por punto de exactitud solo existe si la exactitud sube."""
    usable = [t for t in table if t["accuracy"] is not None]
    deltas = []
    for a, b in zip(usable, usable[1:]):
        accuracy_delta = b["accuracy"] - a["accuracy"]
        cost_delta = b["cost_usd_per_call"] - a["cost_usd_per_call"]
        deltas.append(
            {
                "from_effort": a["effort"],
                "to_effort": b["effort"],
                "accuracy_delta": accuracy_delta,
                "reasoning_tokens_delta": b["reasoning_tokens_mean"] - a["reasoning_tokens_mean"],
                "cost_delta_usd": cost_delta,
                "cost_per_accuracy_point_usd": cost_delta / (accuracy_delta * 100) if accuracy_delta > 0 else None,
            }
        )
    return deltas


def control_table(rows: list[dict], model: str, levels: tuple[str, ...], answer: str) -> list[dict]:
    table = []
    for level in levels:
        records = latest_rows(rows, model, CONTROL_EXPERIMENT, level)
        if not records:
            continue
        ok = [r for r in records if r["status"] == "ok"]
        table.append(
            {
                "model": model,
                "effort": level,
                "calls": len(records),
                "errors": len(records) - len(ok),
                "reasoning_tokens_mean": _mean([r["reasoning_tokens"] for r in ok]),
                "total_output_tokens_mean": mean_output_tokens(ok) if ok else None,
                "answer_correct_rate": (sum(r["raw_output"].strip() == answer for r in ok) / len(ok)) if ok else None,
            }
        )
    return table
