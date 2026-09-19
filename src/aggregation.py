import csv
from pathlib import Path

from src.metrics import (
    accuracy,
    mean_input_tokens,
    mean_latency,
    mean_output_tokens,
    median_latency,
    parse_rate,
    total_cost,
)
from src.pricing import PRICES

PART1_COLUMNS = [
    "model",
    "model_name",
    "cases",
    "errors",
    "accuracy",
    "latency_mean",
    "latency_p50",
    "input_tokens_mean",
    "output_tokens_mean",
    "cost_usd_total",
    "price_verified_at",
    "parse_rate",
    "qualitative_notes",
]


def latest_per_case(rows: list[dict]) -> list[dict]:
    """Última fila por caso: una corrida de humo previa no cuenta doble."""
    latest = {row["case_id"]: row for row in rows}
    return list(latest.values())


def part1_table(rows: list[dict]) -> list[dict]:
    part1 = [r for r in rows if r.get("part") == "1" and r.get("model_id") in PRICES]
    table = []
    for model_key in PRICES:
        model_rows = latest_per_case([r for r in part1 if r["model_id"] == model_key])
        if not model_rows:
            continue
        ok_rows = [r for r in model_rows if r["status"] == "ok"]
        table.append(
            {
                "model": model_key,
                "model_name": PRICES[model_key]["model_id"],
                "cases": len(model_rows),
                "errors": len(model_rows) - len(ok_rows),
                "accuracy": accuracy(model_rows),
                "latency_mean": mean_latency(ok_rows) if ok_rows else None,
                "latency_p50": median_latency(ok_rows) if ok_rows else None,
                "input_tokens_mean": mean_input_tokens(ok_rows) if ok_rows else None,
                "output_tokens_mean": mean_output_tokens(ok_rows) if ok_rows else None,
                "cost_usd_total": total_cost(model_rows),
                "price_verified_at": PRICES[model_key]["verified_at"],
                "parse_rate": parse_rate(model_rows),
                "qualitative_notes": "",
            }
        )
    return table


def write_csv(table: list[dict], path: Path, columns: list[str] = PART1_COLUMNS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
