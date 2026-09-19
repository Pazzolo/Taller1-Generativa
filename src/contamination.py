from collections import Counter

from src.metrics import accuracy

EXPERIMENT = "contaminated"

CONTAMINATED_COLUMNS = [
    "case_id", "trap", "effort", "runs", "errors", "expected", "correct_rate", "predictions",
    "reasoning_tokens_mean", "reasoning_tokens_max", "output_tokens_mean",
]
RUN_COLUMNS = ["case_id", "trap", "effort", "run", "expected", "prediction", "correct", "reasoning_tokens", "output_tokens", "raw_output"]


def cell_rows(rows: list[dict], model: str, case_id: str, effort: str) -> list[dict]:
    """Filas de una celda (caso, nivel); si una llamada se repitió (corrida) gana la última."""
    latest = {}
    for r in rows:
        if r.get("part") == "4b" and r["model_id"] == model and r["experiment"] == EXPERIMENT and r["case_id"] == case_id and r["effort"] == effort:
            latest[r["run"]] = r
    return [latest[k] for k in sorted(latest)]


def contaminated_table(rows: list[dict], model: str, cases: list, levels: tuple[str, ...]) -> list[dict]:
    table = []
    for case in cases:
        for level in levels:
            records = cell_rows(rows, model, case.id, level)
            if not records:
                continue
            ok = [r for r in records if r["status"] == "ok"]
            counts = Counter(str(r["predicted"]) for r in ok)
            reasoning = [r["reasoning_tokens"] for r in ok if r["reasoning_tokens"] is not None]
            output = [r["output_tokens"] for r in ok if r["output_tokens"] is not None]
            table.append(
                {
                    "case_id": case.id,
                    "trap": case.trap,
                    "effort": level,
                    "runs": len(records),
                    "errors": len(records) - len(ok),
                    "expected": case.expected.category,
                    "correct_rate": accuracy(ok) if ok else None,
                    "predictions": "; ".join(f"{label}:{n}" for label, n in counts.most_common()),
                    "reasoning_tokens_mean": sum(reasoning) / len(reasoning) if reasoning else None,
                    "reasoning_tokens_max": max(reasoning) if reasoning else None,
                    "output_tokens_mean": sum(output) / len(output) if output else None,
                }
            )
    return table


def runs_table(rows: list[dict], model: str, cases: list, levels: tuple[str, ...]) -> list[dict]:
    table = []
    for case in cases:
        for level in levels:
            for r in cell_rows(rows, model, case.id, level):
                ok = r["status"] == "ok"
                table.append(
                    {
                        "case_id": case.id, "trap": case.trap, "effort": level, "run": r["run"],
                        "expected": case.expected.category,
                        "prediction": r["predicted"] if ok else None,
                        "correct": r["correct"] if ok else None,
                        "reasoning_tokens": r["reasoning_tokens"] if ok else None,
                        "output_tokens": r["output_tokens"] if ok else None,
                        "raw_output": r["raw_output"] if ok else r["error_message"],
                    }
                )
    return table
