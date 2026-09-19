from src.aggregation import latest_per_case
from src.pricing import cost_usd
from src.results import call_key


def plan(model_key: str, experiment: str, cells: list[dict], cases, runs: int) -> list[dict]:
    return [
        {"model_key": model_key, "experiment": experiment, "case": case, "run": run, **cell}
        for cell in cells
        for case in cases
        for run in range(1, runs + 1)
    ]


def key_of(call: dict) -> tuple:
    return call_key(
        {
            "model_id": call["model_key"], "experiment": call["experiment"], "case_id": call["case"].id,
            "run": call["run"], "temperature": call["temperature"], "top_p": call["top_p"],
            "top_k": call["top_k"], "effort": call.get("effort"), "prompt_variant": call.get("prompt_variant", "base"),
        }
    )


def estimate(model_key: str, calls: int, rows: list[dict]) -> str:
    part1 = [r for r in rows if r.get("part") == "1" and r.get("model_id") == model_key and r["status"] == "ok"]
    if not part1:
        return "estimated tokens/cost: n/a (no Part 1 rows for this model)"
    part1 = latest_per_case(part1)
    tokens_in = sum(r["input_tokens"] for r in part1) / len(part1) * calls
    tokens_out = sum(r["output_tokens"] for r in part1) / len(part1) * calls
    cost = cost_usd(model_key, tokens_in, tokens_out)
    return f"estimated input tokens: {tokens_in:,.0f}\nestimated output tokens: {tokens_out:,.0f}\nestimated cost: ${cost:.4f}"
