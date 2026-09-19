import argparse

from src.aggregation import latest_per_case, part1_table
from src.config import RESULTS_PATH
from src.models.registry import get_runner
from src.pricing import cost_usd
from src.results import call_key, completed_keys, read_results
from src.runner import run_case
from src.schemas import load_cases
from src.sweeps import (
    DECODING_EXPERIMENT,
    DECODING_MODEL,
    MAX_OUTPUT_TOKENS,
    RUNS,
    TOPK_EXPERIMENT,
    TOPK_MODEL,
    decoding_cells,
    topk_cells,
)

SWEEPS = {
    "decoding": (DECODING_EXPERIMENT, DECODING_MODEL, decoding_cells),
    "topk": (TOPK_EXPERIMENT, TOPK_MODEL, topk_cells),
}


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
            "top_k": call["top_k"], "effort": None, "prompt_variant": "base",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Parte 2.b: barrido de decoding (temperature x top_p) y de top_k.")
    parser.add_argument("--sweep", choices=[*SWEEPS, "all"], default="all")
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--limit-cases", type=int, help="usar solo los primeros N casos oficiales (pilotos)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = load_cases()[: args.limit_cases]
    rows = read_results(RESULTS_PATH)
    done = completed_keys(rows)

    for name in SWEEPS if args.sweep == "all" else [args.sweep]:
        experiment, model_key, cells = SWEEPS[name]
        calls = plan(model_key, experiment, cells(), cases, args.runs)
        pending = [c for c in calls if key_of(c) not in done]
        print(f"[{name}] model={model_key}  {len(calls)} calls planned, {len(calls) - len(pending)} already done, {len(pending)} to run")

        if args.dry_run:
            print(estimate(model_key, len(pending), rows))
            continue

        runner = get_runner(model_key, max_output_tokens=MAX_OUTPUT_TOKENS)
        for i, call in enumerate(pending, 1):
            run_case(
                runner, model_key, call["case"], part="2b", experiment=experiment, run=call["run"],
                temperature=call["temperature"], top_p=call["top_p"], top_k=call["top_k"],
            )
            if i % 50 == 0 or i == len(pending):
                print(f"  {i}/{len(pending)}")


if __name__ == "__main__":
    main()
