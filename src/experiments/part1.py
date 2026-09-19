import argparse
from pathlib import Path

from src.config import RAW_DIR, RESULTS_PATH
from src.metrics import accuracy, mean_latency, total_cost
from src.models.base import ModelRunner
from src.models.mock_runner import MockRunner
from src.pricing import PRICES
from src.runner import run_case
from src.schemas import Case, load_cases

MOCK_RESULTS_PATH = RAW_DIR / "mock_results.jsonl"


def run(
    runner: ModelRunner,
    cases: list[Case],
    model_key: str = "mock",
    results_path: Path | None = None,
) -> list[dict]:
    return [
        run_case(
            runner, model_key, case, part="1", experiment="model_comparison", results_path=results_path
        )
        for case in cases
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parte 1: clasificación de tickets con un modelo.")
    parser.add_argument("--model", default="mock", choices=["mock", *PRICES])
    parser.add_argument("--limit", type=int, help="usar solo los primeros N casos oficiales")
    args = parser.parse_args()

    cases = load_cases()[: args.limit]
    if args.model == "mock":
        runner, path = MockRunner(['{"category": "billing"}']), MOCK_RESULTS_PATH
    else:
        from src.models.registry import get_runner

        runner, path = get_runner(args.model), RESULTS_PATH

    records = run(runner, cases, args.model, path)
    for r in records:
        line = f"{r['case_id']}  {r['status']:<5} expected={r['expected']:<9} predicted={str(r['predicted']):<9} correct={r['correct']}"
        if r["status"] == "error":
            line += f"  {r['error_type']} (http {r['http_status']}): {r['error_message']}"
        print(line)
    print(f"model={args.model}  cases={len(records)}  accuracy={accuracy(records):.2f}  -> {path}")
    if any(r["status"] == "ok" for r in records):
        print(f"mean latency={mean_latency(records):.2f}s  total cost=${total_cost(records):.6f}")


if __name__ == "__main__":
    main()
