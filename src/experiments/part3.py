import argparse

from src.config import RESULTS_PATH
from src.models.registry import get_runner
from src.planning import estimate, key_of, plan
from src.results import completed_keys, read_results
from src.runner import run_case
from src.schemas import load_cases

MODEL = "propietario_economico"
VARIANTS = ("zero_shot", "few_shot", "cot", "structured")
RUNS = 5
MAX_OUTPUT_TOKENS = 512
EXPERIMENT = "prompt_variants"


def main() -> None:
    parser = argparse.ArgumentParser(description="Parte 3: zero-shot, few-shot, CoT y salida estructurada.")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--limit-cases", type=int, help="usar solo los primeros N casos oficiales (pilotos)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = load_cases()[: args.limit_cases]
    cells = [{"temperature": None, "top_p": None, "top_k": None, "prompt_variant": v} for v in args.variants]
    calls = plan(args.model, EXPERIMENT, cells, cases, args.runs)
    rows = read_results(RESULTS_PATH)
    done = completed_keys(rows)
    pending = [c for c in calls if key_of(c) not in done]
    print(f"model={args.model}  variants={args.variants}  {len(calls)} calls planned, {len(calls) - len(pending)} already done, {len(pending)} to run")

    if args.dry_run:
        print(estimate(args.model, len(pending), rows))
        print("(basado en las medias de la Parte 1: few-shot y CoT costarán más)")
        return

    runner = get_runner(args.model, max_output_tokens=MAX_OUTPUT_TOKENS)
    for i, call in enumerate(pending, 1):
        run_case(
            runner, args.model, call["case"], part="3", experiment=EXPERIMENT, run=call["run"],
            prompt_variant=call["prompt_variant"],
        )
        if i % 20 == 0 or i == len(pending):
            print(f"  {i}/{len(pending)}")


if __name__ == "__main__":
    main()
