import argparse

from src.config import CONTAMINATED_PATH, RESULTS_PATH
from src.contamination import EXPERIMENT
from src.models.registry import get_runner
from src.planning import estimate, key_of, plan
from src.results import completed_keys, read_results
from src.runner import run_case
from src.schemas import load_cases

MODEL = "openai_razonamiento"
LEVELS = ("low", "high")
EXTRA_LEVELS = ("xhigh",)  # comprobación adicional con el esfuerzo máximo aceptado; el plan pide solo low y high
REPORT_LEVELS = LEVELS + EXTRA_LEVELS
RUNS = 10
MAX_OUTPUT_TOKENS = 16000


def main() -> None:
    parser = argparse.ArgumentParser(description="Parte 4.b: casos contaminados con esfuerzo bajo y alto.")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--levels", nargs="+", default=list(LEVELS), choices=("none", "low", "medium", "high", "xhigh"))
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = load_cases(CONTAMINATED_PATH, split="contaminated")
    cells = [{"temperature": None, "top_p": None, "top_k": None, "effort": level} for level in args.levels]
    calls = plan(args.model, EXPERIMENT, cells, cases, args.runs)
    rows = read_results(RESULTS_PATH)
    done = completed_keys(rows)
    pending = [c for c in calls if key_of(c) not in done]
    print(f"model={args.model}  cases={len(cases)}  levels={args.levels}  {len(calls)} calls planned, {len(calls) - len(pending)} already done, {len(pending)} to run")

    if args.dry_run:
        print(estimate(args.model, len(pending), rows))
        return

    runner = get_runner(args.model, max_output_tokens=MAX_OUTPUT_TOKENS)
    for i, call in enumerate(pending, 1):
        record = run_case(
            runner, args.model, call["case"], part="4b", experiment=EXPERIMENT, run=call["run"], effort=call["effort"],
        )
        detail = record["error_message"][:120] if record["status"] == "error" else (
            f"predicted={record['predicted']} expected={record['expected']} reasoning={record['reasoning_tokens']}"
        )
        print(f"  {i}/{len(pending)}  {call['case'].id} effort={call['effort']:<5} run={call['run']:<2} {record['status']}  {detail}")


if __name__ == "__main__":
    main()
