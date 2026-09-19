import argparse

from src.config import RESULTS_PATH, TABLES_DIR
from src.exposure import PARAMETERS, part2a_table, write_part2a_csv
from src.models.registry import get_runner
from src.results import read_results
from src.runner import run_case
from src.schemas import Case, Expected

MODELS = ("propietario_grande", "propietario_economico", "open_weight_pequeno")
DEFAULT_RUNS = 5
PROBE_CASE = Case(
    id="probe_01",
    ticket="I was charged twice for the same subscription.",
    expected=Expected(category="billing"),
    split="debug",
)


def planned_calls(models, runs: int) -> int:
    return len(models) * len(PARAMETERS) * 2 * runs


def run_setting(runner, model_key: str, parameter: str, value, runs: int) -> None:
    """Repite la llamada `runs` veces; ante un error se detiene, pues repetirlo no aporta información."""
    for run in range(1, runs + 1):
        record = run_case(
            runner, model_key, PROBE_CASE, part="2a", experiment="param_exposure",
            prompt_variant="probe", run=run, **{parameter: value},
        )
        if record["status"] == "error":
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Parte 2.a: matriz de exposición de parámetros.")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"{planned_calls(args.models, args.runs)} calls planned (máximo; un error detiene las repeticiones del ajuste)")
        print(f"models={args.models}  parameters={list(PARAMETERS)}  runs={args.runs}")
        return

    for model_key in args.models:
        runner = get_runner(model_key)
        for parameter, (low, high) in PARAMETERS.items():
            for value in (low, high):
                run_setting(runner, model_key, parameter, value, args.runs)

    table = part2a_table(read_results(RESULTS_PATH))
    path = TABLES_DIR / "part2a.csv"
    write_part2a_csv(table, path)
    for row in table:
        print(
            f"{row['model']:<22} {row['parameter']:<12} declared={row['declared']:<15} "
            f"observed={row['observed_state']:<26} http={row['http_status']}  "
            f"distinct low/high={row['distinct_low']}/{row['distinct_high']}  {row['error_message'][:90]}"
        )
    print(f"-> {path}")


if __name__ == "__main__":
    main()
