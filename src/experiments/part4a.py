import argparse

from src.config import RESULTS_PATH
from src.models.registry import get_runner
from src.planning import estimate, key_of, plan
from src.results import completed_keys, read_results
from src.runner import run_case
from src.reasoning import CONTROL_EXPERIMENT
from src.schemas import Case, Expected, load_cases

MODEL = "openai_razonamiento"
# La tabla del curso lista también "max", pero la API lo rechaza para este modelo (400): queda registrado en el piloto.
LEVELS = ("none", "low", "medium", "high", "xhigh", "max")
DEFAULT_LEVELS = LEVELS[:-1]
RUNS = 3
CONTROL_RUNS = 3
# Tope de seguridad: en un modelo de razonamiento incluye los tokens de razonamiento.
MAX_OUTPUT_TOKENS = 16000
EXPERIMENT = "reasoning_effort"
# Caso auxiliar: el control usa el acertijo, no un ticket; los campos de verificación de esas filas no significan nada.
CONTROL_CASE = Case(id="puzzle_01", ticket="", expected=Expected(category="billing"), split="debug")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parte 4.a: exactitud y costo frente al esfuerzo de razonamiento.")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS), choices=LEVELS)
    parser.add_argument("--control", action="store_true", help="acertijo de control: comprueba que el dial sí cambia los tokens de razonamiento")
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--limit-cases", type=int, help="usar solo los primeros N casos oficiales (pilotos)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiment, prompt_variant = (CONTROL_EXPERIMENT, "puzzle") if args.control else (EXPERIMENT, "base")
    cases = [CONTROL_CASE] if args.control else load_cases()[: args.limit_cases]
    runs = CONTROL_RUNS if args.control and args.runs == RUNS else args.runs
    cells = [{"temperature": None, "top_p": None, "top_k": None, "effort": level, "prompt_variant": prompt_variant} for level in args.levels]
    calls = plan(args.model, experiment, cells, cases, runs)
    rows = read_results(RESULTS_PATH)
    done = completed_keys(rows)
    pending = [c for c in calls if key_of(c) not in done]
    print(f"model={args.model}  levels={args.levels}  {len(calls)} calls planned, {len(calls) - len(pending)} already done, {len(pending)} to run")

    if args.dry_run:
        print(estimate(args.model, len(pending), rows))
        print("(sin filas de la Parte 1 para este modelo no hay estimación: hacer primero el piloto)")
        return

    runner = get_runner(args.model, max_output_tokens=MAX_OUTPUT_TOKENS)
    for i, call in enumerate(pending, 1):
        record = run_case(
            runner, args.model, call["case"], part="4a", experiment=experiment, run=call["run"], effort=call["effort"],
            prompt_variant=prompt_variant,
        )
        detail = record["error_message"][:120] if record["status"] == "error" else (
            f"reasoning={record['reasoning_tokens']} out={record['output_tokens']} correct={record['correct']}"
        )
        print(f"  {i}/{len(pending)}  effort={call['effort']:<7} {call['case'].id}  {record['status']}  {detail}")


if __name__ == "__main__":
    main()
