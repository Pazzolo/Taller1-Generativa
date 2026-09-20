"""Punto de entrada para reproducir los experimentos.

`--part X` corre todas las ejecuciones que hacen falta para reproducir los resultados de esa parte
(las mismas que se usaron en el informe); `--all` corre las siete. `--dry-run` solo cuenta y estima,
sin llamar a ninguna API, en las partes que lo soportan (2.a a 4.b).
"""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import COURSE_MODELS

PARTS = ("0", "1", "2a", "2b", "3", "4a", "4b")

# Invocaciones canónicas (argumentos de línea de comandos) de cada parte.
PLAN = {
    "0": [[]],
    "1": [["--model", model] for model in COURSE_MODELS],
    "2a": [[]],
    "2b": [[]],
    "3": [[]],
    "4a": [[], ["--control"]],
    "4b": [[], ["--levels", "low", "high"]],
}
SUPPORTS_DRY_RUN = {"2a", "2b", "3", "4a", "4b"}


def invocations(part: str, dry_run: bool) -> list[list[str]]:
    extra = ["--dry-run"] if dry_run and part in SUPPORTS_DRY_RUN else []
    return [[*args, *extra] for args in PLAN[part]]


def run_part(part: str, dry_run: bool = False) -> None:
    if dry_run and part not in SUPPORTS_DRY_RUN:
        print(f"[parte {part}] --dry-run no aplica (no llama a APIs de pago o no tiene estimación): se omite")
        return
    module = importlib.import_module(f"src.experiments.part{part}")
    for argv in invocations(part, dry_run):
        print(f"[parte {part}] {' '.join(argv) or '(sin argumentos)'}")
        sys.argv = [f"part{part}", *argv]  # cada experimento lee su propia línea de comandos
        module.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta los experimentos del Taller 01.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--part", choices=PARTS)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="cuenta y estima sin llamar a la API")
    args = parser.parse_args()

    for part in PARTS if args.all else (args.part,):
        run_part(part, args.dry_run)


if __name__ == "__main__":
    main()
