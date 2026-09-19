import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PARTS = ("0", "1", "2a", "2b", "3", "4a", "4b")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta los experimentos del Taller 01.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--part", choices=PARTS)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    for part in PARTS if args.all else (args.part,):
        importlib.import_module(f"src.experiments.part{part}").main()


if __name__ == "__main__":
    main()
