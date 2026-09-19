import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PLOTS_DIR
from src.experiments.part4a import LEVELS, MODEL
from src.plots_part4 import plot_accuracy_vs_reasoning, plot_cost_vs_accuracy
from src.prompts import PUZZLE_ANSWER
from src.reasoning import control_table, effort_table
from src.results import read_results


def main() -> None:
    rows = read_results()
    tickets = [t for t in effort_table(rows, MODEL, LEVELS) if t["accuracy"] is not None]
    control = control_table(rows, MODEL, LEVELS, PUZZLE_ANSWER)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_accuracy_vs_reasoning(tickets, control, PLOTS_DIR / "part4_accuracy_vs_reasoning_tokens.png")
    plot_cost_vs_accuracy(tickets, PLOTS_DIR / "part4_cost_vs_accuracy.png")
    print("Parte 4 -> outputs/plots/part4_accuracy_vs_reasoning_tokens.png, part4_cost_vs_accuracy.png")


if __name__ == "__main__":
    main()
