import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aggregation import PART1_COLUMNS, PART3_COLUMNS, load_notes, part1_table, part3_table, write_csv
from src.config import PART1_NOTES_PATH, TABLES_DIR
from src.experiments.part3 import MODEL as PART3_MODEL, VARIANTS as PART3_VARIANTS
from src.experiments.part4a import LEVELS as PART4A_LEVELS, MODEL as PART4A_MODEL
from src.prompts import PUZZLE_ANSWER
from src.reasoning import CONTROL_COLUMNS, DELTA_COLUMNS, EFFORT_COLUMNS, control_table, deltas_table, effort_table
from src.exposure import part2a_table, write_part2a_csv
from src.results import read_results
from src.sweeps import DECODING_COLUMNS, TOPK_COLUMNS, decoding_table, topk_table


def main() -> None:
    rows = read_results()
    write_part2a_csv(part2a_table(rows), TABLES_DIR / "part2a.csv")
    write_csv(decoding_table(rows), TABLES_DIR / "part2b_decoding.csv", DECODING_COLUMNS)
    write_csv(topk_table(rows), TABLES_DIR / "part2b_topk.csv", TOPK_COLUMNS)
    write_csv(part3_table(rows, PART3_MODEL, PART3_VARIANTS), TABLES_DIR / "part3.csv", PART3_COLUMNS)
    efforts = effort_table(rows, PART4A_MODEL, PART4A_LEVELS)
    write_csv(efforts, TABLES_DIR / "part4a.csv", EFFORT_COLUMNS)
    write_csv(deltas_table(efforts), TABLES_DIR / "part4a_deltas.csv", DELTA_COLUMNS)
    write_csv(control_table(rows, PART4A_MODEL, PART4A_LEVELS, PUZZLE_ANSWER), TABLES_DIR / "part4a_control.csv", CONTROL_COLUMNS)
    table = part1_table(rows, load_notes(PART1_NOTES_PATH))
    path = TABLES_DIR / "part1.csv"
    write_csv(table, path)

    print(f"Parte 1 -> {path}  ({len(table)} modelos)")
    for row in table:
        print(
            "  "
            + "  ".join(
                f"{col}={row[col]:.4g}" if isinstance(row[col], float) else f"{col}={row[col]}"
                for col in PART1_COLUMNS
                if col not in ("model_name", "qualitative_notes")
            )
        )


if __name__ == "__main__":
    main()
