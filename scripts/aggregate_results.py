import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aggregation import PART1_COLUMNS, part1_table, write_csv
from src.config import TABLES_DIR
from src.exposure import part2a_table, write_part2a_csv
from src.results import read_results


def main() -> None:
    rows = read_results()
    write_part2a_csv(part2a_table(rows), TABLES_DIR / "part2a.csv")
    table = part1_table(rows)
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
