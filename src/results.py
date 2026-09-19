import json
from pathlib import Path

from src.config import RESULTS_PATH


def append_result(record: dict, path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_results(path: Path = RESULTS_PATH) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


KEY_FIELDS = ("model_id", "experiment", "case_id", "run", "temperature", "top_p", "top_k", "effort", "prompt_variant")


def call_key(row: dict) -> tuple:
    return tuple(row.get(field) for field in KEY_FIELDS)


def completed_keys(rows: list[dict]) -> set:
    return {call_key(r) for r in rows if r.get("status") == "ok"}
