import csv

import pytest

from src.aggregation import PART1_COLUMNS, latest_per_case, part1_table, write_csv


def row(model, case, correct=True, status="ok", part="1", latency=1.0, cost=0.001, parse_ok=True):
    base = {"part": part, "model_id": model, "case_id": case, "status": status, "correct": correct, "parse_ok": parse_ok}
    if status == "ok":
        base.update(latency_seconds=latency, input_tokens=50, output_tokens=5, cost_usd=cost)
    return base


def test_latest_row_per_case_wins():
    rows = [row("m", "c1", correct=False), row("m", "c2"), row("m", "c1", correct=True)]
    result = latest_per_case(rows)
    assert len(result) == 2
    assert next(r for r in result if r["case_id"] == "c1")["correct"] is True


def test_smoke_run_duplicate_is_not_double_counted():
    rows = [row("propietario_economico", "c1"), row("propietario_economico", "c1"), row("propietario_economico", "c2")]
    (entry,) = part1_table(rows)
    assert entry["cases"] == 2
    assert entry["cost_usd_total"] == pytest.approx(0.002)


def test_table_has_one_row_per_model_with_all_columns():
    rows = [row("propietario_economico", "c1"), row("propietario_grande", "c1", correct=False)]
    table = part1_table(rows)
    assert [t["model"] for t in table] == ["propietario_grande", "propietario_economico"]
    assert all(set(t) == set(PART1_COLUMNS) for t in table)


def test_errors_count_as_incorrect_but_do_not_pollute_latency():
    rows = [row("propietario_economico", "c1", latency=2.0), row("propietario_economico", "c2", status="error", correct=False, parse_ok=False)]
    (entry,) = part1_table(rows)
    assert entry["errors"] == 1
    assert entry["accuracy"] == 0.5
    assert entry["parse_rate"] == 0.5
    assert entry["latency_mean"] == 2.0


def test_model_with_only_errors_yields_none_stats_without_crashing():
    rows = [row("propietario_economico", "c1", status="error", correct=False, parse_ok=False)]
    (entry,) = part1_table(rows)
    assert entry["accuracy"] == 0.0
    assert entry["latency_mean"] is None and entry["input_tokens_mean"] is None


def test_other_parts_and_mock_rows_are_excluded():
    rows = [row("propietario_economico", "c1", part="2b"), row("mock", "c1")]
    assert part1_table(rows) == []


def test_price_verified_at_comes_from_pricing_table():
    (entry,) = part1_table([row("propietario_economico", "c1")])
    assert entry["price_verified_at"] == "2026-08-26"


def test_write_csv_roundtrip(tmp_path):
    table = part1_table([row("propietario_economico", "c1")])
    path = tmp_path / "out" / "part1.csv"
    write_csv(table, path)
    with open(path, encoding="utf-8") as f:
        (loaded,) = list(csv.DictReader(f))
    assert list(loaded) == PART1_COLUMNS
    assert loaded["model"] == "propietario_economico" and float(loaded["accuracy"]) == 1.0
