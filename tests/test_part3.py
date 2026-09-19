import pytest

from src.aggregation import PART3_COLUMNS, part3_table
from src.metrics import format_compliance
from src.pricing import cost_breakdown, cost_usd

MODEL = "propietario_economico"
VARIANTS = ("zero_shot", "few_shot", "cot", "structured")


def row(variant="zero_shot", case="c1", run=1, correct=True, valid=True, parse_ok=True, status="ok", inp=100, out=10):
    r = {"part": "3", "model_id": MODEL, "prompt_variant": variant, "case_id": case, "run": run, "status": status,
         "correct": correct, "valid_schema": valid, "parse_ok": parse_ok}
    if status == "ok":
        r.update(input_tokens=inp, output_tokens=out)
    return r


def test_cost_breakdown_adds_up_to_total_cost():
    parts = cost_breakdown(MODEL, 2_000, 500)
    assert parts == pytest.approx((2_000 * 0.15 / 1e6, 500 * 0.60 / 1e6))
    assert sum(parts) == pytest.approx(cost_usd(MODEL, 2_000, 500))
    assert cost_breakdown(MODEL, None, 5) is None


def test_format_compliance_is_independent_of_correctness():
    rows = [row(correct=False), row(correct=True), row(valid=False, parse_ok=False, correct=False)]
    assert format_compliance(rows) == pytest.approx(2 / 3)


def test_table_has_one_row_per_variant_in_the_requested_order():
    rows = [row("cot"), row("zero_shot")]
    table = part3_table(rows, MODEL, VARIANTS)
    assert [t["prompt_variant"] for t in table] == ["zero_shot", "cot"]
    assert all(list(t) == PART3_COLUMNS for t in table)


def test_input_and_output_costs_are_reported_separately():
    (entry,) = part3_table([row(inp=1000, out=200), row(case="c2", inp=1000, out=200)], MODEL, VARIANTS)
    assert entry["cost_input_usd"] == pytest.approx(2 * 1000 * 0.15 / 1e6)
    assert entry["cost_output_usd"] == pytest.approx(2 * 200 * 0.60 / 1e6)
    assert entry["cost_usd_total"] == pytest.approx(entry["cost_input_usd"] + entry["cost_output_usd"])
    assert (entry["input_tokens_mean"], entry["output_tokens_mean"]) == (1000, 200)


def test_valid_format_with_wrong_category_lowers_accuracy_not_format_compliance():
    rows = [row(case="c1"), row(case="c2", correct=False)]
    (entry,) = part3_table(rows, MODEL, VARIANTS)
    assert (entry["accuracy"], entry["format_compliance"]) == (0.5, 1.0)


def test_repeated_calls_keep_the_latest_and_errors_are_excluded_from_tokens():
    rows = [row(correct=False), row(correct=True), row(case="c2", status="error", correct=False, valid=False, parse_ok=False)]
    (entry,) = part3_table(rows, MODEL, VARIANTS)
    assert (entry["calls"], entry["errors"], entry["accuracy"]) == (2, 1, 0.5)
    assert entry["input_tokens_mean"] == 100


def test_other_parts_and_models_are_ignored():
    other = row()
    other["part"] = "1"
    assert part3_table([other, {**row(), "model_id": "propietario_grande"}], MODEL, VARIANTS) == []
