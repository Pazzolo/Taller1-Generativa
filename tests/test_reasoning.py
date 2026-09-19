import pytest

from src.aggregation import write_csv
from src.reasoning import CONTROL_EXPERIMENT, EFFORT_COLUMNS, EXPERIMENT, control_table, deltas_table, effort_table

MODEL = "openai_razonamiento"
LEVELS = ("none", "low", "medium", "high", "xhigh", "max")


def row(effort="low", case="c1", run=1, correct=True, reasoning=0, out=8, status="ok", experiment=EXPERIMENT, text="x",
        latency=1.0, cost=0.0001, message="boom"):
    r = {"part": "4a", "model_id": MODEL, "experiment": experiment, "effort": effort, "case_id": case, "run": run,
         "status": status, "correct": correct}
    if status == "ok":
        r.update(reasoning_tokens=reasoning, output_tokens=out, latency_seconds=latency, cost_usd=cost, raw_output=text)
    else:
        r.update(error_message=message, raw_output=None)
    return r


def test_table_splits_reasoning_from_visible_output_tokens():
    (entry,) = effort_table([row(reasoning=40, out=48)], MODEL, LEVELS)
    assert (entry["reasoning_tokens_mean"], entry["visible_output_tokens_mean"], entry["total_output_tokens_mean"]) == (40, 8, 48)
    assert list(entry) == EFFORT_COLUMNS


def test_levels_come_out_in_the_requested_order():
    rows = [row("high"), row("none"), row("low")]
    assert [t["effort"] for t in effort_table(rows, MODEL, LEVELS)] == ["none", "low", "high"]


def test_rejected_level_reports_the_literal_error_and_no_stats():
    rows = [row("low"), row("max", status="error", correct=False, message="Unsupported value: 'max'")]
    table = {t["effort"]: t for t in effort_table(rows, MODEL, LEVELS)}
    assert table["max"]["errors"] == 1 and table["max"]["accuracy"] is None
    assert table["max"]["error_message"] == "Unsupported value: 'max'"
    assert table["low"]["errors"] == 0


def test_repeated_call_keeps_the_latest_row():
    (entry,) = effort_table([row(correct=False), row(correct=True)], MODEL, LEVELS)
    assert (entry["calls"], entry["accuracy"]) == (1, 1.0)


def test_other_experiments_and_models_are_excluded():
    other = row(experiment=CONTROL_EXPERIMENT)
    assert effort_table([other, {**row(), "model_id": "propietario_grande"}], MODEL, LEVELS) == []


def test_deltas_compare_consecutive_levels_and_skip_rejected_ones():
    rows = [row("none", correct=False, reasoning=0, cost=0.0001), row("low", correct=True, reasoning=20, cost=0.0003),
            row("max", status="error", correct=False)]
    deltas = deltas_table(effort_table(rows, MODEL, LEVELS))
    assert len(deltas) == 1
    d = deltas[0]
    assert (d["from_effort"], d["to_effort"]) == ("none", "low")
    assert d["accuracy_delta"] == 1.0 and d["reasoning_tokens_delta"] == 20
    assert d["cost_delta_usd"] == pytest.approx(0.0002)
    assert d["cost_per_accuracy_point_usd"] == pytest.approx(0.0002 / 100)


def test_cost_per_accuracy_point_is_undefined_when_accuracy_does_not_improve():
    rows = [row("none", cost=0.0001), row("low", cost=0.0003)]
    (d,) = deltas_table(effort_table(rows, MODEL, LEVELS))
    assert d["accuracy_delta"] == 0 and d["cost_per_accuracy_point_usd"] is None
    assert d["cost_delta_usd"] > 0


def test_control_table_scores_the_puzzle_answer_exactly():
    rows = [row("low", experiment=CONTROL_EXPERIMENT, text="5", reasoning=25, out=35),
            row("low", case="c2", experiment=CONTROL_EXPERIMENT, text="10", reasoning=27, out=37)]
    (entry,) = control_table(rows, MODEL, LEVELS, "5")
    assert entry["answer_correct_rate"] == 0.5 and entry["reasoning_tokens_mean"] == 26


def test_csv_roundtrip_with_none_values(tmp_path):
    table = effort_table([row("low"), row("max", status="error", correct=False)], MODEL, LEVELS)
    write_csv(table, tmp_path / "t.csv", EFFORT_COLUMNS)
    assert (tmp_path / "t.csv").read_text(encoding="utf-8").count("\n") == 3
