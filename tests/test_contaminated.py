import pytest

from src.aggregation import write_csv
from src.config import CATEGORIES, CONTAMINATED_PATH
from src.contamination import CONTAMINATED_COLUMNS, RUN_COLUMNS, contaminated_table, runs_table
from src.schemas import load_cases

MODEL = "openai_razonamiento"
LEVELS = ("low", "high")
CASES = load_cases(CONTAMINATED_PATH, split="contaminated")

KEYWORDS = {
    "billing": ("charged", "payment", "invoice", "refund"),
    "technical": ("crash", "glitch", "servers", "freezing"),
    "account": ("profile", "email address", "password"),
}


def test_there_are_exactly_three_contaminated_cases_one_per_trap():
    assert len(CASES) == 3
    assert {c.trap for c in CASES} == {"distraction", "misleading_frame", "spurious_correlation"}
    assert [c.id for c in CASES] == ["contaminated_01", "contaminated_02", "contaminated_03"]


def test_ground_truth_is_valid_and_each_category_appears_once():
    assert sorted(c.expected.category for c in CASES) == sorted(CATEGORIES)


def test_contaminated_cases_do_not_overlap_the_official_or_debug_cases():
    used = {c.ticket.strip().lower() for c in load_cases(split=None)}
    ids = {c.id for c in load_cases(split=None)}
    assert all(c.ticket.strip().lower() not in used and c.id not in ids for c in CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.trap)
def test_each_trap_is_real_the_ticket_carries_cues_for_a_wrong_category(case):
    text = case.ticket.lower()
    assert any(k in text for k in KEYWORDS[case.expected.category]), "falta la señal de la categoría correcta"
    wrong = [c for c in CATEGORIES if c != case.expected.category]
    assert any(k in text for c in wrong for k in KEYWORDS[c]), "no hay señales de una categoría incorrecta"


def row(case_id="contaminated_01", effort="low", run=1, predicted="account", correct=True, reasoning=0, out=8, status="ok"):
    r = {"part": "4b", "model_id": MODEL, "experiment": "contaminated", "case_id": case_id, "effort": effort, "run": run, "status": status}
    if status == "ok":
        r.update(predicted=predicted, correct=correct, reasoning_tokens=reasoning, output_tokens=out, raw_output='{"category": "%s"}' % predicted)
    else:
        r.update(error_message="boom", raw_output=None)
    return r


def test_table_reports_rate_prediction_counts_and_reasoning_per_case_and_level():
    rows = [row(run=1), row(run=2), row(run=3, predicted="billing", correct=False, reasoning=30, out=40),
            row(effort="high", run=1, reasoning=12, out=20)]
    table = contaminated_table(rows, MODEL, CASES, LEVELS)
    low, high = table
    assert (low["effort"], low["runs"], low["expected"]) == ("low", 3, "account")
    assert low["correct_rate"] == pytest.approx(2 / 3)
    assert low["predictions"] == "account:2; billing:1"
    assert (low["reasoning_tokens_mean"], low["reasoning_tokens_max"]) == (10, 30)
    assert (high["effort"], high["correct_rate"], high["reasoning_tokens_mean"]) == ("high", 1.0, 12)
    assert all(list(t) == CONTAMINATED_COLUMNS for t in table)


def test_repeated_run_keeps_the_latest_row():
    table = contaminated_table([row(run=1, correct=False, predicted="billing"), row(run=1)], MODEL, CASES, LEVELS)
    assert (table[0]["runs"], table[0]["correct_rate"]) == (1, 1.0)


def test_errors_are_counted_and_do_not_break_the_stats():
    table = contaminated_table([row(run=1), row(run=2, status="error")], MODEL, CASES, LEVELS)
    assert (table[0]["runs"], table[0]["errors"], table[0]["correct_rate"]) == (2, 1, 1.0)


def test_cell_with_only_errors_has_no_rate():
    (entry,) = contaminated_table([row(status="error")], MODEL, CASES, LEVELS)
    assert entry["correct_rate"] is None and entry["reasoning_tokens_mean"] is None


def test_runs_table_has_one_row_per_call_with_the_plans_columns():
    rows = [row(run=1), row(run=2, predicted="billing", correct=False)]
    table = runs_table(rows, MODEL, CASES, LEVELS)
    assert len(table) == 2 and all(list(t) == RUN_COLUMNS for t in table)
    assert [t["correct"] for t in table] == [True, False]


def test_other_parts_and_models_are_excluded():
    other = {**row(), "part": "4a"}
    assert contaminated_table([other, {**row(), "model_id": "propietario_grande"}], MODEL, CASES, LEVELS) == []


def test_csv_roundtrip(tmp_path):
    write_csv(contaminated_table([row(), row(status="error", run=2)], MODEL, CASES, LEVELS), tmp_path / "t.csv", CONTAMINATED_COLUMNS)
    assert (tmp_path / "t.csv").read_text(encoding="utf-8").count("\n") == 2
