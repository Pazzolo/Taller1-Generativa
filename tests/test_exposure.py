import csv

import pytest

from src.exposure import PART2A_COLUMNS, PARAMETERS, assess, latest_batch, part2a_table, write_part2a_csv


def row(model="propietario_economico", param="temperature", value=0.0, run=1, text="a", status="ok",
        http=None, message="", ts="2026-09-19T10:00:00-05:00"):
    r = {"part": "2a", "model_id": model, "run": run, "status": status, "timestamp": ts,
         "temperature": None, "top_p": None, "top_k": None}
    r[param] = value
    if status == "ok":
        r["raw_output"] = text
    else:
        r.update(http_status=http, error_message=message, raw_output=None)
    return r


def batch(texts, **kw):
    return [row(run=i, text=t, **kw) for i, t in enumerate(texts, 1)]


def test_acts_when_high_setting_is_more_diverse_than_low():
    result = assess(batch(["a"] * 5, value=0.0), batch(list("abcde"), value=2.0))
    assert result["observed_state"] == "accepted_and_acts"
    assert (result["distinct_low"], result["distinct_high"]) == (1, 5)


def test_accepted_but_does_not_act_when_diversity_is_the_same():
    result = assess(batch(["a"] * 5), batch(["a"] * 5))
    assert result["observed_state"] == "accepted_and_does_not_act"


def test_ignored_parameter_with_random_outputs_at_both_ends_does_not_act():
    result = assess(batch(list("abcde")), batch(list("vwxyz")))
    assert result["observed_state"] == "accepted_and_does_not_act"


def test_http_400_is_a_rejection_with_literal_message():
    err = row(status="error", http=400, message="Unrecognized request argument supplied: top_k")
    result = assess([err], [err])
    assert result["observed_state"] == "rejected"
    assert result["http_status"] == 400
    assert result["error_message"] == "Unrecognized request argument supplied: top_k"


def test_one_rejected_end_is_enough_to_mark_rejected():
    err = row(status="error", http=400, message="out of range")
    assert assess(batch(["a"] * 5), [err])["observed_state"] == "rejected"


@pytest.mark.parametrize("http", [None, 429, 500, 503])
def test_connection_and_server_errors_are_inconclusive_not_rejections(http):
    err = row(status="error", http=http, message="boom")
    assert assess([err], [err])["observed_state"] == "inconclusive"


def test_whitespace_differences_do_not_count_as_diversity():
    assert assess(batch(["a"] * 5), batch(["a", " a", "a ", "a\n", "a"]))["observed_state"] == "accepted_and_does_not_act"


def test_latest_batch_ignores_earlier_repetitions_of_the_experiment():
    old = batch(["x", "y", "z", "w", "v"])
    new = batch(["a", "a"])
    assert [r["raw_output"] for r in latest_batch(old + new)] == ["a", "a"]


def make_rows(model="propietario_economico", param="temperature"):
    low, high = PARAMETERS[param]
    return batch(["a"] * 5, model=model, param=param, value=low) + batch(list("abcde"), model=model, param=param, value=high)


def test_table_has_one_row_per_tested_model_and_parameter():
    rows = make_rows(param="temperature") + make_rows(param="top_p")
    table = part2a_table(rows)
    assert [(t["model"], t["parameter"]) for t in table] == [("propietario_economico", "temperature"), ("propietario_economico", "top_p")]
    assert all(set(t) == set(PART2A_COLUMNS) for t in table)


def test_table_reports_declared_observed_and_date():
    (entry,) = part2a_table(make_rows())
    assert entry["declared"] == "sí"
    assert entry["observed_state"] == "accepted_and_acts"
    assert entry["verified_at"] == "2026-09-19"
    assert (entry["low_value"], entry["high_value"]) == PARAMETERS["temperature"]


def test_declared_no_but_observed_rejected_is_visible_side_by_side():
    err = row(param="top_k", value=1, status="error", http=400, message="Unrecognized request argument supplied: top_k")
    err_high = row(param="top_k", value=100, status="error", http=400, message="Unrecognized request argument supplied: top_k")
    (entry,) = part2a_table([err, err_high])
    assert (entry["declared"], entry["observed_state"], entry["http_status"]) == ("no", "rejected", 400)


def test_other_parts_are_ignored():
    rows = make_rows()
    for r in rows:
        r["part"] = "1"
    assert part2a_table(rows) == []


def test_write_csv_roundtrip(tmp_path):
    path = tmp_path / "t" / "part2a.csv"
    write_part2a_csv(part2a_table(make_rows()), path)
    with open(path, encoding="utf-8") as f:
        (loaded,) = list(csv.DictReader(f))
    assert list(loaded) == PART2A_COLUMNS
    assert loaded["observed_state"] == "accepted_and_acts"
