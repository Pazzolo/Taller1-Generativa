import pytest

from src.metrics import stability
from src.sweeps import (
    DECODING_COLUMNS,
    TOPK_COLUMNS,
    decoding_cells,
    decoding_table,
    greedy_match,
    topk_cells,
    topk_table,
)

MODEL = "propietario_economico"
LOCAL = "open_weight_pequeno"


def row(case="c1", run=1, predicted="billing", expected="billing", status="ok", model=MODEL, experiment="temperature_top_p",
        temperature=0.0, top_p=1.0, top_k=None, raw=None, out_tokens=6, valid=True):
    r = {"part": "2b", "model_id": model, "experiment": experiment, "case_id": case, "run": run, "status": status,
         "temperature": temperature, "top_p": top_p, "top_k": top_k}
    if status == "ok":
        r.update(predicted=predicted if valid else None, valid_schema=valid, parse_ok=valid, correct=valid and predicted == expected,
                 raw_output=raw if raw is not None else '{"category": "%s"}' % predicted, output_tokens=out_tokens)
    else:
        r.update(predicted=None, valid_schema=False, parse_ok=False, correct=False, raw_output=None)
    return r


def test_grids_have_the_sizes_the_plan_requires():
    assert len(decoding_cells()) == 15
    assert len(topk_cells()) == 4  # baseline greedy + top_k 1, 5, 40


def test_stability_is_share_of_the_modal_answer_averaged_over_cases():
    rows = [row("c1", r, "billing") for r in range(1, 6)] + [
        row("c2", 1, "billing"), row("c2", 2, "billing"), row("c2", 3, "billing"), row("c2", 4, "account"), row("c2", 5, "account"),
    ]
    assert stability(rows) == pytest.approx((1.0 + 0.6) / 2)


def test_stability_treats_invalid_outputs_as_one_answer_and_skips_errors():
    rows = [row("c1", 1, "billing"), row("c1", 2, valid=False), row("c1", 3, valid=False), row("c1", 4, status="error")]
    assert stability(rows) == pytest.approx(2 / 3)


def test_decoding_table_summarizes_one_cell():
    rows = [row("c1", r, "billing") for r in range(1, 6)] + [row("c2", r, "account", expected="billing") for r in range(1, 6)]
    (entry,) = decoding_table(rows)
    assert (entry["temperature"], entry["top_p"], entry["calls"], entry["errors"]) == (0.0, 1.0, 10, 0)
    assert entry["accuracy"] == 0.5
    assert entry["stability"] == 1.0
    assert entry["mean_output_tokens"] == 6
    assert list(entry) == DECODING_COLUMNS


def test_cells_are_kept_apart_by_temperature_and_top_p():
    rows = [row(temperature=0.0, top_p=0.5), row(temperature=1.5, top_p=0.5, predicted="account")]
    table = decoding_table(rows)
    assert [(t["temperature"], t["top_p"], t["accuracy"]) for t in table] == [(0.0, 0.5, 1.0), (1.5, 0.5, 0.0)]


def test_repeated_call_keeps_only_the_latest_row_per_case_and_run():
    rows = [row("c1", 1, "account"), row("c1", 1, "billing")]
    (entry,) = decoding_table(rows)
    assert (entry["calls"], entry["accuracy"]) == (1, 1.0)


def test_errors_count_as_incorrect_but_not_in_stability_or_tokens():
    rows = [row("c1", 1), row("c1", 2, status="error")]
    (entry,) = decoding_table(rows)
    assert (entry["calls"], entry["errors"], entry["accuracy"]) == (2, 1, 0.5)
    assert entry["stability"] == 1.0 and entry["mean_output_tokens"] == 6


def test_other_models_and_experiments_are_excluded():
    rows = [row(model="propietario_grande"), row(experiment="top_k")]
    assert decoding_table(rows) == []


def topk_rows(topk1_outputs, greedy_outputs):
    greedy = [row("c1", i, raw=o, model=LOCAL, experiment="top_k", temperature=0.0, top_p=None) for i, o in enumerate(greedy_outputs, 1)]
    topk1 = [row("c1", i, raw=o, model=LOCAL, experiment="top_k", temperature=0.7, top_p=None, top_k=1) for i, o in enumerate(topk1_outputs, 1)]
    return greedy, topk1


def test_greedy_match_true_when_top_k_1_reproduces_greedy_exactly():
    greedy, topk1 = topk_rows(['{"category": "billing"}'] * 5, ['{"category": "billing"}'] * 5)
    assert greedy_match(topk1, greedy) == 1.0


def test_greedy_match_false_if_any_run_differs():
    greedy, topk1 = topk_rows(['{"category": "billing"}'] * 4 + ['{"category": "account"}'], ['{"category": "billing"}'] * 5)
    assert greedy_match(topk1, greedy) == 0.0


def test_greedy_match_is_none_without_data():
    assert greedy_match([], []) is None


def test_topk_table_has_baseline_and_only_top_k_1_gets_greedy_match():
    greedy, topk1 = topk_rows(['{"category": "billing"}'] * 5, ['{"category": "billing"}'] * 5)
    top5 = [row("c1", 1, model=LOCAL, experiment="top_k", temperature=0.7, top_p=None, top_k=5)]
    table = topk_table(greedy + topk1 + top5)
    assert [(t["top_k"], t["temperature"]) for t in table] == [(None, 0.0), (1, 0.7), (5, 0.7)]
    assert [t["greedy_match"] for t in table] == [None, 1.0, None]
    assert list(table[0]) == TOPK_COLUMNS
