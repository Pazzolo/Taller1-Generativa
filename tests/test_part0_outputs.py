import json

import pytest

from src.config import CATEGORIES, RAW_DIR


def load(name):
    path = RAW_DIR / name
    if not path.exists():
        pytest.skip(f"{name} no existe: correr `uv run -m src.experiments.part0`")
    return json.loads(path.read_text(encoding="utf-8"))


def by_prefix(records):
    out = {}
    for r in records:
        out.setdefault(r["prefix_id"], []).append(r)
    return out


def test_part0a_entropy_and_nucleus_never_decrease_with_temperature():
    for entries in by_prefix(load("part0a.json")["records"]).values():
        entries.sort(key=lambda r: r["temperature"])
        assert [r["entropy_bits"] for r in entries] == sorted(r["entropy_bits"] for r in entries)
        assert [r["nucleus_size_p90"] for r in entries] == sorted(r["nucleus_size_p90"] for r in entries)


def test_part0a_top_tokens_keep_their_order_and_probabilities_are_valid():
    for r in load("part0a.json")["records"]:
        assert len(r["top_tokens"]) == len(r["top_probs"]) == 15
        assert r["top_probs"] == sorted(r["top_probs"], reverse=True)
        assert 0 < sum(r["top_probs"]) <= 1 + 1e-9


def test_part0a_chosen_high_confidence_prefix_is_the_lowest_entropy_candidate():
    data = load("part0a.json")
    candidates = [s for s in data["prefix_scan"] if s["prefix"] != data["prefixes"]["low_confidence"]]
    best = min(candidates, key=lambda s: s["entropy_bits_t1"])
    assert best["prefix"] == data["prefixes"]["high_confidence"]


def test_part0a_the_two_prefixes_really_differ_in_confidence_at_t1():
    records = {(r["prefix_id"], r["temperature"]): r for r in load("part0a.json")["records"]}
    assert records[("high_confidence", 1.0)]["entropy_bits"] < records[("low_confidence", 1.0)]["entropy_bits"] / 2


def test_part0b_records_every_test_the_plan_asks_for():
    b = load("part0b.json")
    assert b["test1_do_sample_false_ignores_temperature"]["identical"] in (True, False)
    assert len(b["test2_top_k_1"]["runs"]) == 5
    assert len(b["test3_top_k_vs_top_p"]["panels"]) == 2
    assert b["test4_degeneration"]["raw_output"]


def test_part0b_top_k_1_reproduces_greedy_for_every_seed():
    test2 = load("part0b.json")["test2_top_k_1"]
    assert test2["all_equal_to_greedy"] is test2["all_identical_to_each_other"] is True
    assert len({r["seed"] for r in test2["runs"]}) == 5


def test_part0b_top_k_and_top_p_sets_differ_on_both_prefixes_in_opposite_directions():
    panels = {p["prefix_id"]: p for p in load("part0b.json")["test3_top_k_vs_top_p"]["panels"]}
    assert all(p["sets_differ"] for p in panels.values())
    assert panels["high_confidence"]["top_p_size"] < panels["high_confidence"]["top_k_size"]
    assert panels["low_confidence"]["top_p_size"] > panels["low_confidence"]["top_k_size"]


def test_part0b_membership_labels_are_consistent_with_the_set_sizes():
    for p in load("part0b.json")["test3_top_k_vs_top_p"]["panels"]:
        assert sum(m in ("both", "top_k_only") for m in p["membership"]) == p["top_k_size"]


def test_part0c_gpt2_base_never_produces_a_valid_classification():
    c = load("part0c.json")
    assert c["cases"] == 10 and len(c["records"]) == 10
    assert c["valid_schema"] == 0 and c["correct"] == 0
    assert all(r["predicted"] is None or r["predicted"] not in CATEGORIES for r in c["records"])
