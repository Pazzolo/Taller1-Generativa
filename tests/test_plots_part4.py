import pytest

pytest.importorskip("matplotlib")

from src.plots_part4 import describe_range, group_points, plot_accuracy_vs_reasoning, plot_cost_vs_accuracy  # noqa: E402


def test_group_points_merges_identical_coordinates_and_joins_labels_in_order():
    grouped = group_points([(0.0, 1.0, "none"), (0.0, 1.0, "low"), (0.4, 1.0, "high")])
    assert grouped == [(0.0, 1.0, "none, low"), (0.4, 1.0, "high")]


def test_group_points_ignores_float_noise():
    assert len(group_points([(0.1 + 0.2, 1.0, "a"), (0.3, 1.0, "b")])) == 1


def test_describe_range_states_a_flat_result_plainly():
    assert describe_range(1.0, 1.0) == "Exactitud 1.00 en todos los niveles"
    assert describe_range(0.8, 1.0) == "Exactitud entre 0.80 y 1.00"


def tickets():
    return [
        {"effort": "none", "accuracy": 1.0, "reasoning_tokens_mean": 0.0, "cost_usd_per_call": 2.0e-5},
        {"effort": "low", "accuracy": 1.0, "reasoning_tokens_mean": 0.0, "cost_usd_per_call": 2.0e-5},
        {"effort": "high", "accuracy": 1.0, "reasoning_tokens_mean": 0.4, "cost_usd_per_call": 2.2e-5},
    ]


def test_both_figures_are_written(tmp_path):
    control = [{"effort": "none", "answer_correct_rate": 1.0, "reasoning_tokens_mean": 0.0},
               {"effort": "low", "answer_correct_rate": 1.0, "reasoning_tokens_mean": 26.0}]
    plot_accuracy_vs_reasoning(tickets(), control, tmp_path / "a.png")
    plot_cost_vs_accuracy(tickets(), tmp_path / "b.png")
    assert (tmp_path / "a.png").stat().st_size > 5_000 and (tmp_path / "b.png").stat().st_size > 5_000
