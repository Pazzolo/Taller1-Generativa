import pytest

from src.metrics import (
    accuracy,
    mean_input_tokens,
    mean_latency,
    mean_output_tokens,
    median_latency,
    parse_rate,
    total_cost,
)

ROWS = [
    {"correct": True, "parse_ok": True, "latency_seconds": 1.0, "input_tokens": 50, "output_tokens": 5, "cost_usd": 0.001},
    {"correct": False, "parse_ok": True, "latency_seconds": 3.0, "input_tokens": 60, "output_tokens": 7, "cost_usd": 0.002},
    {"correct": False, "parse_ok": False, "latency_seconds": 8.0, "input_tokens": 70, "output_tokens": 30, "cost_usd": None},
    {"correct": True, "parse_ok": True},
]


def test_accuracy_and_parse_rate_use_all_rows():
    assert accuracy(ROWS) == 0.5
    assert parse_rate(ROWS) == 0.75


def test_means_ignore_missing_values():
    assert mean_latency(ROWS) == pytest.approx(4.0)
    assert mean_input_tokens(ROWS) == pytest.approx(60)
    assert mean_output_tokens(ROWS) == pytest.approx(14)


def test_median_latency_differs_from_mean_with_outlier():
    assert median_latency(ROWS) == 3.0


def test_total_cost_skips_none_and_missing():
    assert total_cost(ROWS) == pytest.approx(0.003)
