import pytest

from src.pricing import PRICES, cost_usd


def test_cost_matches_manual_calculation():
    assert cost_usd("propietario_economico", 1_000_000, 1_000_000) == pytest.approx(0.15 + 0.60)
    assert cost_usd("propietario_grande", 2_000, 500) == pytest.approx(2_000 * 5 / 1e6 + 500 * 25 / 1e6)


def test_missing_token_counts_yield_none_not_a_guess():
    assert cost_usd("propietario_economico", None, 10) is None
    assert cost_usd("propietario_economico", 10, None) is None


def test_unknown_model_key_raises():
    with pytest.raises(KeyError):
        cost_usd("does_not_exist", 1, 1)


def test_every_price_row_has_a_verification_date():
    for row in PRICES.values():
        assert row["verified_at"]
        assert row["input_per_million"] > 0 and row["output_per_million"] > 0
