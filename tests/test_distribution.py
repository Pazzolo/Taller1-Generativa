import numpy as np
import pytest

from src.distribution import entropy_bits, nucleus_size, ranked, temperature_probs, top_k_indices, top_p_indices


def test_probs_sum_to_one_and_keep_the_ranking_for_any_temperature():
    logits = np.array([3.0, 1.0, 0.5, -2.0])
    for t in (0.1, 0.7, 1.0, 1.5, 2.0):
        p = temperature_probs(logits, t)
        assert p.sum() == pytest.approx(1.0)
        assert list(ranked(p)) == [0, 1, 2, 3]


def test_low_temperature_sharpens_and_high_temperature_flattens():
    logits = np.array([3.0, 1.0, 0.5, -2.0])
    assert temperature_probs(logits, 0.1)[0] > temperature_probs(logits, 1.0)[0] > temperature_probs(logits, 2.0)[0]


def test_temperature_one_matches_a_plain_softmax():
    logits = np.array([2.0, 1.0, 0.1])
    expected = np.exp(logits) / np.exp(logits).sum()
    assert temperature_probs(logits, 1.0) == pytest.approx(expected)


def test_softmax_is_stable_with_huge_logits():
    p = temperature_probs(np.array([1000.0, 999.0]), 1.0)
    assert np.isfinite(p).all() and p.sum() == pytest.approx(1.0)


def test_entropy_of_known_distributions():
    assert entropy_bits([1.0, 0.0, 0.0]) == 0.0
    assert entropy_bits([0.5, 0.5]) == pytest.approx(1.0)
    assert entropy_bits(np.full(8, 1 / 8)) == pytest.approx(3.0)


def test_entropy_rises_with_temperature():
    logits = np.array([4.0, 2.0, 1.0, 0.0, -1.0])
    values = [entropy_bits(temperature_probs(logits, t)) for t in (0.1, 0.7, 1.0, 1.5, 2.0)]
    assert values == sorted(values) and values[0] < values[-1]


def test_nucleus_size_is_the_minimum_prefix_reaching_p():
    probs = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    assert nucleus_size(probs, 0.5) == 1
    assert nucleus_size(probs, 0.9) == 3
    assert nucleus_size(probs, 0.91) == 4
    assert nucleus_size(probs, 1.0) == 5


def test_nucleus_does_not_depend_on_input_order():
    probs = np.array([0.05, 0.5, 0.05, 0.3, 0.1])
    assert nucleus_size(probs, 0.9) == 3
    assert top_p_indices(probs, 0.9) == {1, 3, 4}


def test_top_k_and_top_p_can_select_different_sets():
    confident = np.array([0.97, 0.01, 0.01, 0.005, 0.005])
    flat = np.full(20, 0.05)
    assert top_k_indices(confident, 3) != top_p_indices(confident, 0.9)
    assert len(top_p_indices(confident, 0.9)) == 1
    assert len(top_p_indices(flat, 0.9)) == 18
    assert top_k_indices(flat, 5) < top_p_indices(flat, 0.9) or len(top_k_indices(flat, 5)) == 5


def test_top_p_matches_the_transformers_warper():
    """La definición propia debe coincidir con la que aplica generate()."""
    torch = pytest.importorskip("torch")
    generation = pytest.importorskip("transformers.generation")
    rng = np.random.default_rng(0)
    for _ in range(25):
        logits = rng.normal(size=200) * rng.uniform(0.5, 4)
        probs = temperature_probs(logits, 1.0)
        for p in (0.5, 0.9):
            warper = generation.TopPLogitsWarper(top_p=p)
            kept = torch.isfinite(warper(None, torch.tensor(logits)[None].clone())[0]).nonzero().flatten().tolist()
            assert set(kept) == top_p_indices(probs, p)


def test_top_k_matches_the_transformers_warper():
    torch = pytest.importorskip("torch")
    generation = pytest.importorskip("transformers.generation")
    rng = np.random.default_rng(1)
    for _ in range(25):
        logits = rng.normal(size=200)
        warper = generation.TopKLogitsWarper(top_k=5)
        kept = torch.isfinite(warper(None, torch.tensor(logits)[None].clone())[0]).nonzero().flatten().tolist()
        assert set(kept) == top_k_indices(temperature_probs(logits, 1.0), 5)


def test_most_confident_prefix_is_the_one_with_lowest_entropy():
    from src.experiments.part0 import most_confident

    assert most_confident({"a": 8.6, "b": 2.1, "c": 5.0}) == "b"
