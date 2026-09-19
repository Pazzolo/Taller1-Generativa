import numpy as np


def temperature_probs(logits, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max()
    exp = np.exp(scaled)
    return exp / exp.sum()


def entropy_bits(probs) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def ranked(probs) -> np.ndarray:
    """Índices de tokens de mayor a menor probabilidad (los empates conservan el orden del vocabulario)."""
    return np.argsort(-np.asarray(probs), kind="stable")


def nucleus_size(probs, p: float = 0.9) -> int:
    """Menor número de tokens cuya masa acumulada es >= p."""
    cumulative = np.cumsum(np.asarray(probs, dtype=np.float64)[ranked(probs)])
    return int(np.searchsorted(cumulative, p - 1e-12) + 1)


def top_k_indices(probs, k: int) -> set[int]:
    return {int(i) for i in ranked(probs)[:k]}


def top_p_indices(probs, p: float) -> set[int]:
    return {int(i) for i in ranked(probs)[: nucleus_size(probs, p)]}
