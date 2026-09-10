"""Small dependency-light ranking helpers."""

from __future__ import annotations

import numpy as np


def reciprocal_rank(relevance: np.ndarray) -> float:
    positions = np.flatnonzero(np.asarray(relevance, dtype=bool))
    return float(1.0 / (positions[0] + 1)) if len(positions) else 0.0


def hits_at_k(relevance: np.ndarray, k: int) -> float:
    return float(np.asarray(relevance, dtype=bool)[:k].any())


def ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    rel = np.asarray(relevance, dtype=float)[:k]
    discounts = np.log2(np.arange(2, len(rel) + 2))
    dcg = float((rel / discounts).sum())
    ideal = np.sort(rel)[::-1]
    idcg = float((ideal / discounts).sum())
    return dcg / idcg if idcg else 0.0
