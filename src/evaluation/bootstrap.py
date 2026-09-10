"""Grouped bootstrap utilities."""

from __future__ import annotations

import numpy as np


def grouped_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(groups == group) for group in sampled])


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    values = np.asarray(values, dtype=float)
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))
