"""Empirical CDF computation."""

from __future__ import annotations

import numpy as np


def compute_ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted unique values and ECDF probabilities."""
    sorted_vals = np.sort(values.astype(float))
    n = len(sorted_vals)
    if n == 0:
        return np.array([]), np.array([])
    y = np.arange(1, n + 1) / n
    return sorted_vals, y


def empirical_quantile(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, q))
