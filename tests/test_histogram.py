"""Tests for histogram without zeros."""

from __future__ import annotations

import numpy as np

from hospital_demand_analyzer.plots.histogram_plot import _conditional_mass
from scipy import stats


def test_conditional_poisson_mass():
    lam = 3.0
    norm = _conditional_mass(stats.poisson, (lam,), positive_only=True)
    assert abs(norm - (1 - np.exp(-lam))) < 1e-9


def test_positive_only_data_has_no_zero_bar():
    data = np.array([0, 0, 1, 2, 2, 3, 0, 5], dtype=float)
    pos = data[data > 0]
    vals, counts = np.unique(pos, return_counts=True)
    assert 0 not in vals
    assert counts.sum() == len(pos)
