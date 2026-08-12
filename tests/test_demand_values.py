"""Tests for demand value preparation."""

from __future__ import annotations

import numpy as np

from hospital_demand_analyzer.domain.demand_values import (
    CONTINUOUS,
    DISCRETE,
    prepare_demand_values,
    sanitize_demand,
)
from hospital_demand_analyzer.statistics.fitter import fit_all


def test_sanitize_demand_keeps_nonneg_and_drops_negative():
    values = np.array([-1, 0, 2.0, 3.5, -0.1])
    cleaned = sanitize_demand(values)
    assert np.array_equal(cleaned, np.array([0.0, 2.0, 3.5]))


def test_prepare_demand_values_default_includes_zeros():
    values = np.array([0, 0, 1, 2, -5])
    result = prepare_demand_values(values, include_zeros=True)
    assert np.array_equal(result, np.array([0.0, 0.0, 1.0, 2.0]))


def test_prepare_demand_values_can_exclude_zeros():
    values = np.array([0, 0, 1, 2])
    result = prepare_demand_values(values, include_zeros=False)
    assert np.array_equal(result, np.array([1.0, 2.0]))


def test_fit_all_discrete_mode_uses_discrete_distributions():
    data = np.array([0, 0, 1, 1, 2, 3], dtype=float)
    results = fit_all(data, include_zeros=True, value_mode=DISCRETE)
    assert results
    assert all(r.dist_type == "discrete" for r in results)


def test_fit_all_continuous_mode_uses_continuous_distributions():
    data = np.array([0, 0, 1.2, 1.5, 2.1, 3.0], dtype=float)
    results = fit_all(data, include_zeros=True, value_mode=CONTINUOUS)
    assert results
    assert all(r.dist_type == "continuous" for r in results)
