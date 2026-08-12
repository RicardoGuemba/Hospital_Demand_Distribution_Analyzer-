"""Tests for statistics."""

from __future__ import annotations

import numpy as np
import pytest

from hospital_demand_analyzer.statistics.ecdf import compute_ecdf, empirical_quantile
from hospital_demand_analyzer.statistics.fitter import fit_all, top_n
from hospital_demand_analyzer.statistics.metrics import compute_aic, compute_bic


def test_ecdf():
    data = np.array([1, 2, 3, 4, 5])
    x, y = compute_ecdf(data)
    assert len(x) == 5
    assert y[-1] == 1.0


def test_aic_bic():
    log_lik = -100
    aic = compute_aic(log_lik, 2)
    bic = compute_bic(log_lik, 2, 50)
    assert aic == pytest.approx(204.0)
    assert bic > aic


def test_fit_poisson():
    np.random.seed(42)
    data = np.random.poisson(5, 200).astype(float)
    results = fit_all(data)
    converged = [r for r in results if r.converged]
    assert len(converged) >= 1
    top = top_n(results, 2)
    assert len(top) >= 1
    assert top[0].aic < float("inf")


def test_fit_failure_does_not_stop():
    data = np.array([0.0])
    results = fit_all(data)
    assert len(results) > 0
