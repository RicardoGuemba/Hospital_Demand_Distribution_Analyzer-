"""Tests for select_fits_for_display."""

from __future__ import annotations

import numpy as np

from hospital_demand_analyzer.domain.models import DistributionFitResult
from hospital_demand_analyzer.statistics.fitter import fit_all, select_fits_for_display


def _result(name: str, aic: float, dist_type: str = "continuous") -> DistributionFitResult:
    return DistributionFitResult(
        name=name, dist_type=dist_type, params={}, log_likelihood=-aic,
        n_params=1, aic=aic, aicc=aic, bic=aic, converged=True,
    )


def test_select_fits_promotes_poisson_when_many_zeros():
    np.random.seed(0)
    data = np.zeros(500)
    data[:50] = np.random.poisson(3, 50)
    results = fit_all(data, include_zeros=True)
    display = select_fits_for_display(results, zero_ratio=0.9)
    names = {r.name for r in display}
    assert "Poisson" in names


def test_select_fits_poisson_ranked():
    results = [
        _result("Gamma", 100),
        _result("Weibull", 110),
        _result("Poisson", 120, "discrete"),
    ]
    display = select_fits_for_display(results, zero_ratio=0.8)
    assert any(r.name == "Poisson" for r in display)
