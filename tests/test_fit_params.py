"""Tests for fit parameter formatting."""

from __future__ import annotations

from hospital_demand_analyzer.domain.models import DistributionFitResult
from hospital_demand_analyzer.statistics.fit_params import best_fit, format_fit_params


def _result(name: str, aic: float, params: dict | None = None) -> DistributionFitResult:
    return DistributionFitResult(
        name=name,
        dist_type="discrete",
        params=params or {"lambda": 2.0},
        log_likelihood=-aic,
        n_params=1,
        aic=aic,
        aicc=aic,
        bic=aic,
        converged=True,
    )


def test_format_fit_params_poisson():
    fit = _result("Poisson", 100.0, {"lambda": 3.5})
    text = format_fit_params(fit)
    assert "Poisson" in text
    assert "λ=3.5" in text


def test_best_fit_returns_lowest_aic():
    results = [_result("Poisson", 120.0), _result("Negative Binomial", 100.0, {"n": 2, "p": 0.5})]
    best = best_fit(results)
    assert best is not None
    assert best.name == "Negative Binomial"
