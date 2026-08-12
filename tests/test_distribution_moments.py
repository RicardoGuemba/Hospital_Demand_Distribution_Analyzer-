"""Tests for theoretical distribution moments."""

from __future__ import annotations

from hospital_demand_analyzer.domain.models import DistributionFitResult
from hospital_demand_analyzer.statistics.distribution_moments import theoretical_expected_value


def _poisson_fit(lam: float, *, converged: bool = True) -> DistributionFitResult:
    return DistributionFitResult(
        name="Poisson",
        dist_type="discrete",
        params={"lambda": lam},
        log_likelihood=-100.0,
        n_params=1,
        aic=202.0,
        aicc=202.0,
        bic=204.0,
        converged=converged,
    )


def test_theoretical_expected_poisson():
    fit = _poisson_fit(3.0)
    assert theoretical_expected_value(fit) == 3.0


def test_theoretical_expected_non_converged_returns_none():
    fit = _poisson_fit(2.0, converged=False)
    assert theoretical_expected_value(fit) is None
