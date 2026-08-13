"""Tests for demand sampling with zero-inflated distributions."""

from __future__ import annotations

import numpy as np

from hospital_demand_analyzer.domain.models import DistributionFitResult
from hospital_demand_analyzer.statistics.demand_sampler import sample_hourly_demand
from hospital_demand_analyzer.statistics.distributions.zero_inflated import ZINB_NAME, ZIP_NAME


def _zip_fit(pi: float = 0.7, lam: float = 3.0) -> DistributionFitResult:
    return DistributionFitResult(
        name=ZIP_NAME,
        dist_type="discrete",
        params={"pi": pi, "lambda": lam},
        log_likelihood=-50.0,
        n_params=2,
        aic=104.0,
        aicc=104.0,
        bic=110.0,
        converged=True,
    )


def _poisson_fit(lam: float) -> DistributionFitResult:
    return DistributionFitResult(
        name="Poisson",
        dist_type="discrete",
        params={"lambda": lam},
        log_likelihood=-10.0,
        n_params=1,
        aic=22.0,
        aicc=22.0,
        bic=24.0,
        converged=True,
    )


def test_zip_sampler_can_return_zero():
    rng = np.random.default_rng(42)
    samples = [sample_hourly_demand(_zip_fit(), rng) for _ in range(300)]
    assert any(s == 0 for s in samples)


def test_poisson_sampler_can_return_zero():
    rng = np.random.default_rng(1)
    samples = [sample_hourly_demand(_poisson_fit(0.05), rng) for _ in range(500)]
    assert any(s == 0 for s in samples)
    assert all(isinstance(s, int) and s >= 0 for s in samples)


def test_nbinom_sampler_can_return_zero():
    fit = DistributionFitResult(
        name="Negative Binomial",
        dist_type="discrete",
        params={"n": 0.5, "p": 0.9},
        log_likelihood=-10.0,
        n_params=2,
        aic=24.0,
        aicc=24.0,
        bic=28.0,
        converged=True,
    )
    rng = np.random.default_rng(7)
    samples = [sample_hourly_demand(fit, rng) for _ in range(500)]
    assert any(s == 0 for s in samples)
    assert all(s >= 0 for s in samples)
