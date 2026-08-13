"""Sample hourly demand from fitted distribution parameters."""

from __future__ import annotations

import numpy as np
from scipy import stats

from hospital_demand_analyzer.domain.models import DistributionFitResult
from hospital_demand_analyzer.statistics.distributions.zero_inflated import ZINB_NAME, ZIP_NAME, zinb_dist, zip_dist
from hospital_demand_analyzer.statistics.fit_params import params_tuple_for_fit

DIST_MAP = {
    "Poisson": stats.poisson,
    "Negative Binomial": stats.nbinom,
    ZIP_NAME: zip_dist,
    ZINB_NAME: zinb_dist,
    "Exponential": stats.expon,
    "Gamma": stats.gamma,
    "Lognormal": stats.lognorm,
    "Weibull": stats.weibull_min,
}


def fit_params_tuple(fit: DistributionFitResult) -> tuple[float, ...]:
    """Convert stored fit params to scipy args tuple."""
    return params_tuple_for_fit(fit.name, fit.params)


def get_dist_for_fit(fit: DistributionFitResult):
    dist = DIST_MAP.get(fit.name)
    if dist is None:
        raise ValueError(f"Distribuição não suportada para simulação: {fit.name}")
    return dist, fit_params_tuple(fit)


def sample_hourly_demand(
    fit: DistributionFitResult,
    rng: np.random.Generator,
) -> int:
    """Draw one integer demand value ≥ 0 from the fitted marginal FDC."""
    dist, params = get_dist_for_fit(fit)
    value = dist.rvs(*params, random_state=rng)
    return max(0, int(round(float(value))))


def sample_demand_from_fit(
    fit: DistributionFitResult,
    rng: np.random.Generator,
    *,
    allow_zero: bool = True,
) -> int:
    """Alias for hourly demand sampling (allow_zero kept for API compatibility)."""
    del allow_zero
    return sample_hourly_demand(fit, rng)
