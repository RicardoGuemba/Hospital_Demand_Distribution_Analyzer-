"""Sample hourly demand from fitted distribution parameters."""

from __future__ import annotations

import numpy as np
from scipy import stats

from hospital_demand_analyzer.domain.models import DistributionFitResult

DIST_MAP = {
    "Poisson": stats.poisson,
    "Negative Binomial": stats.nbinom,
    "Exponential": stats.expon,
    "Gamma": stats.gamma,
    "Lognormal": stats.lognorm,
    "Weibull": stats.weibull_min,
}


def fit_params_tuple(fit: DistributionFitResult) -> tuple[float, ...]:
    """Convert stored fit params to scipy args tuple."""
    params = fit.params
    if fit.name == "Poisson":
        return (float(params["lambda"]),)
    if fit.name == "Negative Binomial":
        return (float(params["n"]), float(params["p"]))
    ordered = sorted(
        (int(k.split("_")[1]), v) for k, v in params.items() if k.startswith("param_")
    )
    return tuple(float(v) for _, v in ordered)


def get_dist_for_fit(fit: DistributionFitResult):
    dist = DIST_MAP.get(fit.name)
    if dist is None:
        raise ValueError(f"Distribuição não suportada para simulação: {fit.name}")
    return dist, fit_params_tuple(fit)


def sample_hourly_demand(
    fit: DistributionFitResult,
    rng: np.random.Generator,
    *,
    max_attempts: int = 500,
) -> int:
    """Draw one integer demand value strictly greater than zero."""
    dist, params = get_dist_for_fit(fit)
    for _ in range(max_attempts):
        value = dist.rvs(*params, random_state=rng)
        demand = int(round(float(value)))
        if demand > 0:
            return demand
    return 1
