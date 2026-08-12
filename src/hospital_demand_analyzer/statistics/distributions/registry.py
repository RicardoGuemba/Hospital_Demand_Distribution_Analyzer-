"""Distribution registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats


@dataclass
class DistributionSpec:
    name: str
    dist_type: str  # discrete / continuous
    scipy_name: str
    n_params: int
    fit_fn: Callable | None = None


def _fit_scipy(data: np.ndarray, dist_name: str) -> tuple[dict, float, bool]:
    dist = getattr(stats, dist_name)
    try:
        params = dist.fit(data)
        log_lik = float(np.sum(dist.logpdf(data, *params))) if hasattr(dist, "logpdf") else float(np.sum(dist.logpmf(data, *params)))
        param_names = ["param_%d" % i for i in range(len(params))]
        return dict(zip(param_names, params, strict=False)), log_lik, True
    except Exception:
        return {}, float("-inf"), False


def _fit_poisson(data: np.ndarray) -> tuple[dict, float, bool]:
    int_data = np.round(data).astype(int)
    int_data = int_data[int_data >= 0]
    if len(int_data) == 0:
        return {}, float("-inf"), False
    lam = float(np.mean(int_data))
    if lam < 0:
        return {}, float("-inf"), False
    log_lik = float(np.sum(stats.poisson.logpmf(int_data, lam)))
    return {"lambda": lam}, log_lik, True


def _fit_nbinom(data: np.ndarray) -> tuple[dict, float, bool]:
    try:
        int_data = data.astype(int)
        mean = np.mean(int_data)
        var = np.var(int_data)
        if var <= mean or mean <= 0:
            return {}, float("-inf"), False
        p = mean / var
        n = mean * p / (1 - p)
        if n <= 0 or p <= 0 or p >= 1:
            return {}, float("-inf"), False
        log_lik = float(np.sum(stats.nbinom.logpmf(int_data, n, p)))
        return {"n": n, "p": p}, log_lik, True
    except Exception:
        return {}, float("-inf"), False


REGISTRY: list[DistributionSpec] = [
    DistributionSpec("Poisson", "discrete", "poisson", 1, _fit_poisson),
    DistributionSpec("Negative Binomial", "discrete", "nbinom", 2, _fit_nbinom),
    DistributionSpec("Exponential", "continuous", "expon", 1),
    DistributionSpec("Gamma", "continuous", "gamma", 2),
    DistributionSpec("Lognormal", "continuous", "lognorm", 2),
    DistributionSpec("Weibull", "continuous", "weibull_min", 2),
]

DISCRETE_DIST_MAP = {
    "Poisson": stats.poisson,
    "Negative Binomial": stats.nbinom,
}


def get_distribution(name: str) -> DistributionSpec | None:
    for d in REGISTRY:
        if d.name == name:
            return d
    return None
