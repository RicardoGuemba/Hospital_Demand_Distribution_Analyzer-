"""Fit metrics: AIC, BIC, goodness-of-fit, tail score."""

from __future__ import annotations

import numpy as np
from scipy import stats

TAIL_PERCENTILES = [90, 95, 97.5, 99]
TAIL_WEIGHTS = {90: 0.15, 95: 0.25, 97.5: 0.25, 99: 0.35}


def compute_aic(log_likelihood: float, n_params: int) -> float:
    return 2 * n_params - 2 * log_likelihood


def compute_aicc(aic: float, n: int, n_params: int) -> float:
    if n - n_params - 1 <= 0:
        return aic
    return aic + (2 * n_params * (n_params + 1)) / (n - n_params - 1)


def compute_bic(log_likelihood: float, n_params: int, n: int) -> float:
    return n_params * np.log(n) - 2 * log_likelihood


def compute_tail_fit_score(
    empirical: np.ndarray,
    theoretical_quantile_fn,
) -> tuple[float, dict[str, float]]:
    """Tail Fit Score: weighted mean of relative absolute errors at tail percentiles."""
    errors: dict[str, float] = {}
    weighted_sum = 0.0
    weight_total = 0.0
    for p in TAIL_PERCENTILES:
        emp_q = float(np.percentile(empirical, p))
        try:
            theo_q = float(theoretical_quantile_fn(p / 100))
        except Exception:
            theo_q = 0.0
        if emp_q > 0:
            rel_err = abs(theo_q - emp_q) / emp_q
        else:
            rel_err = abs(theo_q - emp_q)
        key = f"P{p}"
        errors[key] = rel_err
        w = TAIL_WEIGHTS[p]
        weighted_sum += w * rel_err
        weight_total += w
    score = weighted_sum / weight_total if weight_total > 0 else float("inf")
    return score, errors


def ks_test(empirical: np.ndarray, dist, params: tuple) -> tuple[float, float | None]:
    try:
        if hasattr(dist, "cdf"):
            stat, pval = stats.kstest(empirical, dist.cdf, args=params)
            return float(stat), float(pval)
    except Exception:
        pass
    return float("nan"), None


def cvm_test(empirical: np.ndarray, dist, params: tuple) -> float:
    try:
        result = stats.cramervonmises(empirical, dist.cdf, args=params)
        return float(result.statistic)
    except Exception:
        return float("nan")


def ad_test(empirical: np.ndarray, dist, params: tuple) -> float:
    try:
        from scipy.stats import anderson
        # Anderson-Darling for normal only in scipy; use simplified approach
        sorted_data = np.sort(empirical)
        n = len(sorted_data)
        cdf_vals = dist.cdf(sorted_data, *params)
        cdf_vals = np.clip(cdf_vals, 1e-10, 1 - 1e-10)
        i = np.arange(1, n + 1)
        s = np.sum((2 * i - 1) / n * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1])))
        return float(-n - s)
    except Exception:
        return float("nan")
