"""Distribution fitting engine."""

from __future__ import annotations

import time

import numpy as np
from scipy import stats

from hospital_demand_analyzer.domain.models import DistributionFitResult
from hospital_demand_analyzer.domain.demand_values import (
    CONTINUOUS,
    DISCRETE,
    sanitize_demand,
)
from hospital_demand_analyzer.statistics.distributions.registry import (
    DISCRETE_DIST_MAP,
    REGISTRY,
    _fit_scipy,
)
from hospital_demand_analyzer.statistics.distributions.zero_inflated import ZINB_NAME, ZIP_NAME
from hospital_demand_analyzer.statistics.fit_params import params_tuple_for_fit
from hospital_demand_analyzer.statistics.metrics import (
    ad_test,
    compute_aic,
    compute_aicc,
    compute_bic,
    compute_tail_fit_score,
    cvm_test,
    ks_test,
)


def _make_quantile_fn(dist, params: tuple):
    def qfn(p: float) -> float:
        if hasattr(dist, "ppf"):
            return float(dist.ppf(p, *params))
        return 0.0
    return qfn


def _resolve_dist(spec, param_dict: dict):
    if spec.name in DISCRETE_DIST_MAP:
        return DISCRETE_DIST_MAP[spec.name], params_tuple_for_fit(spec.name, param_dict)
    dist = getattr(stats, spec.scipy_name, None)
    return dist, params_tuple_for_fit(spec.name, param_dict)


def fit_single(data: np.ndarray, spec) -> DistributionFitResult:
    """Fit a single distribution to data."""
    t0 = time.perf_counter()
    warnings: list[str] = []
    n = len(data)

    if n < 5:
        return DistributionFitResult(
            name=spec.name, dist_type=spec.dist_type, params={},
            log_likelihood=float("-inf"), n_params=spec.n_params,
            aic=float("inf"), aicc=float("inf"), bic=float("inf"),
            converged=False, warnings=["Insufficient sample size"],
        )

    pos_data = sanitize_demand(data)
    if len(pos_data) < len(data):
        warnings.append("Valores negativos excluídos do ajuste")

    if spec.dist_type == "continuous" and np.any(pos_data == 0):
        fit_data = pos_data[pos_data > 0]
    else:
        fit_data = pos_data

    if len(fit_data) == 0:
        return DistributionFitResult(
            name=spec.name, dist_type=spec.dist_type, params={},
            log_likelihood=float("-inf"), n_params=spec.n_params,
            aic=float("inf"), aicc=float("inf"), bic=float("inf"),
            converged=False, warnings=warnings + ["Sem observações válidas para ajuste"],
            fit_time_ms=(time.perf_counter() - t0) * 1000,
        )

    fit_data = np.round(fit_data).astype(int) if spec.dist_type == "discrete" else fit_data

    if spec.fit_fn:
        param_dict, log_lik, converged = spec.fit_fn(fit_data)
        params_tuple = tuple(param_dict.values())
    else:
        param_dict, log_lik, converged = _fit_scipy(fit_data, spec.scipy_name)
        params_tuple = tuple(param_dict.values())

    if not converged or log_lik == float("-inf"):
        return DistributionFitResult(
            name=spec.name, dist_type=spec.dist_type, params=param_dict,
            log_likelihood=log_lik, n_params=spec.n_params,
            aic=float("inf"), aicc=float("inf"), bic=float("inf"),
            converged=False, warnings=warnings + ["Fit did not converge"],
            fit_time_ms=(time.perf_counter() - t0) * 1000,
        )

    aic = compute_aic(log_lik, spec.n_params)
    aicc = compute_aicc(aic, n, spec.n_params)
    bic = compute_bic(log_lik, spec.n_params, n)

    dist, params_tuple = _resolve_dist(spec, param_dict)
    ks_stat, ks_pval = float("nan"), None
    cvm_stat = float("nan")
    ad_stat = float("nan")
    tail_score = float("nan")
    tail_errors: dict[str, float] = {}

    if dist is not None and params_tuple:
        try:
            ks_stat, ks_pval = ks_test(fit_data, dist, params_tuple)
            cvm_stat = cvm_test(fit_data, dist, params_tuple)
            ad_stat = ad_test(fit_data, dist, params_tuple)
            qfn = _make_quantile_fn(dist, params_tuple)
            tail_score, tail_errors = compute_tail_fit_score(fit_data.astype(float), qfn)
        except Exception:
            warnings.append("Some goodness-of-fit metrics unavailable")

    return DistributionFitResult(
        name=spec.name,
        dist_type=spec.dist_type,
        params=param_dict,
        log_likelihood=log_lik,
        n_params=spec.n_params,
        aic=aic,
        aicc=aicc,
        bic=bic,
        ks_stat=ks_stat,
        ks_pvalue=ks_pval,
        cvm_stat=cvm_stat,
        ad_stat=ad_stat,
        tail_fit_score=tail_score,
        tail_errors=tail_errors,
        converged=True,
        warnings=warnings,
        fit_time_ms=(time.perf_counter() - t0) * 1000,
    )


def fit_all(
    data: np.ndarray,
    include_zeros: bool = True,
    *,
    value_mode: str = DISCRETE,
) -> list[DistributionFitResult]:
    """Fit registered distributions; failures don't stop others."""
    data = sanitize_demand(data)
    if not include_zeros:
        data = data[data > 0]
    if len(data) == 0:
        return []

    if value_mode == CONTINUOUS:
        specs = [s for s in REGISTRY if s.dist_type == "continuous"]
    else:
        specs = [s for s in REGISTRY if s.dist_type == "discrete"]

    results: list[DistributionFitResult] = []
    for spec in specs:
        try:
            result = fit_single(data, spec)
            results.append(result)
        except Exception:
            results.append(
                DistributionFitResult(
                    name=spec.name, dist_type=spec.dist_type, params={},
                    log_likelihood=float("-inf"), n_params=spec.n_params,
                    aic=float("inf"), aicc=float("inf"), bic=float("inf"),
                    converged=False, warnings=["Fit raised exception"],
                )
            )

    valid = [r for r in results if r.converged and r.aic < float("inf")]
    if valid:
        min_aic = min(r.aic for r in valid)
        min_bic = min(r.bic for r in valid)
        for r in results:
            if r.converged:
                r.delta_aic = r.aic - min_aic
                r.delta_bic = r.bic - min_bic

        valid.sort(key=lambda r: r.aic)
        for i, r in enumerate(valid):
            r.rank = i + 1
        valid_tail = sorted(valid, key=lambda r: r.tail_fit_score or float("inf"))
        for i, r in enumerate(valid_tail):
            r.rank_tail = i + 1

    return results


def top_n(results: list[DistributionFitResult], n: int = 2) -> list[DistributionFitResult]:
    valid = [r for r in results if r.converged]
    valid.sort(key=lambda r: r.aic)
    return valid[:n]


def select_fits_for_display(
    results: list[DistributionFitResult],
    zero_ratio: float,
    max_fits: int = 3,
    *,
    value_mode: str = DISCRETE,
) -> list[DistributionFitResult]:
    """Pick distributions to overlay on plots; promote Poisson when zeros dominate."""
    valid = sorted([r for r in results if r.converged], key=lambda r: r.aic)
    if not valid:
        return []

    if value_mode == CONTINUOUS:
        return valid[:max_fits]

    selected: list[DistributionFitResult] = valid[:2]
    names = {r.name for r in selected}

    if zero_ratio >= 0.10:
        for candidate in (ZIP_NAME, ZINB_NAME, "Poisson", "Negative Binomial"):
            disc = next((r for r in valid if r.name == candidate), None)
            if disc and disc.name not in names:
                if len(selected) >= 2:
                    selected[-1] = disc
                else:
                    selected.append(disc)
                names.add(disc.name)
                break

    return selected[:max_fits]
