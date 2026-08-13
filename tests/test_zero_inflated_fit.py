"""Tests for zero-inflated Poisson and ZINB fitting."""

from __future__ import annotations

import numpy as np

from hospital_demand_analyzer.statistics.distributions.zero_inflated import (
    ZINB_NAME,
    ZIP_NAME,
    fit_zinb,
    fit_zip,
    zip_dist,
    zinb_dist,
)
from hospital_demand_analyzer.statistics.fitter import fit_all, top_n


def _zero_inflated_poisson(n: int, pi: float, lam: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        if rng.random() < pi:
            out.append(0)
        else:
            out.append(int(rng.poisson(lam)))
    return np.array(out, dtype=float)


def _zero_inflated_nb(n: int, pi: float, n_nb: float, p_nb: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        if rng.random() < pi:
            out.append(0)
        else:
            out.append(int(rng.negative_binomial(n_nb, p_nb)))
    return np.array(out, dtype=float)


def test_zip_fit_converges_on_zero_heavy_data():
    data = _zero_inflated_poisson(400, pi=0.40, lam=3.0, seed=7)
    params, log_lik, ok = fit_zip(data)
    assert ok
    assert log_lik > float("-inf")
    assert 0.0 < params["pi"] < 1.0
    assert params["lambda"] > 0.0


def test_zinb_fit_converges_on_overdispersed_data():
    data = _zero_inflated_nb(300, pi=0.30, n_nb=4.0, p_nb=0.4, seed=11)
    params, log_lik, ok = fit_zinb(data)
    assert ok
    assert log_lik > float("-inf")
    assert 0.0 < params["pi"] < 1.0
    assert params["n"] > 0.0
    assert 0.0 < params["p"] < 1.0


def test_fit_all_includes_zip_and_zinb():
    zip_data = _zero_inflated_poisson(200, pi=0.5, lam=2.0, seed=3)
    zip_results = fit_all(zip_data, include_zeros=True)
    zip_names = {r.name for r in zip_results if r.converged}
    assert ZIP_NAME in zip_names

    zinb_data = _zero_inflated_nb(300, pi=0.30, n_nb=4.0, p_nb=0.4, seed=11)
    zinb_results = fit_all(zinb_data, include_zeros=True)
    zinb_names = {r.name for r in zinb_results if r.converged}
    assert ZINB_NAME in zinb_names


def test_zip_beats_poisson_on_structural_zeros():
    data = _zero_inflated_poisson(500, pi=0.60, lam=4.0, seed=21)
    results = fit_all(data, include_zeros=True)
    best = top_n(results, 1)[0]
    assert best.name in {ZIP_NAME, ZINB_NAME}


def test_zip_sampling_allows_zeros():
    rng = np.random.default_rng(0)
    samples = [zip_dist.rvs(0.5, 2.0, random_state=rng) for _ in range(200)]
    assert any(s == 0 for s in samples)
    assert any(s > 0 for s in samples)


def test_zinb_mean():
    mean = zinb_dist.mean(0.3, 4.0, 0.4)
    assert mean > 0.0
