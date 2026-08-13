"""Tests for replenishment Qmin Monte Carlo."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hospital_demand_analyzer.domain.models import DistributionFitResult
from hospital_demand_analyzer.plots.canvas import MplCanvas
from hospital_demand_analyzer.plots.replenishment_qmin_plot import plot_demand_accumulation_trajectories
from hospital_demand_analyzer.statistics.replenishment_qmin import (
    compute_accumulated_demand_stats,
    run_replenishment_qmin_monte_carlo,
    simulate_demand_accumulation_trajectory,
)


def _poisson_fit(lam: float = 2.0) -> DistributionFitResult:
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


def test_fractional_replenishment_150_minutes():
    rng = np.random.default_rng(42)
    fit = _poisson_fit(3.0)
    traj = simulate_demand_accumulation_trajectory(150.0, fit, rng=rng, saldo_inicial=50)
    assert traj.times[-1] == pytest.approx(150.0)
    assert len(traj.times) >= 3
    assert traj.total_demand == int(traj.cumulative_demand[-1])
    assert traj.total_demand >= 0
    assert len(traj.balances) == len(traj.times)


def test_trajectory_saldo_decreases_until_time_end():
    rng = np.random.default_rng(7)
    fit = _poisson_fit(1.5)
    result = run_replenishment_qmin_monte_carlo("100", 120.0, fit, n_trajetorias=20, seed=7)
    traj = result.trajectories[0]
    assert traj.times[0] == 0.0
    assert traj.times[-1] == pytest.approx(120.0)
    assert traj.balances[0] == pytest.approx(result.saldo_inicial)
    assert np.all(np.diff(traj.balances) <= 1e-9)


def test_qmin_proposed_is_positive_integer():
    fit = _poisson_fit(2.0)
    result = run_replenishment_qmin_monte_carlo(
        "100", 120.0, fit, n_trajetorias=200, seed=99,
    )
    assert result.qmin_proposed == float(int(result.qmin_proposed))
    assert result.qmin_proposed >= 1
    assert result.saldo_inicial >= result.qmin_proposed
    assert result.n_trajetorias == 200
    assert len(result.trajectories) == 200
    assert all(math.isfinite(v) for v in result.accumulated_demands)


def test_no_nan_stats_with_all_zero_demand():
    fit = _poisson_fit(0.0)
    result = run_replenishment_qmin_monte_carlo(
        "zero", 120.0, fit, n_trajetorias=50, seed=1,
    )
    assert result.qmin_proposed >= 1
    for field in (
        "mean", "median", "q1", "q3", "p5", "p90", "p95", "p99", "std", "min_", "max_",
    ):
        assert math.isfinite(getattr(result, field))


def test_nan_fit_params_are_sanitized():
    fit = DistributionFitResult(
        name="Gamma",
        dist_type="continuous",
        params={"param_0": float("nan"), "param_1": 2.0},
        log_likelihood=-10.0,
        n_params=2,
        aic=22.0,
        aicc=22.0,
        bic=24.0,
        converged=True,
    )
    result = run_replenishment_qmin_monte_carlo(
        "bad", 120.0, fit, n_trajetorias=5, seed=3,
    )
    assert "param_0" not in result.parametros
    assert math.isfinite(result.qmin_proposed)


def test_compute_accumulated_demand_stats_quartiles():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    stats = compute_accumulated_demand_stats(values)
    assert stats["q1"] == pytest.approx(3.25)
    assert stats["median"] == pytest.approx(5.5)
    assert stats["q3"] == pytest.approx(7.75)
    assert stats["p95"] == pytest.approx(9.55)


def test_plot_monte_carlo_style_renders(qapp):
    fit = _poisson_fit(2.0)
    result = run_replenishment_qmin_monte_carlo("100", 120.0, fit, n_trajetorias=10, seed=5)
    canvas = MplCanvas(6, 4)
    plot_demand_accumulation_trajectories(canvas, result)
    qapp.processEvents()
    assert canvas.ax.get_legend() is not None
