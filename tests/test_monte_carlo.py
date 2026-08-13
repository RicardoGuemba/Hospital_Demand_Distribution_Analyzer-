"""Tests for Monte Carlo TempQMin simulation."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from hospital_demand_analyzer.domain.models import DistributionFitResult
from hospital_demand_analyzer.statistics.demand_sampler import sample_hourly_demand
from hospital_demand_analyzer.statistics.monte_carlo import (
    MINUTES_PER_STEP,
    SALDO_FACTOR_MAX,
    SALDO_FACTOR_MIN,
    default_qmin,
    default_saldo_inicial,
    format_tempqmin,
    normalize_inventory_level,
    run_tempqmin_monte_carlo,
    simulate_single_trajectory,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _poisson_fit(lam: float = 2.0) -> DistributionFitResult:
    return DistributionFitResult(
        name="Poisson",
        dist_type="discrete",
        params={"lambda": lam},
        log_likelihood=-100.0,
        n_params=1,
        aic=202.0,
        aicc=202.0,
        bic=204.0,
        converged=True,
    )


def test_default_qmin_is_positive_integer():
    assert default_qmin(1.0) == 24
    assert default_qmin(0.5) == 12
    assert default_qmin(0.001) == 1


def test_default_saldo_inicial_is_positive_integer():
    rng = np.random.default_rng(1)
    qmin = 24
    for _ in range(30):
        saldo = default_saldo_inicial(qmin, rng)
        assert isinstance(saldo, int)
        assert saldo >= 1
        factor = saldo / qmin
        assert SALDO_FACTOR_MIN <= factor <= SALDO_FACTOR_MAX + 0.5


def test_format_tempqmin_shows_hours():
    assert format_tempqmin(72.0) == "72.0 min (1.2 h)"
    assert format_tempqmin(0.0) == "0.0 min (0.0 h)"


def test_normalize_inventory_level():
    assert normalize_inventory_level(0.4) == 1
    assert normalize_inventory_level(10.6) == 11


def test_sample_hourly_demand_is_non_negative_integer():
    fit = _poisson_fit(lam=2.0)
    rng = np.random.default_rng(42)
    for _ in range(50):
        d = sample_hourly_demand(fit, rng)
        assert isinstance(d, int)
        assert d >= 0


def test_tempqmin_zero_when_saldo_at_or_below_qmin():
    fit = _poisson_fit()
    rng = np.random.default_rng(42)
    traj = simulate_single_trajectory(100.0, 100.0, fit, rng=rng)
    assert traj.temp_qmin == 0.0
    traj2 = simulate_single_trajectory(50.0, 100.0, fit, rng=rng)
    assert traj2.temp_qmin == 0.0


def test_monte_carlo_returns_n_trajectories_in_minutes():
    fit = _poisson_fit(lam=5.0)
    result = run_tempqmin_monte_carlo(
        "SKU1", saldo_inicial=500, qmin=100, fit=fit,
        n_trajetorias=50, seed=123,
    )
    assert result.saldo_inicial == 500
    assert result.qmin == 100
    assert result.n_trajetorias == 50
    assert result.delta_t_minutos == MINUTES_PER_STEP
    assert len(result.trajectories) == 50
    assert result.mean >= 0
    assert result.p5 <= result.p95


def test_fractional_tempqmin_on_crossing_in_minutes():
    fit = _poisson_fit(lam=0.0)

    import hospital_demand_analyzer.statistics.monte_carlo as mc

    original = mc.sample_hourly_demand
    calls = iter([10, 50])

    def mock_sample(_fit, _rng):
        return next(calls)

    mc.sample_hourly_demand = mock_sample
    try:
        rng = np.random.default_rng(0)
        traj = mc.simulate_single_trajectory(520.0, 500.0, fit, rng=rng)
        assert abs(traj.temp_qmin - 72.0) < 1e-9
    finally:
        mc.sample_hourly_demand = original


def test_monte_carlo_poisson_zero_hours_preserve_balance():
    fit = _poisson_fit(lam=0.05)

    import hospital_demand_analyzer.statistics.monte_carlo as mc

    calls = iter([0, 0, 0, 100])

    def mock_sample(_fit, _rng):
        return next(calls)

    original = mc.sample_hourly_demand
    mc.sample_hourly_demand = mock_sample
    try:
        rng = np.random.default_rng(0)
        traj = mc.simulate_single_trajectory(520.0, 500.0, fit, rng=rng)
        assert traj.balances[0] == pytest.approx(520.0)
        assert traj.balances[-1] == pytest.approx(500.0)
        assert traj.temp_qmin == pytest.approx(3 * MINUTES_PER_STEP + (20.0 / 100.0) * MINUTES_PER_STEP)
    finally:
        mc.sample_hourly_demand = original


def test_monte_carlo_zip_zero_hours_preserve_balance():
    from hospital_demand_analyzer.statistics.distributions.zero_inflated import ZIP_NAME

    fit = DistributionFitResult(
        name=ZIP_NAME,
        dist_type="discrete",
        params={"pi": 0.99, "lambda": 50.0},
        log_likelihood=-10.0,
        n_params=2,
        aic=24.0,
        aicc=24.0,
        bic=30.0,
        converged=True,
    )

    import hospital_demand_analyzer.statistics.monte_carlo as mc

    calls = iter([0, 0, 0, 100])

    def mock_sample(_fit, _rng):
        return next(calls)

    original = mc.sample_hourly_demand
    mc.sample_hourly_demand = mock_sample
    try:
        rng = np.random.default_rng(0)
        traj = mc.simulate_single_trajectory(520.0, 500.0, fit, rng=rng)
        assert traj.balances[0] == pytest.approx(520.0)
        assert traj.balances[-1] == pytest.approx(500.0)
        assert traj.temp_qmin == pytest.approx(3 * MINUTES_PER_STEP + (20.0 / 100.0) * MINUTES_PER_STEP)
    finally:
        mc.sample_hourly_demand = original


def test_plot_monte_carlo_trajectories_renders_lines(qapp):
    from hospital_demand_analyzer.plots.canvas import PlotWidget
    from hospital_demand_analyzer.plots.monte_carlo_plot import plot_monte_carlo_trajectories

    fit = _poisson_fit(lam=4.0)
    result = run_tempqmin_monte_carlo(
        "SKU1", saldo_inicial=300.0, qmin=80.0, fit=fit,
        n_trajetorias=15, seed=99,
    )
    pw = PlotWidget(8, 5)
    plot_monte_carlo_trajectories(pw.canvas, result)
    qapp.processEvents()

    lines = pw.canvas.ax.get_lines()
    assert len(lines) >= 15
    assert any(len(line.get_xdata()) > 1 for line in lines[:15])
