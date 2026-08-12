"""UI tests for Monte Carlo parameter refresh behaviour."""

from __future__ import annotations

import time

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QWidget

from hospital_demand_analyzer.app.controller import AppController
from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.domain.models import DistributionFitResult, HourlySeries
from hospital_demand_analyzer.statistics.monte_carlo import run_tempqmin_monte_carlo
from hospital_demand_analyzer.ui.views.monte_carlo_view import MonteCarloView


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


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _seed_state(state: AppState) -> None:
    ts = pd.date_range("2024-01-01", periods=24, freq="h")
    hs = HourlySeries(
        produto="100",
        descricao="Produto teste",
        timestamps=ts,
        quantities=pd.Series([1.0] * 24, index=ts),
    )
    state.hourly_series = {"100": hs}
    state.descriptive_df = pd.DataFrame([{
        "produto": "100",
        "descricao": "Produto teste",
        "n": 24,
        "n_zeros": 0,
        "zero_ratio": 0.0,
        "non_zero_hours": 24,
        "sum": 24.0,
        "mean": 1.0,
        "expected_empirical": 1.0,
        "expected_conditional": 1.0,
        "median": 1.0,
        "std": 0.0,
        "var": 0.0,
        "cv": 0.0,
        "min": 1.0,
        "max": 1.0,
        "q25": 1.0,
        "q50": 1.0,
        "q75": 1.0,
        "iqr": 0.0,
        "p90": 1.0,
        "p95": 1.0,
        "p975": 1.0,
        "p99": 1.0,
        "skewness": 0.0,
        "kurtosis": 0.0,
    }])
    state.selected_sku = "100"
    fit = _poisson_fit(lam=2.0)
    key = AppController._fit_cache_key("100", False, state.demand_value_mode)
    state.fit_results[key] = [fit]


def _build_view(qapp) -> MonteCarloView:
    state = AppState()
    _seed_state(state)
    parent = QWidget()
    controller = AppController(state, parent)
    view = MonteCarloView(state, controller, parent)
    view._populate_products()
    view.product_combo.setCurrentIndex(0)
    qapp.processEvents()
    return view


def test_param_change_clears_plot_and_schedules_refresh(qapp):
    view = _build_view(qapp)
    fit = _poisson_fit(lam=2.0)
    result = run_tempqmin_monte_carlo(
        "100", saldo_inicial=120, qmin=24, fit=fit, n_trajetorias=5, seed=1,
    )
    view._on_simulation_done(result)
    qapp.processEvents()

    assert len(view.traj_plot.canvas.ax.get_lines()) >= 5
    assert view._last_run_params is not None

    view.qmin_spin.setValue(view.qmin_spin.value() + 10)
    qapp.processEvents()

    assert len(view.traj_plot.canvas.ax.get_lines()) == 0
    assert "recalculando" in view.info_label.text().lower()
    assert view._param_timer.isActive()


def test_auto_refresh_reruns_simulation_with_new_params(qapp):
    view = _build_view(qapp)
    fit = _poisson_fit(lam=2.0)
    result = run_tempqmin_monte_carlo(
        "100", saldo_inicial=120, qmin=24, fit=fit, n_trajetorias=5, seed=1,
    )
    view._on_simulation_done(result)
    qapp.processEvents()

    new_qmin = view.qmin_spin.value() + 10
    view.qmin_spin.setValue(new_qmin)
    view._auto_refresh_simulation()

    deadline = time.time() + 5.0
    while time.time() < deadline:
        qapp.processEvents()
        if (
            view._mc_worker is None
            and view._last_run_params is not None
            and view._last_run_params[2] == new_qmin
            and len(view.traj_plot.canvas.ax.get_lines()) >= 5
        ):
            break
        time.sleep(0.02)

    assert view._last_run_params is not None
    assert view._last_run_params[2] == new_qmin
    assert len(view.traj_plot.canvas.ax.get_lines()) >= 5
