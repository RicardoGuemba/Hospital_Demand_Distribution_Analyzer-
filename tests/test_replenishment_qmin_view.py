"""UI tests for replenishment Qmin plot refresh behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QWidget

from hospital_demand_analyzer.app.controller import AppController
from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.domain.models import HourlySeries, ReplenishmentQminResult
from hospital_demand_analyzer.statistics.replenishment_qmin import run_replenishment_qmin_monte_carlo
from hospital_demand_analyzer.ui.views.replenishment_qmin_view import ReplenishmentQminView


def _poisson_fit(lam: float = 2.0):
    from hospital_demand_analyzer.domain.models import DistributionFitResult
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
    state.session_selected_skus = {"100"}
    state.session_sku_order = ["100"]
    fit = _poisson_fit(2.0)
    key = AppController._fit_cache_key("100", False, state.demand_value_mode)
    state.fit_results[key] = [fit]


def _build_view(qapp) -> ReplenishmentQminView:
    state = AppState()
    _seed_state(state)
    parent = QWidget()
    controller = AppController(state, parent)
    view = ReplenishmentQminView(state, controller, parent)
    view.refresh()
    qapp.processEvents()
    return view


def test_sku_done_replots_selected_sku(qapp):
    view = _build_view(qapp)
    fit = _poisson_fit(2.0)
    result = run_replenishment_qmin_monte_carlo(
        "100", 120.0, fit, n_trajetorias=5, seed=1,
    )
    view.state.replenishment_qmin_results["100"] = result
    view.table.selectRow(0)
    view._on_sku_done("100", result)
    qapp.processEvents()

    assert len(view.traj_plot.canvas.ax.get_lines()) >= 5
    assert view._last_run_params is None


def test_param_change_clears_plot(qapp):
    view = _build_view(qapp)
    fit = _poisson_fit(2.0)
    result = run_replenishment_qmin_monte_carlo(
        "100", 120.0, fit, n_trajetorias=5, seed=1,
    )
    view._on_done()
    view.state.replenishment_qmin_results["100"] = result
    view.table.selectRow(0)
    view._render_plot("100")
    view._last_run_params = view._current_run_params()
    qapp.processEvents()

    assert len(view.traj_plot.canvas.ax.get_lines()) >= 5

    view.replenish_spin.setValue(view.replenish_spin.value() + 30)
    qapp.processEvents()

    assert len(view.traj_plot.canvas.ax.get_lines()) == 0
    assert "alterados" in view.status_label.text().lower()


def test_batch_done_sets_last_run_params(qapp):
    view = _build_view(qapp)
    view.replenish_spin.setValue(120)
    view.n_spin.setValue(50)
    view._on_done()
    assert view._last_run_params == (120, 50)
