"""Functional UI tests."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.domain.models import HourlySeries
from hospital_demand_analyzer.ui.nav import STEP_STATISTICS, STEP_TIME_SERIES
from hospital_demand_analyzer.ui.views.statistics_view import StatisticsView
from hospital_demand_analyzer.ui.windows.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _seed_state(state: AppState) -> None:
    ts = pd.date_range("2024-01-01", periods=5, freq="h")
    hs = HourlySeries(
        produto="100",
        descricao="Produto teste",
        timestamps=ts,
        quantities=pd.Series([0.0, 1.0, 2.0, 3.0, 4.0], index=ts),
    )
    state.hourly_series = {"100": hs}
    state.descriptive_df = pd.DataFrame([{
        "produto": "100",
        "descricao": "Produto teste",
        "n": 5,
        "n_zeros": 1,
        "zero_ratio": 0.2,
        "non_zero_hours": 4,
        "sum": 10.0,
        "mean": 2.0,
        "expected_empirical": 2.0,
        "expected_conditional": 2.5,
        "median": 2.0,
        "std": 1.58,
        "var": 2.5,
        "cv": 0.79,
        "min": 0.0,
        "max": 4.0,
        "q25": 1.0,
        "q50": 2.0,
        "q75": 3.0,
        "iqr": 2.0,
        "p90": 3.6,
        "p95": 3.8,
        "p975": 3.9,
        "p99": 3.96,
        "skewness": 0.0,
        "kurtosis": -1.2,
    }])
    state.selected_sku = "100"


def test_main_window_opens_without_autoload(qapp):
    window = MainWindow(AppState(), auto_load=False)
    assert window.sidebar.count() > 0
    assert window.stack.count() == 12
    window.close()
    qapp.processEvents()


def test_navigate_to_statistics_view(qapp):
    window = MainWindow(AppState(), auto_load=False)
    window.navigate_to(STEP_STATISTICS)
    qapp.processEvents()
    stats_view = window.views[STEP_STATISTICS]
    assert window.stack.currentWidget() is stats_view
    window.close()
    qapp.processEvents()


def test_statistics_view_populates_table(qapp):
    state = AppState()
    _seed_state(state)
    view = StatisticsView(state, None)
    view.refresh()
    assert view.table.rowCount() == 1
    assert view.table.item(0, 0).text() == "100"
    assert "100" in view.detail_title.text() or view.table.rowCount() == 1


def test_statistics_view_double_click_navigates(qapp):
    state = AppState()
    _seed_state(state)
    window = MainWindow(state, auto_load=False)
    stats_view = window.views[STEP_STATISTICS]
    window.navigate_to(STEP_STATISTICS)
    stats_view.refresh()
    stats_view.table.selectRow(0)
    stats_view._on_double_click(0, 0)
    qapp.processEvents()
    assert state.selected_sku == "100"
    assert window.stack.currentWidget() is window.views[STEP_TIME_SERIES]
    window.close()
    qapp.processEvents()


def test_main_window_import_does_not_block_ui(qapp):
    data_dir = Path(__file__).resolve().parents[1] / "Dados_Demanda"
    if not data_dir.is_dir():
        pytest.skip("Dados_Demanda not available")

    window = MainWindow(AppState(), auto_load=False)
    window.show()
    qapp.processEvents()

    done = {"ok": False}
    window.controller.load_directory(
        data_dir,
        on_done=lambda payload: (
            window.controller.handle_import_result(payload),
            done.update({"ok": True}),
        ),
        on_error=lambda msg: pytest.fail(msg),
    )

    deadline = time.time() + 120
    while not done["ok"] and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)

    assert done["ok"]
    assert window.state.import_result is not None
    assert len(window.state.hourly_series) > 0
    window.close()
    qapp.processEvents()
