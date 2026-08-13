"""Stability tests for bulk CDF fitting."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QWidget

from hospital_demand_analyzer.app.controller import AppController
from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.domain.models import HourlySeries
from hospital_demand_analyzer.ui.views.sku_cdf_view import SkuCdfView


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def host_widget(qapp):
    widget = QWidget()
    yield widget
    widget.close()
    qapp.processEvents()


def _make_state(n_skus: int = 20) -> AppState:
    state = AppState()
    ts = pd.date_range("2024-01-01", periods=80, freq="h")
    rng = np.random.default_rng(7)
    series: dict[str, HourlySeries] = {}
    rows: list[dict] = []
    for i in range(n_skus):
        produto = str(2000 + i)
        quantities = rng.poisson(1.8, size=len(ts)).astype(float)
        series[produto] = HourlySeries(
            produto=produto,
            descricao=f"SKU {i}",
            timestamps=ts,
            quantities=pd.Series(quantities, index=ts),
        )
        rows.append({"produto": produto, "descricao": f"SKU {i}", "mean": float(quantities.mean())})
    state.hourly_series = series
    state.descriptive_df = pd.DataFrame(rows)
    return state


def test_bulk_fit_all_skus_completes(qapp, host_widget):
    state = _make_state(25)
    controller = AppController(state, host_widget)
    progress_events: list[tuple[str, int, int]] = []
    done = {"ok": False, "sku_count": 0}

    def on_progress(produto: str, done_n: int, total: int) -> None:
        progress_events.append((produto, done_n, total))

    def on_sku_done(_produto: str, _results: list) -> None:
        done["sku_count"] += 1

    def on_finished() -> None:
        done["ok"] = True

    queued = controller.start_bulk_fit_all_skus(
        on_progress=on_progress,
        on_sku_done=on_sku_done,
        on_done=on_finished,
        on_error=pytest.fail,
    )
    assert queued == 25

    deadline = time.time() + 120
    while not done["ok"] and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.005)

    assert done["ok"], "bulk fit did not finish in time"
    assert done["sku_count"] == 25
    assert len(state.fit_results) == 25
    assert progress_events, "expected progress updates"

    controller.shutdown()
    qapp.processEvents()


def test_sku_cdf_view_bulk_fit_does_not_rebuild_whole_table(qapp, host_widget):
    state = _make_state(15)
    controller = AppController(state, host_widget)
    view = SkuCdfView(state, controller, host_widget)
    view.show()
    qapp.processEvents()

    view.refresh()
    initial_row_count = view.table.rowCount()
    assert initial_row_count == 15

    rebuild_calls = {"count": 0}
    original_refresh = view.refresh

    def counting_refresh() -> None:
        rebuild_calls["count"] += 1
        original_refresh()

    view.refresh = counting_refresh  # type: ignore[method-assign]

    done = {"ok": False}

    def on_finished() -> None:
        done["ok"] = True

    view._bulk_running = True
    rebuild_calls["count"] = 0
    queued = controller.start_bulk_fit_all_skus(
        on_sku_done=view._on_bulk_sku_done,
        on_sku_failed=view._on_bulk_sku_failed,
        on_done=on_finished,
        on_error=pytest.fail,
    )
    assert queued == 15

    deadline = time.time() + 120
    while not done["ok"] and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.005)

    assert done["ok"]
    assert rebuild_calls["count"] == 0, "bulk fit should use incremental row updates, not full refresh"

    view._bulk_running = False
    view._on_bulk_done()
    qapp.processEvents()

    n_ok = sum(
        1
        for row_idx in range(view.table.rowCount())
        if view.table.item(row_idx, 8) and view.table.item(row_idx, 8).text() == "Ajustado"
    )
    assert n_ok == 15

    view.close()
    controller.shutdown()
    qapp.processEvents()
