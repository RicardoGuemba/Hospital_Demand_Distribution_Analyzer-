"""Tests for background worker lifecycle."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from hospital_demand_analyzer.app.controller import AppController
from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.app.workers import ImportWorker


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


def test_import_worker_completes(qapp, host_widget):
    data_dir = Path(__file__).resolve().parents[1] / "Dados_Demanda"
    if not data_dir.is_dir():
        pytest.skip("Dados_Demanda not available")

    worker = ImportWorker(data_dir)
    worker.setParent(host_widget)
    captured: dict = {}

    worker.finished_ok.connect(lambda payload: captured.update({"payload": payload}))
    worker.finished_err.connect(lambda msg: pytest.fail(msg))
    worker.start()

    deadline = time.time() + 120
    while worker.isRunning() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)

    assert worker.wait(5000)
    qapp.processEvents()
    assert "payload" in captured

    result, series, desc = captured["payload"]
    assert result.total_valid_rows > 0
    assert len(series) > 0
    assert not desc.empty
    assert "skewness" in desc.columns


def test_controller_shutdown_does_not_crash(qapp, host_widget):
    data_dir = Path(__file__).resolve().parents[1] / "Dados_Demanda"
    if not data_dir.is_dir():
        pytest.skip("Dados_Demanda not available")

    controller = AppController(AppState(), host_widget)
    controller.load_directory(data_dir)
    controller.shutdown()
    qapp.processEvents()
    assert controller._import_worker is None


def test_controller_import_updates_state(qapp, host_widget):
    data_dir = Path(__file__).resolve().parents[1] / "Dados_Demanda"
    if not data_dir.is_dir():
        pytest.skip("Dados_Demanda not available")

    state = AppState()
    controller = AppController(state, host_widget)
    done = {"ok": False}

    def on_done(payload) -> None:
        controller.handle_import_result(payload)
        done["ok"] = True

    controller.load_directory(data_dir, on_done=on_done, on_error=pytest.fail)

    deadline = time.time() + 120
    while not done["ok"] and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)

    controller.shutdown()
    qapp.processEvents()
    assert done["ok"]
    assert state.import_result is not None
    assert not state.descriptive_df.empty
