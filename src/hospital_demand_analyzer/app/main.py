"""Application bootstrap."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.ui.windows.main_window import MainWindow
from hospital_demand_analyzer.utils.config import AppConfig
from hospital_demand_analyzer.utils.logging_setup import setup_logging


def run() -> None:
    config = AppConfig.load()
    setup_logging(config.logs_path())
    app = QApplication(sys.argv)
    app.setApplicationName("Hospital Demand Analyzer")
    state = AppState(config)
    window = MainWindow(state)
    window.show()
    sys.exit(app.exec())
