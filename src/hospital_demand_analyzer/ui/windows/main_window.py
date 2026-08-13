"""Main application window."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from hospital_demand_analyzer.app.controller import AppController
from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.ui.nav import (
    NAV_ENTRIES,
    STEP_CLUSTERS,
    STEP_DATA,
    STEP_DISTRIBUTIONS,
    STEP_EXPORT,
    STEP_MONTE_CARLO,
    STEP_OUTLIERS,
    STEP_PRODUCTS,
    STEP_REPLENISHMENT_QMIN,
    STEP_SETTINGS,
    STEP_SKU_CDF,
    STEP_STATISTICS,
    STEP_TIME_SERIES,
)
from hospital_demand_analyzer.ui.views.clusters_view import ClustersView
from hospital_demand_analyzer.ui.views.data_view import DataView
from hospital_demand_analyzer.ui.views.distributions_view import DistributionsView
from hospital_demand_analyzer.ui.views.export_view import ExportView
from hospital_demand_analyzer.ui.views.monte_carlo_view import MonteCarloView
from hospital_demand_analyzer.ui.views.outliers_view import OutliersView
from hospital_demand_analyzer.ui.views.products_view import ProductsView
from hospital_demand_analyzer.ui.views.replenishment_qmin_view import ReplenishmentQminView
from hospital_demand_analyzer.ui.views.settings_view import SettingsView
from hospital_demand_analyzer.ui.views.sku_cdf_view import SkuCdfView
from hospital_demand_analyzer.ui.views.statistics_view import StatisticsView
from hospital_demand_analyzer.ui.views.time_series_view import TimeSeriesView


class MainWindow(QMainWindow):
    def __init__(self, state: AppState, *, auto_load: bool = True) -> None:
        super().__init__()
        self.state = state
        self.controller = AppController(state, self)
        self.setWindowTitle("Hospital Demand Distribution Analyzer")
        self.resize(1400, 900)
        self._row_to_stack: dict[int, int] = {}
        self._key_to_row: dict[str, int] = {}
        self._build_ui()
        state.data_loaded.connect(self._on_data_loaded)
        if auto_load:
            QTimer.singleShot(200, self._auto_load)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self.header = QFrame()
        self.header.setFrameStyle(QFrame.Shape.StyledPanel)
        header_layout = QHBoxLayout(self.header)
        self.lbl_dir = QLabel("Diretório: —")
        self.lbl_files = QLabel("Arquivos: —")
        self.lbl_skus = QLabel("SKUs: —")
        self.lbl_records = QLabel("Registros: —")
        self.lbl_period = QLabel("Período: —")
        self.lbl_updated = QLabel("Atualizado: —")
        for w in [self.lbl_dir, self.lbl_files, self.lbl_skus,
                  self.lbl_records, self.lbl_period, self.lbl_updated]:
            header_layout.addWidget(w)
        main_layout.addWidget(self.header)

        body = QHBoxLayout()
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(210)
        self.sidebar.currentRowChanged.connect(self._on_nav)
        body.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        self.views = {
            STEP_DATA: DataView(self.state, self.controller),
            STEP_OUTLIERS: OutliersView(self.state, self.controller),
            STEP_CLUSTERS: ClustersView(self.state, self.controller),
            STEP_PRODUCTS: ProductsView(self.state, self.controller),
            STEP_STATISTICS: StatisticsView(self.state, self.controller),
            STEP_TIME_SERIES: TimeSeriesView(self.state, self.controller),
            STEP_DISTRIBUTIONS: DistributionsView(self.state, self.controller),
            STEP_MONTE_CARLO: MonteCarloView(self.state, self.controller),
            STEP_SKU_CDF: SkuCdfView(self.state, self.controller),
            STEP_REPLENISHMENT_QMIN: ReplenishmentQminView(self.state, self.controller),
            STEP_EXPORT: ExportView(self.state, self.controller),
            STEP_SETTINGS: SettingsView(self.state),
        }
        for key in (
            STEP_DATA, STEP_OUTLIERS, STEP_CLUSTERS, STEP_PRODUCTS, STEP_STATISTICS,
            STEP_TIME_SERIES, STEP_DISTRIBUTIONS, STEP_MONTE_CARLO, STEP_SKU_CDF,
            STEP_REPLENISHMENT_QMIN, STEP_EXPORT, STEP_SETTINGS,
        ):
            self.stack.addWidget(self.views[key])

        stack_idx = 0
        for row, entry in enumerate(NAV_ENTRIES):
            item = QListWidgetItem(entry.label)
            if entry.key is None:
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                item.setForeground(QColor("#666666"))
                font = QFont()
                font.setBold(True)
                font.setPointSize(9)
                item.setFont(font)
            else:
                self._row_to_stack[row] = stack_idx
                self._key_to_row[entry.key] = row
                stack_idx += 1
            self.sidebar.addItem(item)
        body.addWidget(self.stack, 1)
        main_layout.addLayout(body)

        self.navigate_to(STEP_DATA)

    def _on_nav(self, row: int) -> None:
        stack_idx = self._row_to_stack.get(row)
        if stack_idx is not None:
            self.stack.setCurrentIndex(stack_idx)

    def _on_data_loaded(self) -> None:
        self._update_header()
        QTimer.singleShot(0, self._refresh_after_load)

    def _refresh_after_load(self) -> None:
        current = self.stack.currentWidget()
        for view in self.views.values():
            if view is current:
                if hasattr(view, "refresh"):
                    view.refresh()
            else:
                populate = getattr(view, "populate_light", None)
                if callable(populate):
                    populate()

    def navigate_to(self, key: str) -> None:
        row = self._key_to_row.get(key)
        if row is not None:
            self.sidebar.setCurrentRow(row)

    def _update_header(self) -> None:
        r = self.state.import_result
        if r is None:
            return
        self.lbl_dir.setText(f"Diretório: {r.data_dir}")
        self.lbl_files.setText(f"Arquivos: {r.total_files}")
        self.lbl_skus.setText(f"SKUs: {r.unique_skus}")
        self.lbl_records.setText(f"Registros: {r.total_valid_rows}")
        if r.date_min and r.date_max:
            self.lbl_period.setText(
                f"Período: {r.date_min.strftime('%Y-%m-%d')} — {r.date_max.strftime('%Y-%m-%d')}")
        if self.state.last_updated:
            self.lbl_updated.setText(
                f"Atualizado: {self.state.last_updated.strftime('%Y-%m-%d %H:%M')}")

    def closeEvent(self, event) -> None:
        self.controller.shutdown()
        super().closeEvent(event)

    def _auto_load(self) -> None:
        data_dir = self.state.data_dir
        if not data_dir.is_dir():
            return
        self.lbl_updated.setText("Atualizado: importando...")
        self.controller.load_directory(
            data_dir,
            on_done=self._on_import_done,
            on_error=self.controller.show_error,
        )

    def _on_import_done(self, payload) -> None:
        self.controller.handle_import_result(payload)
