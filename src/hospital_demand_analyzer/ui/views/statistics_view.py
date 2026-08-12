"""Per-SKU descriptive statistics view."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hospital_demand_analyzer.app.controller import AppController
from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.statistics.descriptive import (
    DESCRIPTIVE_TABLE_COLUMNS,
    METRIC_TOOLTIPS,
)
from hospital_demand_analyzer.statistics.distribution_moments import theoretical_expected_value
from hospital_demand_analyzer.statistics.fitter import top_n
from hospital_demand_analyzer.ui.nav import STEP_STATISTICS, STEP_TIME_SERIES
from hospital_demand_analyzer.ui.widgets.sortable_table import SortableTableMixin
from hospital_demand_analyzer.ui.widgets.workflow_bar import WorkflowBar

_CENTRAL_KEYS = frozenset({
    "mean", "expected_empirical", "expected_conditional", "median", "q50",
})


def _format_cell(key: str, value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    if key == "zero_ratio":
        return f"{float(value) * 100:.1f}%"
    if key in ("produto", "descricao", "cluster"):
        return str(value)
    if key in ("n", "n_zeros", "non_zero_hours"):
        return str(int(value))
    if isinstance(value, float):
        if key in ("skewness", "kurtosis", "cv"):
            return f"{value:.3f}"
        if key in _CENTRAL_KEYS or key in ("std", "var", "q25", "q75", "iqr"):
            return f"{value:.4f}"
        return f"{value:.2f}"
    return str(value)


DETAIL_GROUPS: list[tuple[str, list[str]]] = [
    ("Volume e cobertura", ["n", "non_zero_hours", "n_zeros", "zero_ratio", "sum"]),
    ("Tendência central", ["mean", "expected_empirical", "expected_conditional", "median", "q50"]),
    ("Dispersão", ["std", "var", "cv", "iqr", "min", "max"]),
    ("Percentis", ["q25", "q75", "p90", "p95", "p975", "p99"]),
    ("Forma", ["skewness", "kurtosis"]),
]

DETAIL_LABELS = dict(DESCRIPTIVE_TABLE_COLUMNS)
DETAIL_LABELS["expected_theoretical"] = "E[X] teórico (ajuste)"


class StatisticsView(SortableTableMixin, QWidget):
    def __init__(self, state: AppState, controller: AppController, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.controller = controller
        self._detail_produto: str | None = None
        self._column_keys = [k for k, _ in DESCRIPTIVE_TABLE_COLUMNS]
        self._build_ui()
        state.sku_changed.connect(self._sync_selection)
        state.cluster_changed.connect(lambda _: self.populate_light())
        state.clusters_updated.connect(self.populate_light)
        state.fit_completed.connect(self._on_fit_completed)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(WorkflowBar(STEP_STATISTICS))

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Buscar:"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Código ou descrição do produto...")
        self.search.textChanged.connect(self._apply_filter)
        search_row.addWidget(self.search, 1)
        self.count_label = QLabel("0 SKUs")
        search_row.addWidget(self.count_label)
        layout.addLayout(search_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.table = QTableWidget(0, len(DESCRIPTIVE_TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _, label in DESCRIPTIVE_TABLE_COLUMNS])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self._init_sortable_table(self.table, self._column_keys, default_field="non_zero_hours")
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        splitter.addWidget(self.table)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        self.detail_title = QLabel("Selecione um SKU na tabela.")
        self.detail_title.setWordWrap(True)
        self.detail_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        detail_layout.addWidget(self.detail_title)

        self.detail_grid = QGridLayout()
        detail_layout.addLayout(self.detail_grid)
        detail_layout.addStretch()
        splitter.addWidget(detail)
        splitter.setSizes([900, 320])
        layout.addWidget(splitter)

    def populate_light(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        df = self.state.descriptive_df
        if df is None or df.empty:
            self.table.setRowCount(0)
            self.count_label.setText("0 SKUs")
            self.detail_title.setText("Carregue os dados na etapa 1.")
            self._clear_detail()
            return

        work = df.copy()
        if "cluster" not in work.columns:
            work["cluster"] = work["produto"].map(self.state.get_cluster_for_sku)
        self._all_data = work
        self._apply_filter(self.search.text())
        if self._detail_produto:
            self._show_detail(self._detail_produto)

    def _apply_table_sort(self) -> None:
        if hasattr(self, "_all_data"):
            self._apply_filter(self.search.text())

    def _apply_filter(self, text: str = "") -> None:
        if not hasattr(self, "_all_data"):
            return
        df = self._all_data
        if text.strip():
            mask = (
                df["produto"].astype(str).str.contains(text, case=False, na=False)
                | df["descricao"].astype(str).str.contains(text, case=False, na=False)
            )
            df = df[mask]
        df = self._sort_dataframe(df)
        self._populate_table(df)
        self.count_label.setText(f"{len(df)} SKUs")

    def _populate_table(self, df) -> None:
        self.table.setRowCount(len(df))
        for row_idx, (_, row) in enumerate(df.iterrows()):
            produto = str(row["produto"])
            for col_idx, key in enumerate(self._column_keys):
                raw = row.get(key, "")
                if key == "cluster" and (raw is None or raw == ""):
                    raw = self.state.get_cluster_for_sku(produto)
                item = QTableWidgetItem(_format_cell(key, raw))
                if key not in ("produto", "descricao", "cluster"):
                    try:
                        sort_val = float(raw) if raw != "" and raw == raw else 0.0
                    except (TypeError, ValueError):
                        sort_val = 0.0
                    item.setData(Qt.ItemDataRole.UserRole, sort_val)
                if col_idx == 0:
                    item.setData(Qt.ItemDataRole.UserRole + 1, produto)
                self.table.setItem(row_idx, col_idx, item)
        self._update_sort_indicator()
        sku = self.state.selected_sku
        if sku:
            self.table.blockSignals(True)
            self._sync_selection(sku)
            self.table.blockSignals(False)

    def _sync_selection(self, sku: str) -> None:
        if not sku:
            return
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.data(Qt.ItemDataRole.UserRole + 1) == sku:
                self.table.selectRow(row)
                break

    def _on_select(self) -> None:
        rows = self.table.selectedItems()
        if not rows:
            return
        produto_item = self.table.item(rows[0].row(), 0)
        if produto_item is None:
            return
        produto = produto_item.data(Qt.ItemDataRole.UserRole + 1) or produto_item.text()
        self.controller.select_sku(str(produto))
        self._show_detail(str(produto))

    def _theoretical_expected_text(self, produto: str) -> str:
        fits = self.controller.get_fit_results(
            produto, positive_only=False, value_mode=self.state.demand_value_mode,
        )
        if not fits:
            self.controller.request_fit(
                produto, positive_only=False, value_mode=self.state.demand_value_mode,
            )
            return "—"
        best = top_n(fits, 1)
        if not best:
            return "—"
        val = theoretical_expected_value(best[0])
        return f"{val:.4f}" if val is not None else "—"

    def _add_detail_row(self, grid: QGridLayout, row: int, key: str, value: str) -> None:
        label_text = DETAIL_LABELS.get(key, key)
        name_lbl = QLabel(f"{label_text}:")
        tooltip = METRIC_TOOLTIPS.get(key)
        if tooltip:
            name_lbl.setToolTip(tooltip)
        grid.addWidget(name_lbl, row, 0)
        val_lbl = QLabel(value)
        val_lbl.setStyleSheet("font-weight: bold;")
        if tooltip:
            val_lbl.setToolTip(tooltip)
        grid.addWidget(val_lbl, row, 1)

    def _show_detail(self, produto: str) -> None:
        df = self._all_data if hasattr(self, "_all_data") else self.state.descriptive_df
        if df is None or df.empty:
            return
        match = df[df["produto"].astype(str) == produto]
        if match.empty:
            return
        self._detail_produto = produto
        row = match.iloc[0]
        desc = str(row.get("descricao", ""))
        cluster = row.get("cluster") or self.state.get_cluster_for_sku(produto)
        self.detail_title.setText(f"{produto} — {desc}\nCluster: {cluster or '—'}")
        self._clear_detail()
        grid_row = 0
        for group_name, keys in DETAIL_GROUPS:
            box = QGroupBox(group_name)
            grid = QGridLayout(box)
            for i, key in enumerate(keys):
                self._add_detail_row(grid, i, key, _format_cell(key, row.get(key)))
            if group_name == "Tendência central":
                self._add_detail_row(
                    grid, len(keys), "expected_theoretical",
                    self._theoretical_expected_text(produto),
                )
            self.detail_grid.addWidget(box, grid_row // 2, grid_row % 2)
            grid_row += 1

    def _on_fit_completed(self, produto: str) -> None:
        if self._detail_produto and str(produto) == self._detail_produto:
            self._show_detail(self._detail_produto)

    def _clear_detail(self) -> None:
        while self.detail_grid.count():
            item = self.detail_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _on_double_click(self, row: int, _col: int) -> None:
        item = self.table.item(row, 0)
        if item:
            produto = item.data(Qt.ItemDataRole.UserRole + 1) or item.text()
            self.controller.select_sku(str(produto))
        win = self.window()
        if hasattr(win, "navigate_to"):
            win.navigate_to(STEP_TIME_SERIES)
