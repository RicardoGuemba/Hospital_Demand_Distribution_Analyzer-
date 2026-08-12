"""Products list view."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hospital_demand_analyzer.app.controller import AppController
from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.ui.nav import STEP_PRODUCTS, STEP_TIME_SERIES
from hospital_demand_analyzer.ui.widgets.sortable_table import SortableTableMixin
from hospital_demand_analyzer.ui.widgets.workflow_bar import WorkflowBar

PRODUCT_TABLE_FIELDS = [
    "produto", "descricao", "sum", "mean", "non_zero_hours",
    "zero_ratio", "max", "n", "date_start", "date_end", "cluster",
]


class ProductsView(SortableTableMixin, QWidget):
    def __init__(self, state: AppState, controller: AppController, parent=None) -> None:
        super().__init__()
        self.state = state
        self.controller = controller
        self._build_ui()
        state.cluster_changed.connect(lambda _: self.refresh())
        state.clusters_updated.connect(self.populate_light)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(WorkflowBar(STEP_PRODUCTS))

        search_row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Buscar por código ou descrição...")
        self.search.textChanged.connect(self._filter)
        search_row.addWidget(self.search)
        layout.addLayout(search_row)

        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels([
            "Produto", "Descrição", "Demanda Total", "Média/h", "Horas c/ Demanda",
            "% Horas Zero", "Máximo", "N Obs", "Data Inicial", "Data Final", "Cluster",
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        self._init_sortable_table(self.table, PRODUCT_TABLE_FIELDS, default_field="non_zero_hours")
        layout.addWidget(self.table)

    def populate_light(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        df = self.state.descriptive_df
        if df is None or df.empty:
            self.table.setRowCount(0)
            return
        filtered = df.copy()
        cf = self.state.cluster_filter
        if cf != "Todos":
            filtered = filtered[filtered.get("cluster", "") == cf]
        if "cluster" not in filtered.columns:
            filtered["cluster"] = filtered["produto"].map(self.state.get_cluster_for_sku)

        self._all_data = filtered
        self._filter(self.search.text())

    def _apply_table_sort(self) -> None:
        self._filter(self.search.text())

    def _populate(self, df) -> None:
        self.table.setRowCount(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            produto = str(row["produto"])
            cluster = str(row.get("cluster") or self.state.get_cluster_for_sku(produto))
            numeric = {
                "sum": float(row.get("sum", 0)),
                "mean": float(row.get("mean", 0)),
                "non_zero_hours": float(row.get("non_zero_hours", 0)),
                "zero_ratio": float(row.get("zero_ratio", 0)),
                "max": float(row.get("max", 0)),
                "n": float(row.get("n", 0)),
            }
            items = [
                (produto, produto),
                (str(row.get("descricao", "")), produto),
                (f"{numeric['sum']:.1f}", numeric["sum"]),
                (f"{numeric['mean']:.2f}", numeric["mean"]),
                (str(int(numeric["non_zero_hours"])), numeric["non_zero_hours"]),
                (f"{numeric['zero_ratio']*100:.1f}%", numeric["zero_ratio"]),
                (f"{numeric['max']:.1f}", numeric["max"]),
                (str(int(numeric["n"])), numeric["n"]),
                (str(row.get("date_start", ""))[:10], 0),
                (str(row.get("date_end", ""))[:10], 0),
                (cluster, cluster),
            ]
            for j, (text, sort_val) in enumerate(items):
                if j <= 1:
                    item = QTableWidgetItem(text)
                    item.setData(Qt.ItemDataRole.UserRole, produto)
                elif j in (8, 9):
                    item = QTableWidgetItem(text)
                else:
                    item = self._numeric_item(text, sort_val if isinstance(sort_val, (int, float)) else 0)
                self.table.setItem(i, j, item)
        self._update_sort_indicator()

    def _filter(self, text: str) -> None:
        if not hasattr(self, "_all_data"):
            return
        df = self._all_data
        if text:
            mask = (
                df["produto"].astype(str).str.contains(text, case=False, na=False)
                | df["descricao"].astype(str).str.contains(text, case=False, na=False)
            )
            df = df[mask]
        self._populate(self._sort_dataframe(df))

    def _on_select(self) -> None:
        rows = self.table.selectedItems()
        if rows:
            produto = rows[0].data(Qt.ItemDataRole.UserRole) or rows[0].text()
            self.controller.select_sku(str(produto))

    def _on_double_click(self, row: int, _col: int) -> None:
        item = self.table.item(row, 0)
        if item:
            produto = item.data(Qt.ItemDataRole.UserRole) or item.text()
            self.controller.select_sku(str(produto))
        win = self.window()
        if hasattr(win, "navigate_to"):
            win.navigate_to(STEP_TIME_SERIES)
