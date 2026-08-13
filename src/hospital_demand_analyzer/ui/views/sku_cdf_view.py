"""Bulk CDF report — all SKUs with chosen distribution and parameters."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hospital_demand_analyzer.app.controller import AppController
from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.domain.demand_values import prepare_demand_values
from hospital_demand_analyzer.exports.cdf_report import (
    build_cdf_summary_df,
    build_cdf_summary_row,
    export_all_cdf_pngs,
    export_cdf_rankings_workbook,
    export_cdf_summary,
)
from hospital_demand_analyzer.plots.canvas import PlotWidget
from hospital_demand_analyzer.plots.cdf_bulk_plot import plot_sku_cdf_all_candidates
from hospital_demand_analyzer.statistics.fit_params import best_fit
from hospital_demand_analyzer.ui.nav import STEP_SKU_CDF
from hospital_demand_analyzer.ui.widgets.active_view import ActiveViewMixin
from hospital_demand_analyzer.ui.widgets.workflow_bar import WorkflowBar
from hospital_demand_analyzer.utils.paths import project_root

_TABLE_COLUMNS = [
    ("produto", "Produto"),
    ("descricao", "Descrição"),
    ("cluster", "Cluster"),
    ("cdf", "CDF escolhida"),
    ("params", "Parâmetros"),
    ("aic", "AIC"),
    ("bic", "BIC"),
    ("tail", "Tail Score"),
    ("status", "Status"),
]


class SkuCdfView(ActiveViewMixin, QWidget):
    def __init__(self, state: AppState, controller: AppController, parent=None) -> None:
        super().__init__(parent)
        self.__init_active_view__()
        self.state = state
        self.controller = controller
        self._use_treated = False
        self._bulk_running = False
        self._build_ui()
        state.data_loaded.connect(self.refresh_if_active)
        state.fit_completed.connect(self._on_fit_completed)
        state.clusters_updated.connect(self.refresh_if_active)
        state.value_mode_changed.connect(lambda _: self.refresh_if_active())

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(WorkflowBar(STEP_SKU_CDF))

        info = QLabel(
            "Ajuste a FDC/CDF dos SKUs em escopo (por padrão: Alto Giro) com as distribuições "
            "candidatas do catálogo atual. Altere o filtro de cluster em Configurações para "
            "processar outros grupos."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        ctrl = QHBoxLayout()
        self.series_combo = QComboBox()
        self.series_combo.addItems(["Original", "Tratada"])
        self.series_combo.currentTextChanged.connect(self._on_series_changed)
        ctrl.addWidget(QLabel("Série:"))
        ctrl.addWidget(self.series_combo)

        self.btn_fit_all = QPushButton("Ajustar todos os SKUs")
        self.btn_fit_all.clicked.connect(self._start_bulk_fit)
        ctrl.addWidget(self.btn_fit_all)

        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_bulk_fit)
        ctrl.addWidget(self.btn_cancel)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        ctrl.addWidget(self.progress, 1)

        self.status_label = QLabel("")
        ctrl.addWidget(self.status_label)
        layout.addLayout(ctrl)

        export_row = QHBoxLayout()
        for label, handler in [
            ("Exportar resumo CDF", self._export_summary),
            ("Exportar rankings completos", self._export_rankings),
            ("Exportar gráficos PNG", self._export_pngs),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(handler)
            export_row.addWidget(btn)
        export_row.addStretch()
        layout.addLayout(export_row)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget(0, len(_TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _, label in _TABLE_COLUMNS])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_select)
        splitter.addWidget(self.table)

        self.cdf_plot = PlotWidget(8, 4)
        splitter.addWidget(self.cdf_plot)
        splitter.setSizes([320, 380])
        layout.addWidget(splitter)

    def _on_series_changed(self, text: str) -> None:
        self._use_treated = text == "Tratada"
        self.refresh_if_active()

    def _on_fit_completed(self, _produto: str) -> None:
        if self._bulk_running:
            return
        self.refresh_if_active()

    def refresh(self) -> None:
        series = self._series_dict()
        if not series:
            self.table.setRowCount(0)
            self.cdf_plot.clear()
            self.status_label.setText("Carregue os dados na etapa 1.")
            return

        df = build_cdf_summary_df(
            self.state, self.controller, use_treated=self._use_treated,
        )
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(df))
            for row_idx, (_, row) in enumerate(df.iterrows()):
                self._set_table_row(row_idx, row)
            if not self._bulk_running:
                if self.table.rowCount() > 0 and not self.table.selectedItems():
                    self.table.selectRow(0)
        finally:
            self.table.blockSignals(False)

        self._update_status_counts()
        if not self._bulk_running and self.table.selectedItems():
            self._on_select()

    def _set_table_row(self, row_idx: int, row) -> None:
        produto = str(row["Produto"])
        values = [
            produto,
            str(row.get("Descricao", "")),
            str(row.get("Cluster", "")),
            str(row.get("CDF escolhida", "")),
            str(row.get("Parametros", "")),
            str(row.get("AIC", "")),
            str(row.get("BIC", "")),
            str(row.get("Tail Score", "")),
            str(row.get("Status", "")),
        ]
        for col_idx, text in enumerate(values):
            item = QTableWidgetItem(text)
            if col_idx == 0:
                item.setData(Qt.ItemDataRole.UserRole, produto)
            self.table.setItem(row_idx, col_idx, item)

    def _row_for_produto(self, produto: str) -> int:
        target = str(produto)
        for row_idx in range(self.table.rowCount()):
            item = self.table.item(row_idx, 0)
            if item is None:
                continue
            sku = str(item.data(Qt.ItemDataRole.UserRole) or item.text())
            if sku == target:
                return row_idx
        return -1

    def _update_row_for_sku(self, produto: str, fits: list | None = None, *, status: str | None = None) -> None:
        row = build_cdf_summary_row(
            self.state,
            self.controller,
            produto,
            use_treated=self._use_treated,
            fits=fits,
        )
        if status is not None:
            row["Status"] = status
            row["CDF escolhida"] = row["CDF escolhida"] or ""
        row_idx = self._row_for_produto(produto)
        if row_idx < 0:
            row_idx = self.table.rowCount()
            self.table.blockSignals(True)
            try:
                self.table.insertRow(row_idx)
                self._set_table_row(row_idx, row)
            finally:
                self.table.blockSignals(False)
            return
        self.table.blockSignals(True)
        try:
            self._set_table_row(row_idx, row)
        finally:
            self.table.blockSignals(False)

    def _update_status_counts(self) -> None:
        series = self._series_dict()
        if not series:
            self.status_label.setText("Carregue os dados na etapa 1.")
            return
        n_total = self.table.rowCount()
        n_ok = 0
        for row_idx in range(n_total):
            item = self.table.item(row_idx, 8)
            if item and item.text() == "Ajustado":
                n_ok += 1
        self.status_label.setText(f"{n_ok}/{n_total} SKUs com CDF escolhida (cache ou ajuste).")

    def _series_dict(self) -> dict:
        if self._use_treated and self.state.treated_series:
            merged = dict(self.state.hourly_series)
            merged.update(self.state.treated_series)
            return merged
        return self.state.hourly_series

    def _values_for_sku(self, produto: str):
        hs = self.state.treated_series.get(produto) if self._use_treated else None
        if hs is None:
            hs = self.state.hourly_series.get(produto)
        if hs is None:
            return None
        return prepare_demand_values(hs.quantities.values.astype(float), include_zeros=True)

    def _on_select(self) -> None:
        if self._bulk_running:
            return
        items = self.table.selectedItems()
        if not items:
            return
        produto_item = self.table.item(items[0].row(), 0)
        if produto_item is None:
            return
        produto = str(produto_item.data(Qt.ItemDataRole.UserRole) or produto_item.text())
        values = self._values_for_sku(produto)
        if values is None or len(values) == 0:
            self.cdf_plot.clear()
            return
        fits = self.controller.get_fit_results(
            produto, positive_only=False, value_mode=self.state.demand_value_mode,
        )
        plot_sku_cdf_all_candidates(
            self.cdf_plot.canvas,
            values,
            fits,
            title=f"CDF — {produto}",
            positive_only=False,
        )

    def _start_bulk_fit(self) -> None:
        if not self._series_dict():
            self.status_label.setText("Nenhum SKU disponível.")
            return

        self.btn_fit_all.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self._bulk_running = True
        self.refresh_if_active()

        n = self.controller.start_bulk_fit_all_skus(
            use_treated=self._use_treated,
            on_progress=self._on_bulk_progress,
            on_sku_done=self._on_bulk_sku_done,
            on_sku_failed=self._on_bulk_sku_failed,
            on_done=self._on_bulk_done,
            on_error=self._on_bulk_error,
        )
        if n == 0:
            self._bulk_running = False
            self._on_bulk_done()
            self.status_label.setText("Todos os SKUs já estão em cache — selecione um para ver a CDF.")

    def _on_bulk_sku_done(self, produto: str, results: list) -> None:
        self._update_row_for_sku(produto, results)
        self._update_status_counts()

    def _on_bulk_sku_failed(self, produto: str, message: str) -> None:
        self._update_row_for_sku(produto, fits=[], status=f"Erro: {message[:80]}")
        self._update_status_counts()

    def _on_bulk_progress(self, produto: str, done: int, total: int) -> None:
        if total > 0:
            self.progress.setMaximum(total)
            self.progress.setValue(min(done, total))
        if produto:
            self.status_label.setText(f"Ajustando {produto} ({done + 1}/{total})...")

    def _on_bulk_done(self) -> None:
        self._bulk_running = False
        self.btn_fit_all.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setVisible(False)
        self._update_status_counts()
        if self.table.selectedItems():
            self._on_select()
        elif self.table.rowCount() > 0:
            self.table.blockSignals(True)
            try:
                self.table.selectRow(0)
            finally:
                self.table.blockSignals(False)
            self._on_select()

    def _on_bulk_error(self, message: str) -> None:
        self._bulk_running = False
        self._on_bulk_done()
        self.status_label.setText(f"Erro no ajuste em lote: {message}")

    def _cancel_bulk_fit(self) -> None:
        self.controller.cancel_bulk_fit()
        self.status_label.setText("Cancelamento solicitado...")
        self.btn_cancel.setEnabled(False)

    def _export_summary(self) -> None:
        df = build_cdf_summary_df(
            self.state, self.controller, use_treated=self._use_treated,
        )
        if df.empty:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar resumo CDF",
            str(project_root() / "export_cdf_resumo.csv"),
            "CSV (*.csv);;Excel (*.xlsx)",
        )
        if not path:
            return
        fmt = "xlsx" if path.lower().endswith(".xlsx") else "csv"
        export_cdf_summary(df, Path(path), fmt)

    def _export_rankings(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar rankings CDF",
            str(project_root() / "export_cdf_rankings.csv"),
            "CSV (*.csv);;Excel (*.xlsx)",
        )
        if not path:
            return
        export_cdf_rankings_workbook(
            self.state, self.controller, Path(path), use_treated=self._use_treated,
        )

    def _export_pngs(self) -> None:
        out_dir = QFileDialog.getExistingDirectory(
            self, "Pasta para gráficos PNG", str(project_root()),
        )
        if not out_dir:
            return
        n = export_all_cdf_pngs(
            self.state, self.controller, Path(out_dir), use_treated=self._use_treated,
        )
        self.status_label.setText(f"{n} gráficos PNG exportados em {out_dir}")
