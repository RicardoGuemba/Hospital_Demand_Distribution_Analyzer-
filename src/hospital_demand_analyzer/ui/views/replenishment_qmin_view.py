"""Replenishment Qmin Monte Carlo view — session SKUs only."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hospital_demand_analyzer.app.controller import AppController
from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.domain.models import ReplenishmentQminResult
from hospital_demand_analyzer.exports.replenishment_qmin_report import (
    build_replenishment_summary_df,
    build_replenishment_summary_row,
    export_replenishment_summary,
)
from hospital_demand_analyzer.plots.canvas import PlotWidget
from hospital_demand_analyzer.plots.replenishment_qmin_plot import (
    plot_demand_accumulation_trajectories,
    save_replenishment_qmin_png,
)

from hospital_demand_analyzer.statistics.replenishment_qmin import DEFAULT_REPLENISHMENT_MINUTES
from hospital_demand_analyzer.ui.nav import STEP_REPLENISHMENT_QMIN
from hospital_demand_analyzer.ui.widgets.active_view import ActiveViewMixin
from hospital_demand_analyzer.ui.widgets.workflow_bar import WorkflowBar
from hospital_demand_analyzer.utils.paths import project_root

_TABLE_COLUMNS = [
    "Produto", "Descrição", "Cluster", "CDF escolhida", "Parâmetros",
    "T reabast. (min)", "Saldo inicial", "Qmin P95", "Mediana", "Q1", "Q3",
    "Média", "P99", "N traj", "Status",
]


class ReplenishmentQminView(ActiveViewMixin, QWidget):
    def __init__(self, state: AppState, controller: AppController, parent=None) -> None:
        super().__init__(parent)
        self.__init_active_view__()
        self.state = state
        self.controller = controller
        self._batch_running = False
        self._last_run_params: tuple[int, int] | None = None
        self._build_ui()
        state.session_skus_changed.connect(self.refresh_if_active)
        state.data_loaded.connect(self.refresh_if_active)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(WorkflowBar(STEP_REPLENISHMENT_QMIN))

        info = QLabel(
            "Simula o consumo hora a hora da FDC ajustada até esgotar o tempo de reabastecimento. "
            "O gráfico exibe saldo decrescente (como Monte Carlo): parte do saldo inicial e desce "
            "até o fim do intervalo; linhas de referência marcam Qmin (P95) e saldo inicial. "
            "Clique em «Calcular Qmin — SKUs da sessão» para simular todos os produtos selecionados "
            "nas etapas 4–8."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.session_label = QLabel("SKUs da sessão: 0")
        self.session_label.setWordWrap(True)
        layout.addWidget(self.session_label)

        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Tempo reabastecimento (min):"))
        self.replenish_spin = QSpinBox()
        self.replenish_spin.setRange(1, 10_080)
        self.replenish_spin.setValue(int(DEFAULT_REPLENISHMENT_MINUTES))
        params_row.addWidget(self.replenish_spin)

        params_row.addWidget(QLabel("N trajetórias:"))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 50_000)
        self.n_spin.setValue(100)
        params_row.addWidget(self.n_spin)
        params_row.addStretch()
        layout.addLayout(params_row)
        for spin in (self.replenish_spin, self.n_spin):
            spin.valueChanged.connect(self._on_simulation_params_changed)

        action_row = QHBoxLayout()
        self.btn_run = QPushButton("Calcular Qmin — SKUs da sessão")
        self.btn_run.setToolTip(
            "Simula demanda acumulada no tempo de reabastecimento para cada SKU selecionado na sessão"
        )
        self.btn_run.setMinimumWidth(260)
        self.btn_run.clicked.connect(self._start_batch)
        action_row.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_batch)
        action_row.addWidget(self.btn_cancel)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        action_row.addWidget(self.progress, 1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        action_row.addWidget(self.status_label, 1)
        layout.addLayout(action_row)

        export_row = QHBoxLayout()
        for label, handler in [
            ("Exportar resumo Qmin", self._export_summary),
            ("Exportar gráfico PNG", self._export_png),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(handler)
            export_row.addWidget(btn)
        export_row.addStretch()
        layout.addLayout(export_row)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget(0, len(_TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(_TABLE_COLUMNS)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_select)
        splitter.addWidget(self.table)

        self.traj_plot = PlotWidget(8, 4)
        splitter.addWidget(self.traj_plot)
        splitter.setSizes([320, 380])
        layout.addWidget(splitter)

    def refresh(self) -> None:
        skus = [s for s in self.state.session_sku_order if s in self.state.session_selected_skus]
        if skus:
            self.session_label.setText(
                f"SKUs da sessão ({len(skus)}): " + ", ".join(skus[:12])
                + (" …" if len(skus) > 12 else "")
            )
        else:
            self.session_label.setText(
                "Nenhum SKU selecionado nesta sessão — selecione produtos nas etapas 4–8."
            )

        df = build_replenishment_summary_df(self.state)
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(df))
            for row_idx, (_, row) in enumerate(df.iterrows()):
                self._set_table_row(row_idx, row)
            if not self._batch_running and self.table.rowCount() > 0 and not self.table.selectedItems():
                self.table.selectRow(0)
        finally:
            self.table.blockSignals(False)

        self._update_status_counts()
        if not self._batch_running and self.table.selectedItems():
            self._on_select()

    def _set_table_row(self, row_idx: int, row) -> None:
        produto = str(row.get("Produto", ""))
        values = [
            produto,
            str(row.get("Descricao", "")),
            str(row.get("Cluster", "")),
            str(row.get("CDF escolhida", "")),
            str(row.get("Parametros", "")),
            str(row.get("Tempo reabastecimento (min)", "")),
            str(row.get("Saldo inicial", "")),
            str(row.get("Qmin proposta P95", "")),
            str(row.get("Mediana", "")),
            str(row.get("Q1", "")),
            str(row.get("Q3", "")),
            str(row.get("Media", "")),
            str(row.get("P99", "")),
            str(row.get("N trajetorias", "")),
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

    def _update_row_for_sku(self, produto: str, result: ReplenishmentQminResult | None = None) -> None:
        row = build_replenishment_summary_row(self.state, produto, result)
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
        n_ok = 0
        for row_idx in range(self.table.rowCount()):
            item = self.table.item(row_idx, 14)
            if item and item.text() == "Simulado":
                n_ok += 1
        total = self.table.rowCount()
        self.status_label.setText(f"{n_ok}/{total} SKUs simulados (sessão corrente).")

    def _selected_produto(self) -> str | None:
        items = self.table.selectedItems()
        if not items:
            return None
        produto_item = self.table.item(items[0].row(), 0)
        if produto_item is None:
            return None
        return str(produto_item.data(Qt.ItemDataRole.UserRole) or produto_item.text())

    def _current_run_params(self) -> tuple[int, int]:
        return (self.replenish_spin.value(), self.n_spin.value())

    def _on_simulation_params_changed(self, *_args) -> None:
        if self._batch_running:
            return
        if self._last_run_params is None:
            return
        if self._current_run_params() == self._last_run_params:
            return
        self.traj_plot.clear()
        self.status_label.setText("Parâmetros alterados — execute nova simulação.")

    def _render_plot(self, produto: str) -> None:
        result = self.state.replenishment_qmin_results.get(produto)
        if result is None:
            self.traj_plot.clear()
            return
        plot_demand_accumulation_trajectories(
            self.traj_plot.canvas,
            result,
            title=f"Saldo simulado — {produto}",
        )
        self.traj_plot.canvas.update()

    def _on_select(self) -> None:
        if self._batch_running:
            return
        produto = self._selected_produto()
        if produto is None:
            self.traj_plot.clear()
            return
        self._render_plot(produto)

    def _start_batch(self) -> None:
        if not self.state.session_selected_skus:
            self.status_label.setText("Selecione ao menos um SKU nas etapas anteriores.")
            return

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self._batch_running = True
        self.traj_plot.clear()
        self.refresh_if_active()

        n = self.controller.start_replenishment_qmin_batch(
            replenishment_minutes=float(self.replenish_spin.value()),
            n_trajetorias=self.n_spin.value(),
            on_progress=self._on_progress,
            on_sku_done=self._on_sku_done,
            on_sku_failed=self._on_sku_failed,
            on_done=self._on_done,
            on_error=self._on_error,
        )
        if n == 0:
            self._batch_running = False
            self._on_done()
            self.status_label.setText(
                "Nenhum SKU da sessão com série horária disponível."
            )

    def _on_progress(self, produto: str, done: int, total: int) -> None:
        if total > 0:
            self.progress.setMaximum(total)
            self.progress.setValue(min(done, total))
        if produto:
            self.status_label.setText(f"Simulando {produto} ({done + 1}/{total})...")

    def _on_sku_done(self, produto: str, result: ReplenishmentQminResult) -> None:
        self._update_row_for_sku(produto, result)
        self._update_status_counts()
        if produto == self._selected_produto():
            self._render_plot(produto)

    def _on_sku_failed(self, produto: str, message: str) -> None:
        row_idx = self._row_for_produto(produto)
        if row_idx < 0:
            self.table.blockSignals(True)
            try:
                row_idx = self.table.rowCount()
                self.table.insertRow(row_idx)
                item = QTableWidgetItem(str(produto))
                item.setData(Qt.ItemDataRole.UserRole, str(produto))
                self.table.setItem(row_idx, 0, item)
                self.table.setItem(row_idx, 14, QTableWidgetItem(f"Erro: {message[:60]}"))
            finally:
                self.table.blockSignals(False)
        else:
            self.table.setItem(row_idx, 14, QTableWidgetItem(f"Erro: {message[:60]}"))
        self._update_status_counts()

    def _on_done(self) -> None:
        self._batch_running = False
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setVisible(False)
        self._last_run_params = self._current_run_params()
        self._update_status_counts()
        produto = self._selected_produto()
        if produto is None and self.table.rowCount() > 0:
            self.table.blockSignals(True)
            try:
                self.table.selectRow(0)
            finally:
                self.table.blockSignals(False)
            produto = self._selected_produto()
        if produto:
            self._render_plot(produto)

    def _on_error(self, message: str) -> None:
        self._batch_running = False
        self._on_done()
        self.status_label.setText(f"Erro na simulação: {message}")

    def _cancel_batch(self) -> None:
        self.controller.cancel_replenishment_qmin_batch()
        self.status_label.setText("Cancelamento solicitado...")
        self.btn_cancel.setEnabled(False)

    def _export_summary(self) -> None:
        df = build_replenishment_summary_df(self.state)
        if df.empty:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar resumo Qmin reabastecimento",
            str(project_root() / "export_qmin_reabastecimento.csv"),
            "CSV (*.csv);;Excel (*.xlsx)",
        )
        if not path:
            return
        fmt = "xlsx" if path.lower().endswith(".xlsx") else "csv"
        export_replenishment_summary(df, Path(path), fmt)

    def _export_png(self) -> None:
        items = self.table.selectedItems()
        if not items:
            return
        produto_item = self.table.item(items[0].row(), 0)
        if produto_item is None:
            return
        produto = str(produto_item.data(Qt.ItemDataRole.UserRole) or produto_item.text())
        result = self.state.replenishment_qmin_results.get(produto)
        if result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar gráfico PNG",
            str(project_root() / f"{produto}_qmin_reabastecimento.png"),
            "PNG (*.png)",
        )
        if path:
            save_replenishment_qmin_png(Path(path), result, title=f"Demanda acumulada — {produto}")
