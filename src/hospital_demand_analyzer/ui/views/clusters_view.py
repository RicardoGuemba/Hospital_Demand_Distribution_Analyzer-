"""Clusters view."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hospital_demand_analyzer.app.controller import AppController
from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.clustering.cutoffs import (
    CUTOFF_FIELDS,
    preview_cluster_counts,
    suggest_threshold_cutoffs,
)
from hospital_demand_analyzer.plots.cluster_plot import plot_cluster_scatter
from hospital_demand_analyzer.ui.nav import STEP_CLUSTERS, STEP_PRODUCTS
from hospital_demand_analyzer.ui.widgets.workflow_bar import WorkflowBar


class ClusterCard(QFrame):
    clicked = None

    def __init__(self, name: str, color: str, parent=None) -> None:
        super().__init__(parent)
        self.name = name
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setStyleSheet(f"border-left: 4px solid {color};")
        self.layout = QVBoxLayout(self)
        self.title = QLabel(name)
        self.title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.body = QLabel("—")
        self.body.setWordWrap(True)
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.body)

    def set_stats(self, n: int, pct: float, mean_d: float, freq: float, p95: float) -> None:
        self.body.setText(
            f"SKUs: {n} ({pct:.1f}%)\n"
            f"Média: {mean_d:.2f}/h\n"
            f"Frequência: {freq:.2%}\n"
            f"P95: {p95:.1f}"
        )

    def mousePressEvent(self, event) -> None:
        if self.clicked:
            self.clicked(self.name)
        super().mousePressEvent(event)


class ClustersView(QWidget):
    def __init__(self, state: AppState, controller: AppController, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.controller = controller
        self._recalc_running = False
        self._recalc_timer = QTimer(self)
        self._recalc_timer.setSingleShot(True)
        self._recalc_timer.setInterval(450)
        self._recalc_timer.timeout.connect(self._recalculate)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(WorkflowBar(STEP_CLUSTERS))

        header = QHBoxLayout()
        self.header_label = QLabel("Clusters não calculados.")
        self.btn_recalc = QPushButton("Recalcular Clusters")
        self.btn_recalc.clicked.connect(self._recalculate)
        btn_suggest = QPushButton("Sugerir Cortes (P25/P75)")
        btn_suggest.clicked.connect(self._apply_suggested_cutoffs)
        btn_study = QPushButton("Estudar Produtos →")
        btn_study.clicked.connect(self._study_products)
        header.addWidget(self.header_label, 1)
        header.addWidget(btn_suggest)
        header.addWidget(self.btn_recalc)
        header.addWidget(btn_study)
        layout.addLayout(header)

        metric_row = QHBoxLayout()
        metric_row.addWidget(QLabel("Métrica de corte:"))
        self.metric_combo = QComboBox()
        for key, label in CUTOFF_FIELDS.items():
            self.metric_combo.addItem(label, key)
        idx = self.metric_combo.findData(self.state.cluster_cutoff_field)
        if idx >= 0:
            self.metric_combo.setCurrentIndex(idx)
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        metric_row.addWidget(self.metric_combo, 1)
        layout.addLayout(metric_row)

        cutoff_row = QHBoxLayout()
        cutoff_row.addWidget(QLabel("Corte Alto Giro (≥):"))
        self.alto_spin = QDoubleSpinBox()
        self.alto_spin.setRange(0.001, 10000.0)
        self.alto_spin.setDecimals(3)
        self.alto_spin.setSingleStep(0.1)
        self.alto_spin.setValue(self.state.cluster_alto_cutoff)
        self.alto_spin.valueChanged.connect(self._on_cutoff_edited)
        cutoff_row.addWidget(self.alto_spin)

        cutoff_row.addWidget(QLabel("Corte Baixo Giro (<):"))
        self.baixo_spin = QDoubleSpinBox()
        self.baixo_spin.setRange(0.0, 10000.0)
        self.baixo_spin.setDecimals(3)
        self.baixo_spin.setSingleStep(0.05)
        self.baixo_spin.setValue(self.state.cluster_baixo_cutoff)
        self.baixo_spin.valueChanged.connect(self._on_cutoff_edited)
        cutoff_row.addWidget(self.baixo_spin)
        cutoff_row.addStretch()
        layout.addLayout(cutoff_row)

        self.preview_label = QLabel("Prévia: —")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("color: #444; font-size: 11px; background: #f8f9fa; padding: 6px;")
        layout.addWidget(self.preview_label)

        hint = QLabel(
            "Os cortes recalculam automaticamente ao alterar valores. "
            "Diminuir «Alto Giro» inclui mais SKUs no cluster; aumentar reduz."
        )
        hint.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(hint)

        cards = QHBoxLayout()
        self.card_baixo = ClusterCard("Baixo Giro", "#3498db")
        self.card_medio = ClusterCard("Medio Giro", "#f39c12")
        self.card_alto = ClusterCard("Alto Giro", "#e74c3c")
        for card in [self.card_baixo, self.card_medio, self.card_alto]:
            card.clicked = self._select_cluster
            cards.addWidget(card)
        layout.addLayout(cards)

        from hospital_demand_analyzer.plots.canvas import PlotWidget
        self.scatter = PlotWidget(8, 4)
        layout.addWidget(self.scatter)

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels([
            "Produto", "Descrição", "Cluster", "Demanda/h (critério)", "Frequência",
            "P95", "P99", "CV", "Turnover Score", "Dist. Centroide",
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        self._cluster_sort_field = "cutoff_metric_value"
        self._cluster_sort_desc = True
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().setSortIndicatorShown(True)
        self.table.horizontalHeader().sectionClicked.connect(self._on_cluster_table_sort)
        self._cluster_table_fields = [
            "produto", "descricao", "cluster_name", "cutoff_metric_value",
            "non_zero_hour_ratio", "p95", "p99", "cv", "turnover_score", "distance_to_centroid",
        ]
        self._last_assignments = []

        self._update_preview()

    def _on_cluster_table_sort(self, column: int) -> None:
        if column < 0 or column >= len(self._cluster_table_fields):
            return
        field = self._cluster_table_fields[column]
        if self._cluster_sort_field == field:
            self._cluster_sort_desc = not self._cluster_sort_desc
        else:
            self._cluster_sort_field = field
            self._cluster_sort_desc = True
        order = Qt.SortOrder.DescendingOrder if self._cluster_sort_desc else Qt.SortOrder.AscendingOrder
        self.table.horizontalHeader().setSortIndicator(column, order)
        if self._last_assignments:
            self._populate_assignments_table(self._last_assignments)

    def _populate_assignments_table(self, assignments) -> None:
        field = self._cluster_sort_field
        reverse = self._cluster_sort_desc

        def sort_key(a):
            if field in ("produto", "descricao", "cluster_name"):
                return str(getattr(a, field, "")).lower()
            if field == "cutoff_metric_value":
                return a.cutoff_metric_value
            if field == "non_zero_hour_ratio":
                return a.non_zero_hour_ratio
            if field == "p95":
                return a.p95
            if field == "p99":
                return a.p99
            if field == "cv":
                return a.cv
            if field == "turnover_score":
                return a.turnover_score
            if field == "distance_to_centroid":
                return a.distance_to_centroid
            return 0.0

        sorted_assignments = sorted(assignments, key=sort_key, reverse=reverse)
        self.table.setRowCount(len(sorted_assignments))
        for i, a in enumerate(sorted_assignments):
            vals = [a.produto, a.descricao, a.cluster_name,
                    f"{a.cutoff_metric_value:.3f}", f"{a.non_zero_hour_ratio:.3f}",
                    f"{a.p95:.1f}", f"{a.p99:.1f}", f"{a.cv:.3f}",
                    f"{a.turnover_score:.3f}", f"{a.distance_to_centroid:.3f}"]
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QTableWidgetItem(v))

    def populate_light(self) -> None:
        self._sync_spinboxes_from_state()
        self._update_preview()
        self.refresh()

    def _sync_spinboxes_from_state(self) -> None:
        self.alto_spin.blockSignals(True)
        self.baixo_spin.blockSignals(True)
        self.alto_spin.setValue(self.state.cluster_alto_cutoff)
        self.baixo_spin.setValue(self.state.cluster_baixo_cutoff)
        self.alto_spin.blockSignals(False)
        self.baixo_spin.blockSignals(False)
        idx = self.metric_combo.findData(self.state.cluster_cutoff_field)
        if idx >= 0:
            self.metric_combo.blockSignals(True)
            self.metric_combo.setCurrentIndex(idx)
            self.metric_combo.blockSignals(False)

    def _current_field(self) -> str:
        field = self.metric_combo.currentData()
        return str(field) if field else "mean_hourly_demand"

    def _ensure_valid_cutoffs(self) -> bool:
        alto = self.alto_spin.value()
        baixo = self.baixo_spin.value()
        if baixo >= alto:
            self.baixo_spin.blockSignals(True)
            new_baixo = max(0.0, alto * 0.5)
            if new_baixo >= alto:
                new_baixo = max(0.0, alto - 0.01)
            self.baixo_spin.setValue(new_baixo)
            self.baixo_spin.blockSignals(False)
        return self.baixo_spin.value() < self.alto_spin.value()

    def _on_cutoff_edited(self) -> None:
        if not self._ensure_valid_cutoffs():
            return
        self._update_preview()
        self._recalc_timer.start()

    def _on_metric_changed(self) -> None:
        self.state.cluster_cutoff_field = self._current_field()
        self._apply_suggested_cutoffs(recalculate=False)
        self._update_preview()
        self._recalc_timer.start()

    def _apply_suggested_cutoffs(self, recalculate: bool = True) -> None:
        df = self.state.feature_df
        if df is None or df.empty:
            return
        field = self._current_field()
        baixo, alto = suggest_threshold_cutoffs(df, field=field)
        self.alto_spin.blockSignals(True)
        self.baixo_spin.blockSignals(True)
        self.alto_spin.setValue(alto)
        self.baixo_spin.setValue(baixo)
        self.alto_spin.blockSignals(False)
        self.baixo_spin.blockSignals(False)
        self._update_preview()
        if recalculate:
            self._recalculate()

    def _update_preview(self) -> None:
        df = self.state.feature_df
        if df is None or df.empty:
            self.preview_label.setText("Prévia: carregue os dados primeiro.")
            return
        if not self._ensure_valid_cutoffs():
            self.preview_label.setText("Prévia: ajuste Baixo < Alto para continuar.")
            return
        alto = self.alto_spin.value()
        baixo = self.baixo_spin.value()
        field = self._current_field()
        counts = preview_cluster_counts(df, alto, baixo, field)
        total = sum(counts.values()) or 1
        metric_label = CUTOFF_FIELDS.get(field, field)
        self.preview_label.setText(
            f"Prévia ({metric_label}) — "
            f"Alto ≥ {alto:.3f}: {counts['Alto Giro']} SKUs ({counts['Alto Giro']/total*100:.1f}%) | "
            f"Médio: {counts['Medio Giro']} | "
            f"Baixo < {baixo:.3f}: {counts['Baixo Giro']}"
        )

    def _recalculate(self) -> None:
        if not self._ensure_valid_cutoffs():
            self.controller.show_error(
                "Corte Baixo Giro deve ser menor que Corte Alto Giro."
            )
            return
        alto = self.alto_spin.value()
        baixo = self.baixo_spin.value()
        field = self._current_field()

        self.btn_recalc.setEnabled(False)
        self.btn_recalc.setText("Calculando...")
        try:
            self.controller.run_clustering(
                alto_cutoff=alto,
                baixo_cutoff=baixo,
                cutoff_field=field,
                on_done=self._on_cluster_done,
                on_error=self._on_cluster_error,
            )
        except Exception as exc:
            self._on_cluster_error(str(exc))

    def _on_cluster_error(self, message: str) -> None:
        self.btn_recalc.setEnabled(True)
        self.btn_recalc.setText("Recalcular Clusters")
        self.controller.show_error(message)

    def _on_cluster_done(self, result) -> None:
        self.btn_recalc.setEnabled(True)
        self.btn_recalc.setText("Recalcular Clusters")
        self.refresh()

    def _select_cluster(self, name: str) -> None:
        self.state.selected_cluster = name
        self.state.cluster_filter = name
        self.state.cluster_changed.emit(name)

    def _study_products(self) -> None:
        win = self.window()
        if hasattr(win, "navigate_to"):
            win.navigate_to(STEP_PRODUCTS)

    def refresh(self) -> None:
        cr = self.state.cluster_result
        if cr is None:
            self.header_label.setText(
                "Dados carregados. Ajuste os cortes — o recálculo é automático."
            )
            self._update_preview()
            return

        if cr.method == "threshold":
            field_label = CUTOFF_FIELDS.get(cr.cutoff_field, cr.cutoff_field)
            self.header_label.setText(
                f"SKUs: {len(cr.assignments)} | {field_label} | "
                f"Alto ≥ {cr.alto_cutoff:.3f} | Baixo < {cr.baixo_cutoff:.3f} | "
                f"Silhouette: {cr.silhouette:.3f}"
            )
        else:
            self.header_label.setText(
                f"SKUs: {len(cr.assignments)} | Features: {', '.join(cr.features_used)} | "
                f"Scaler: {cr.scaler_name} | K={cr.n_clusters} | "
                f"Silhouette: {cr.silhouette:.3f} | Inertia: {cr.inertia:.1f}"
            )

        cards = {"Baixo Giro": self.card_baixo, "Medio Giro": self.card_medio, "Alto Giro": self.card_alto}
        for name, card in cards.items():
            s = cr.cluster_summaries.get(name, {})
            card.set_stats(
                int(s.get("n_skus", 0)), s.get("pct", 0),
                s.get("mean_demand", 0), s.get("mean_frequency", 0), s.get("mean_p95", 0),
            )

        plot_cluster_scatter(
            self.scatter.canvas, cr.assignments, cr, self.state.selected_sku,
            x_feat=cr.cutoff_field,
        )

        self._last_assignments = list(cr.assignments)
        self._populate_assignments_table(self._last_assignments)

        self._update_preview()
