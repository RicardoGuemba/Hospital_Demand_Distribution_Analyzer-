"""Distributions view."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hospital_demand_analyzer.app.controller import AppController
from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.plots.canvas import PlotWidget
from hospital_demand_analyzer.plots.ecdf_plot import plot_ecdf
from hospital_demand_analyzer.plots.histogram_plot import plot_histogram
from hospital_demand_analyzer.domain.demand_values import prepare_demand_values, sanitize_demand
from hospital_demand_analyzer.statistics.fitter import select_fits_for_display
from hospital_demand_analyzer.ui.nav import STEP_DISTRIBUTIONS
from hospital_demand_analyzer.ui.widgets.active_view import ActiveViewMixin
from hospital_demand_analyzer.ui.widgets.value_mode_combo import ValueModeCombo
from hospital_demand_analyzer.ui.widgets.workflow_bar import WorkflowBar

WARNING_DEFAULT = (
    "Melhor distribuição significa melhor modelo entre as candidatas avaliadas, "
    "não a 'verdadeira distribuição' da demanda."
)


class DistributionsView(ActiveViewMixin, QWidget):
    def __init__(self, state: AppState, controller: AppController, parent=None) -> None:
        super().__init__(parent)
        self.__init_active_view__()
        self.state = state
        self.controller = controller
        self._pending_cache_key: str | None = None
        self._build_ui()
        state.sku_changed.connect(self.refresh_if_active)
        state.cluster_changed.connect(lambda _: self.refresh_if_active())
        state.fit_completed.connect(self._on_fit_completed)
        state.value_mode_changed.connect(lambda _: self.refresh_if_active())

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(WorkflowBar(STEP_DISTRIBUTIONS))

        controls = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["SKU", "Cluster Agregado"])
        self.mode_combo.currentTextChanged.connect(lambda _: self.refresh_if_active())
        self.consider_zeros = QCheckBox("Incluir zeros (demanda ≥ 0)")
        self.consider_zeros.setChecked(True)
        self.consider_zeros.setToolTip(
            "Marcado (padrão): usa toda a grade horária com demanda ≥ 0.\n"
            "Desmarcado: analisa apenas horas com demanda > 0."
        )
        self.consider_zeros.stateChanged.connect(lambda _: self.refresh_if_active())
        self.value_mode_combo = ValueModeCombo(self.state)
        self.value_mode_combo.currentIndexChanged.connect(lambda _: self.refresh_if_active())
        self.series_combo = QComboBox()
        self.series_combo.addItems(["Original", "Tratada"])
        self.series_combo.currentTextChanged.connect(lambda _: self.refresh_if_active())
        controls.addWidget(QLabel("Modo:"))
        controls.addWidget(self.mode_combo)
        controls.addWidget(self.consider_zeros)
        controls.addWidget(QLabel("Representação:"))
        controls.addWidget(self.value_mode_combo)
        controls.addWidget(QLabel("Série:"))
        controls.addWidget(self.series_combo)
        controls.addStretch()
        layout.addLayout(controls)

        self.warning = QLabel(WARNING_DEFAULT)
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #856404; background: #fff3cd; padding: 4px;")
        layout.addWidget(self.warning)

        splitter = QSplitter()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.hist_plot = PlotWidget(6, 3)
        self.ecdf_plot = PlotWidget(6, 3)
        left_layout.addWidget(self.hist_plot)
        left_layout.addWidget(self.ecdf_plot)
        splitter.addWidget(left)

        self.rank_table = QTableWidget(0, 11)
        self.rank_table.setHorizontalHeaderLabels([
            "Rank", "Distribuição", "Tipo", "AIC", "ΔAIC", "BIC", "ΔBIC",
            "KS", "CvM", "AD", "Tail Score",
        ])
        splitter.addWidget(self.rank_table)
        splitter.setSizes([600, 400])
        layout.addWidget(splitter)

    def _on_fit_completed(self, cache_key: str) -> None:
        if not getattr(self, "_view_active", False):
            return
        if self._pending_cache_key and cache_key != self._pending_cache_key:
            return
        self.refresh_if_active()

    def refresh(self) -> None:
        include_zeros = self.consider_zeros.isChecked()
        positive_only = not include_zeros
        value_mode = self.state.demand_value_mode
        use_treated = self.series_combo.currentText() == "Tratada"
        agg_mode = self.mode_combo.currentText()
        suffix = "" if include_zeros else " (apenas demanda > 0)"

        if agg_mode == "Cluster Agregado" and self.state.selected_cluster not in ("Todos", ""):
            values_full = self._aggregate_cluster_values(use_treated)
            cache_key = "__cluster__"
        else:
            sku = self.state.selected_sku
            if not sku:
                self.warning.setText("Selecione um produto na etapa 4 ou 5.")
                return
            hs = (self.state.treated_series.get(sku) if use_treated
                  else self.state.hourly_series.get(sku))
            if hs is None:
                return
            values_full = hs.quantities.values.astype(float)
            cache_key = sku

        values_valid = sanitize_demand(values_full)
        if len(values_valid) == 0:
            self.warning.setText("Nenhum dado com demanda ≥ 0 disponível para ajuste.")
            return

        values_plot = prepare_demand_values(values_valid, include_zeros=include_zeros)
        if len(values_plot) == 0:
            self.warning.setText("Nenhuma hora com demanda > 0 para o modo sem zeros.")
            return

        zero_ratio_plot = (
            float((values_valid == 0).sum()) / len(values_valid) if include_zeros else 0.0
        )

        self._pending_cache_key = cache_key
        results = self.controller.request_fit_values(
            cache_key,
            values_plot,
            positive_only=positive_only,
            value_mode=value_mode,
        )

        if results is None:
            self.warning.setText(
                f"Calculando ajustes em segundo plano ({len(values_plot):,} observações)..."
            )
            plot_histogram(
                self.hist_plot.canvas, values_plot, "auto", [],
                title="Histograma + Distribuições" + suffix,
                positive_only=positive_only,
                value_mode=value_mode,
            )
            plot_ecdf(
                self.ecdf_plot.canvas, values_plot, [],
                title="ECDF + CDFs" + suffix,
                positive_only=positive_only,
            )
            self.rank_table.setRowCount(0)
            return

        self.warning.setText(WARNING_DEFAULT)
        self._render_results(
            results, values_plot, zero_ratio_plot, suffix, positive_only, value_mode,
        )

    def _render_results(
        self,
        results: list,
        values_plot: np.ndarray,
        zero_ratio_plot: float,
        suffix: str,
        positive_only: bool,
        value_mode: str,
    ) -> None:
        display_fits = select_fits_for_display(
            results or [], zero_ratio_plot, value_mode=value_mode,
        )
        plot_histogram(
            self.hist_plot.canvas, values_plot, "auto", display_fits,
            title="Histograma + Distribuições" + suffix,
            positive_only=positive_only,
            value_mode=value_mode,
        )
        plot_ecdf(
            self.ecdf_plot.canvas, values_plot, display_fits,
            title="ECDF + CDFs" + suffix,
            positive_only=positive_only,
        )

        converged = [r for r in results if r.converged]
        self.rank_table.setRowCount(len(converged))
        for i, r in enumerate(sorted(converged, key=lambda x: x.aic)):
            cols = [
                str(r.rank or i + 1), r.name, r.dist_type,
                f"{r.aic:.1f}", f"{r.delta_aic:.1f}",
                f"{r.bic:.1f}", f"{r.delta_bic:.1f}",
                f"{r.ks_stat:.4f}" if r.ks_stat == r.ks_stat else "—",
                f"{r.cvm_stat:.4f}" if r.cvm_stat == r.cvm_stat else "—",
                f"{r.ad_stat:.4f}" if r.ad_stat == r.ad_stat else "—",
                f"{r.tail_fit_score:.4f}" if r.tail_fit_score == r.tail_fit_score else "—",
            ]
            for j, c in enumerate(cols):
                self.rank_table.setItem(i, j, QTableWidgetItem(c))

    def _aggregate_cluster_values(self, use_treated: bool) -> np.ndarray:
        cluster = self.state.selected_cluster
        parts = []
        for produto, a in self.state.cluster_map.items():
            if a.cluster_name != cluster:
                continue
            hs = (self.state.treated_series.get(produto) if use_treated
                  else self.state.hourly_series.get(produto))
            if hs:
                parts.append(hs.quantities.values)
        if not parts:
            return np.array([])
        return np.concatenate(parts)
