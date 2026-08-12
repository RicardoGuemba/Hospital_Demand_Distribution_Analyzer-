"""Original vs Treated comparison view."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
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
from hospital_demand_analyzer.statistics.descriptive import compute_descriptive
from hospital_demand_analyzer.statistics.distribution_moments import theoretical_expected_value
from hospital_demand_analyzer.statistics.fitter import fit_all, top_n
from hospital_demand_analyzer.ui.nav import STEP_COMPARISON
from hospital_demand_analyzer.ui.widgets.active_view import ActiveViewMixin
from hospital_demand_analyzer.ui.widgets.workflow_bar import WorkflowBar


class ComparisonView(ActiveViewMixin, QWidget):
    def __init__(self, state: AppState, controller: AppController, parent=None) -> None:
        super().__init__(parent)
        self.__init_active_view__()
        self.state = state
        self.controller = controller
        self._build_ui()
        state.sku_changed.connect(self.refresh_if_active)
        state.outliers_changed.connect(self.refresh_if_active)
        state.value_mode_changed.connect(self.refresh_if_active)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(WorkflowBar(STEP_COMPARISON))
        self.info = QLabel("Selecione um SKU com outliers aplicados.")
        layout.addWidget(self.info)

        splitter = QSplitter()
        plots = QWidget()
        plots_layout = QVBoxLayout(plots)
        row1 = QHBoxLayout()
        self.hist_orig = PlotWidget(5, 2.5)
        self.hist_treat = PlotWidget(5, 2.5)
        row1.addWidget(self.hist_orig)
        row1.addWidget(self.hist_treat)
        plots_layout.addLayout(row1)
        row2 = QHBoxLayout()
        self.ecdf_orig = PlotWidget(5, 2.5)
        self.ecdf_treat = PlotWidget(5, 2.5)
        row2.addWidget(self.ecdf_orig)
        row2.addWidget(self.ecdf_treat)
        plots_layout.addLayout(row2)
        splitter.addWidget(plots)

        self.metrics_table = QTableWidget(0, 4)
        self.metrics_table.setHorizontalHeaderLabels(["Métrica", "Original", "Tratado", "Diferença"])
        splitter.addWidget(self.metrics_table)
        layout.addWidget(splitter)

    def refresh(self) -> None:
        sku = self.state.selected_sku
        if not sku:
            return
        orig = self.state.hourly_series.get(sku)
        treat = self.state.treated_series.get(sku)
        if orig is None:
            return
        if treat is None:
            self.info.setText(f"SKU {sku}: nenhum tratamento de outlier aplicado.")
            return

        self.info.setText(f"Comparação Original × Tratado — SKU {sku}")

        orig_vals = orig.quantities.values
        treat_vals = treat.quantities.values
        ds_o = compute_descriptive(orig)
        ds_t = compute_descriptive(treat)
        value_mode = self.state.demand_value_mode
        fit_o = top_n(fit_all(orig_vals, value_mode=value_mode), 1)
        fit_t = top_n(fit_all(treat_vals, value_mode=value_mode), 1)

        plot_histogram(
            self.hist_orig.canvas, orig_vals, title="Histograma Original", value_mode=value_mode,
        )
        plot_histogram(
            self.hist_treat.canvas, treat_vals, title="Histograma Tratado", value_mode=value_mode,
        )
        plot_ecdf(self.ecdf_orig.canvas, orig_vals, fit_o, title="ECDF Original")
        plot_ecdf(self.ecdf_treat.canvas, treat_vals, fit_t, title="ECDF Tratado")

        best_o = fit_o[0].name if fit_o else "—"
        best_t = fit_t[0].name if fit_t else "—"
        theo_o = theoretical_expected_value(fit_o[0]) if fit_o else None
        theo_t = theoretical_expected_value(fit_t[0]) if fit_t else None
        rows = [
            ("N", ds_o.n, ds_t.n, ds_t.n - ds_o.n),
            ("Média aritmética", ds_o.mean, ds_t.mean, ds_t.mean - ds_o.mean),
            ("E[X] empírico", ds_o.expected_empirical, ds_t.expected_empirical,
             ds_t.expected_empirical - ds_o.expected_empirical),
            ("E[X|X>0]", ds_o.expected_conditional, ds_t.expected_conditional,
             ds_t.expected_conditional - ds_o.expected_conditional),
            ("E[X] teórico (ajuste)", theo_o, theo_t,
             (theo_t - theo_o) if theo_o is not None and theo_t is not None else "—"),
            ("Mediana", ds_o.median, ds_t.median, ds_t.median - ds_o.median),
            ("Máximo", ds_o.max_, ds_t.max_, ds_t.max_ - ds_o.max_),
            ("P95", ds_o.p95, ds_t.p95, ds_t.p95 - ds_o.p95),
            ("P99", ds_o.p99, ds_t.p99, ds_t.p99 - ds_o.p99),
            ("Melhor Distribuição", best_o, best_t, "—"),
            ("AIC", fit_o[0].aic if fit_o else 0, fit_t[0].aic if fit_t else 0,
             (fit_t[0].aic - fit_o[0].aic) if fit_o and fit_t else 0),
            ("BIC", fit_o[0].bic if fit_o else 0, fit_t[0].bic if fit_t else 0,
             (fit_t[0].bic - fit_o[0].bic) if fit_o and fit_t else 0),
        ]
        self.metrics_table.setRowCount(len(rows))
        for i, (metric, o, t, d) in enumerate(rows):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(
                f"{o:.2f}" if isinstance(o, float) else str(o)))
            self.metrics_table.setItem(i, 2, QTableWidgetItem(
                f"{t:.2f}" if isinstance(t, float) else str(t)))
            self.metrics_table.setItem(i, 3, QTableWidgetItem(
                f"{d:.2f}" if isinstance(d, float) else str(d)))
