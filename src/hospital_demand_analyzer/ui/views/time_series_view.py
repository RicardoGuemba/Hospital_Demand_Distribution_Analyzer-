"""Time series view with histogram and FDC."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from hospital_demand_analyzer.app.controller import AppController
from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.domain.demand_values import prepare_demand_values
from hospital_demand_analyzer.plots.canvas import PlotWidget
from hospital_demand_analyzer.plots.ecdf_plot import plot_fdc_adjusted, plot_fdc_empirical
from hospital_demand_analyzer.plots.histogram_plot import plot_histogram
from hospital_demand_analyzer.plots.time_series_plot import plot_time_series
from hospital_demand_analyzer.statistics.descriptive import METRIC_TOOLTIPS, compute_descriptive
from hospital_demand_analyzer.statistics.distribution_moments import theoretical_expected_value
from hospital_demand_analyzer.statistics.fitter import select_fits_for_display, top_n
from hospital_demand_analyzer.ui.nav import STEP_TIME_SERIES
from hospital_demand_analyzer.ui.widgets.active_view import ActiveViewMixin
from hospital_demand_analyzer.ui.widgets.value_mode_combo import ValueModeCombo
from hospital_demand_analyzer.ui.widgets.workflow_bar import WorkflowBar


class TimeSeriesView(ActiveViewMixin, QWidget):
    def __init__(self, state: AppState, controller: AppController, parent=None) -> None:
        super().__init__(parent)
        self.__init_active_view__()
        self.state = state
        self.controller = controller
        self._build_ui()
        state.sku_changed.connect(self._sync_product_combo)
        state.sku_changed.connect(self.refresh_if_active)
        state.outliers_changed.connect(self.refresh_if_active)
        state.fit_completed.connect(self.refresh_if_active)
        state.value_mode_changed.connect(self.refresh_if_active)
        state.cluster_changed.connect(self._populate_products)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(WorkflowBar(STEP_TIME_SERIES))

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Produto:"))
        self.product_combo = QComboBox()
        self.product_combo.setEditable(True)
        self.product_combo.setMinimumWidth(320)
        self.product_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.product_combo.currentIndexChanged.connect(self._on_product_selected)
        selector_row.addWidget(self.product_combo, 1)

        self.series_combo = QComboBox()
        self.series_combo.addItems(["Original", "Tratada"])
        self.series_combo.currentTextChanged.connect(self.refresh)
        selector_row.addWidget(QLabel("Série:"))
        selector_row.addWidget(self.series_combo)

        self.bin_combo = QComboBox()
        self.bin_combo.addItems(["auto", "fd", "sturges"])
        self.bin_combo.currentTextChanged.connect(self.refresh)
        selector_row.addWidget(QLabel("Bins:"))
        selector_row.addWidget(self.bin_combo)

        self.hist_exclude_zeros = QCheckBox("Incluir zeros (demanda ≥ 0)")
        self.hist_exclude_zeros.setChecked(True)
        self.hist_exclude_zeros.setToolTip(
            "Marcado (padrão): histograma/FDC com demanda ≥ 0 (inclui horas zero).\n"
            "Desmarcado: reprocessa usando apenas horas com demanda > 0."
        )
        self.hist_exclude_zeros.stateChanged.connect(self._on_zero_mode_changed)
        selector_row.addWidget(self.hist_exclude_zeros)

        selector_row.addWidget(QLabel("Representação:"))
        self.value_mode_combo = ValueModeCombo(self.state)
        self.value_mode_combo.currentIndexChanged.connect(self.refresh)
        selector_row.addWidget(self.value_mode_combo)
        layout.addLayout(selector_row)

        self.info_label = QLabel("Selecione um produto para analisar.")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.ts_plot = PlotWidget(10, 3)
        layout.addWidget(self.ts_plot)

        analysis_splitter = QSplitter()
        self.hist_plot = PlotWidget(5, 3.5)
        self.fdc_emp_plot = PlotWidget(5, 3.5)
        self.fdc_adj_plot = PlotWidget(5, 3.5)
        analysis_splitter.addWidget(self._wrap_plot("Histograma", self.hist_plot))
        analysis_splitter.addWidget(self._wrap_plot("FDC Empírica", self.fdc_emp_plot))
        analysis_splitter.addWidget(self._wrap_plot("FDC Ajustada", self.fdc_adj_plot))
        analysis_splitter.setSizes([400, 400, 400])
        layout.addWidget(analysis_splitter)

        self.stats_group = QGroupBox("Estatísticas (horas nulas = 0)")
        self.stats_layout = QGridLayout(self.stats_group)
        layout.addWidget(self.stats_group)

    def _wrap_plot(self, title: str, plot: PlotWidget) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight: bold;")
        vl.addWidget(lbl)
        vl.addWidget(plot)
        return w

    def _populate_products(self) -> None:
        self.product_combo.blockSignals(True)
        self.product_combo.clear()
        skus = sorted(self.state.filtered_skus() or self.state.hourly_series.keys())
        for sku in skus:
            hs = self.state.hourly_series.get(sku)
            desc = hs.descricao[:40] if hs else ""
            self.product_combo.addItem(f"{sku} — {desc}", sku)
        completer = QCompleter([self.product_combo.itemText(i) for i in range(self.product_combo.count())])
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.product_combo.setCompleter(completer)
        self.product_combo.blockSignals(False)
        self._sync_product_combo(self.state.selected_sku)

    def populate_light(self) -> None:
        self._populate_products()

    def _sync_product_combo(self, sku: str) -> None:
        if not sku:
            return
        idx = self.product_combo.findData(sku)
        if idx >= 0:
            self.product_combo.blockSignals(True)
            self.product_combo.setCurrentIndex(idx)
            self.product_combo.blockSignals(False)

    def _on_zero_mode_changed(self) -> None:
        self.refresh()

    def _on_product_selected(self, index: int) -> None:
        if index < 0:
            return
        sku = self.product_combo.itemData(index)
        if sku and sku != self.state.selected_sku:
            self.controller.select_sku(str(sku))

    def refresh(self) -> None:
        sku = self.state.selected_sku
        if not sku:
            return
        use_treated = self.series_combo.currentText() == "Tratada"
        hs = self.state.treated_series.get(sku) if use_treated else self.state.hourly_series.get(sku)
        if hs is None:
            self.info_label.setText(f"Produto {sku} não encontrado.")
            return

        ds = compute_descriptive(hs)
        first_raw = hs.first_record_at.strftime("%Y-%m-%d %H:%M") if hs.first_record_at else "—"
        last_raw = hs.last_record_at.strftime("%Y-%m-%d %H:%M") if hs.last_record_at else "—"
        period_start = ds.date_start.strftime("%Y-%m-%d %H:%M") if ds.date_start else "—"
        period_end = ds.date_end.strftime("%Y-%m-%d %H:%M") if ds.date_end else "—"
        global_h = self.state.global_n_hours or ds.n

        self.info_label.setText(
            f"Produto: {sku} | {hs.descricao}\n"
            f"Primeiro registro SKU: {first_raw} | Último registro SKU: {last_raw}\n"
            f"Grade global: {period_start} → {period_end} | {global_h} horas (1h) — "
            f"zeros neste SKU: {ds.n_zeros} ({ds.zero_ratio*100:.1f}%) | "
            f"Demanda total SKU: {ds.sum_:.0f}"
        )

        outlier = self.state.outlier_results.get(sku)
        mask = outlier.mask if outlier else None
        plot_time_series(
            self.ts_plot.canvas, hs, outlier_mask=mask,
            title=f"Série Temporal Normalizada — {sku}",
        )

        values = hs.quantities.values.astype(float)
        include_zeros = self.hist_exclude_zeros.isChecked()
        positive_only = not include_zeros
        value_mode = self.state.demand_value_mode
        values_plot = prepare_demand_values(values, include_zeros=include_zeros)
        suffix = "" if include_zeros else " (apenas demanda > 0)"

        if len(values_plot) == 0:
            self.info_label.setText(f"Produto {sku}: nenhuma hora com demanda > 0.")
            return

        fits = self.controller.request_fit(sku, positive_only=positive_only, value_mode=value_mode)
        if fits is None:
            plot_histogram(
                self.hist_plot.canvas, values_plot, self.bin_combo.currentText(), [],
                title=f"Histograma — {sku}{suffix}",
                positive_only=positive_only,
                value_mode=value_mode,
            )
            plot_fdc_empirical(
                self.fdc_emp_plot.canvas, values_plot,
                title=f"FDC Empírica — {sku}{suffix}",
            )
            if self.controller.is_fitting(sku, positive_only, value_mode):
                self.info_label.setText(
                    f"{self.info_label.text()}\nCalculando ajustes de distribuição..."
                )
            return

        zero_ratio_plot = ds.zero_ratio if include_zeros else 0.0
        display_fits = select_fits_for_display(
            fits or [], zero_ratio_plot, value_mode=value_mode,
        )

        plot_histogram(
            self.hist_plot.canvas, values_plot, self.bin_combo.currentText(), display_fits,
            title=f"Histograma — {sku}{suffix}",
            positive_only=positive_only,
            value_mode=value_mode,
        )
        plot_fdc_empirical(
            self.fdc_emp_plot.canvas, values_plot,
            title=f"FDC Empírica — {sku}{suffix}",
        )
        plot_fdc_adjusted(
            self.fdc_adj_plot.canvas, values_plot, display_fits,
            title=f"FDC Ajustada — {sku}{suffix}",
            positive_only=positive_only,
        )

        best_fit = display_fits[0] if display_fits else (top_n(fits, 1)[0] if fits else None)
        theo = theoretical_expected_value(best_fit) if best_fit else None
        theo_text = f"{theo:.3f}" if theo is not None else "—"

        stats = [
            ("N horas", str(ds.n), None),
            ("Horas zero", str(ds.n_zeros), None),
            ("% Zero", f"{ds.zero_ratio*100:.1f}%", None),
            ("Média aritmética", f"{ds.mean:.3f}", "mean"),
            ("E[X] empírico", f"{ds.expected_empirical:.3f}", "expected_empirical"),
            ("E[X|X>0]", f"{ds.expected_conditional:.3f}", "expected_conditional"),
            ("E[X] teórico (ajuste)", theo_text, "expected_theoretical"),
            ("Mediana", f"{ds.median:.3f}", None),
            ("Desvio Padrão", f"{ds.std:.3f}", None),
            ("Mínimo", f"{ds.min_:.1f}", None),
            ("Máximo", f"{ds.max_:.1f}", None),
            ("P90", f"{ds.p90:.1f}", None),
            ("P95", f"{ds.p95:.1f}", None),
            ("P97.5", f"{ds.p975:.1f}", None),
            ("P99", f"{ds.p99:.1f}", None),
        ]
        while self.stats_layout.count():
            item = self.stats_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, (name, val, tip_key) in enumerate(stats):
            name_lbl = QLabel(f"{name}:")
            val_lbl = QLabel(val)
            if tip_key and tip_key in METRIC_TOOLTIPS:
                name_lbl.setToolTip(METRIC_TOOLTIPS[tip_key])
                val_lbl.setToolTip(METRIC_TOOLTIPS[tip_key])
            self.stats_layout.addWidget(name_lbl, i // 4, (i % 4) * 2)
            self.stats_layout.addWidget(val_lbl, i // 4, (i % 4) * 2 + 1)
