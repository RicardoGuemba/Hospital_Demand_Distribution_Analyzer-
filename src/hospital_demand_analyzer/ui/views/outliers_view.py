"""Outliers view."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hospital_demand_analyzer.app.controller import AppController
from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.plots.canvas import PlotWidget
from hospital_demand_analyzer.plots.time_series_plot import plot_time_series
from hospital_demand_analyzer.statistics.descriptive import compute_descriptive
from hospital_demand_analyzer.ui.nav import STEP_OUTLIERS
from hospital_demand_analyzer.ui.widgets.active_view import ActiveViewMixin
from hospital_demand_analyzer.ui.widgets.workflow_bar import WorkflowBar


class OutliersView(ActiveViewMixin, QWidget):
    def __init__(self, state: AppState, controller: AppController, parent=None) -> None:
        super().__init__(parent)
        self.__init_active_view__()
        self.state = state
        self.controller = controller
        self._build_ui()
        state.sku_changed.connect(self.refresh_if_active)
        state.outliers_changed.connect(self.refresh_if_active)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(WorkflowBar(STEP_OUTLIERS))

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Produto:"))
        self.product_combo = QComboBox()
        self.product_combo.setMinimumWidth(320)
        self.product_combo.currentIndexChanged.connect(self._on_product_selected)
        selector_row.addWidget(self.product_combo, 1)
        layout.addLayout(selector_row)

        header = QHBoxLayout()
        self.sku_label = QLabel("SKU: —")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["IQR", "Z-Score", "MAD", "Hampel"])
        self.method_combo.setToolTip(
            "IQR: Tukey 1,5×IQR em horas com demanda > 0 (padrão).\n"
            "Z-Score: |z| > 3 na demanda positiva.\n"
            "MAD: Z-score modificado (limiar 3,5).\n"
            "Hampel: filtro robusto local na série temporal."
        )
        self.method_combo.currentTextChanged.connect(self._update_params)

        self.param_widget = QWidget()
        self.param_layout = QHBoxLayout(self.param_widget)
        self.iqr_spin = QDoubleSpinBox()
        self.iqr_spin.setRange(0.5, 5.0)
        self.iqr_spin.setValue(1.5)
        self.iqr_spin.setSingleStep(0.1)
        self.iqr_spin.setToolTip("Multiplicador k na regra de Tukey: Q1 − k·IQR, Q3 + k·IQR")
        self.zscore_spin = QDoubleSpinBox()
        self.zscore_spin.setRange(2.0, 6.0)
        self.zscore_spin.setValue(3.0)
        self.zscore_spin.setSingleStep(0.1)
        self.mad_spin = QDoubleSpinBox()
        self.mad_spin.setRange(1.0, 10.0)
        self.mad_spin.setValue(3.5)
        self.hampel_window = QSpinBox()
        self.hampel_window.setRange(3, 21)
        self.hampel_window.setValue(5)
        self.hampel_sigma = QDoubleSpinBox()
        self.hampel_sigma.setRange(1.0, 10.0)
        self.hampel_sigma.setValue(3.0)

        btn_apply = QPushButton("Aplicar")
        btn_apply.clicked.connect(self._apply)
        btn_restore = QPushButton("Restaurar")
        btn_restore.clicked.connect(self._restore)
        header.addWidget(self.sku_label)
        header.addWidget(QLabel("Método:"))
        header.addWidget(self.method_combo)
        header.addWidget(self.param_widget)
        header.addWidget(btn_apply)
        header.addWidget(btn_restore)
        layout.addLayout(header)

        self.method_hint = QLabel(
            "IQR (Tukey): quartis calculados só em horas com demanda > 0; "
            "horas zero não são marcadas como outlier."
        )
        self.method_hint.setWordWrap(True)
        self.method_hint.setStyleSheet("color: #555; font-size: 11px;")
        layout.addWidget(self.method_hint)
        self._update_params()

        cards = QGridLayout()
        self.card_obs = self._card("Observações")
        self.card_outliers = self._card("Outliers")
        self.card_pct = self._card("Percentual")
        self.card_max_orig = self._card("Máximo Original")
        self.card_max_treat = self._card("Máximo Tratado")
        for i, c in enumerate([self.card_obs, self.card_outliers, self.card_pct,
                                self.card_max_orig, self.card_max_treat]):
            cards.addWidget(c[0], 0, i)
        layout.addLayout(cards)

        self.plot = PlotWidget(10, 3)
        layout.addWidget(self.plot)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Timestamp", "Quantidade", "Método", "Score", "Motivo", "Outlier",
        ])
        layout.addWidget(self.table)

    def populate_light(self) -> None:
        self._populate_products()

    def _populate_products(self) -> None:
        self.product_combo.blockSignals(True)
        self.product_combo.clear()
        for sku in sorted(self.state.hourly_series.keys()):
            hs = self.state.hourly_series[sku]
            desc = hs.descricao[:40] if hs.descricao else ""
            self.product_combo.addItem(f"{sku} — {desc}", sku)
        self.product_combo.blockSignals(False)
        if self.state.selected_sku:
            idx = self.product_combo.findData(self.state.selected_sku)
            if idx >= 0:
                self.product_combo.setCurrentIndex(idx)
        elif self.product_combo.count() > 0 and not self.state.selected_sku:
            self.product_combo.blockSignals(True)
            self.product_combo.setCurrentIndex(0)
            self.product_combo.blockSignals(False)

    def _on_product_selected(self, index: int) -> None:
        if index < 0:
            return
        sku = self.product_combo.itemData(index)
        if sku and sku != self.state.selected_sku:
            self.controller.select_sku(str(sku))

    def _card(self, title: str) -> tuple:
        g = QGroupBox(title)
        l = QVBoxLayout(g)
        lbl = QLabel("—")
        l.addWidget(lbl)
        return g, lbl

    def _update_params(self) -> None:
        while self.param_layout.count():
            self.param_layout.takeAt(0)
        method = self.method_combo.currentText()
        if method == "IQR":
            self.param_layout.addWidget(QLabel("k (IQR):"))
            self.param_layout.addWidget(self.iqr_spin)
            self.method_hint.setText(
                "IQR (Tukey): Q1 e Q3 em horas com demanda > 0; "
                f"limites Q1 − {self.iqr_spin.value():.1f}×IQR e Q3 + {self.iqr_spin.value():.1f}×IQR. "
                "Horas zero não são outliers."
            )
        elif method == "Z-Score":
            self.param_layout.addWidget(QLabel("|z| >"))
            self.param_layout.addWidget(self.zscore_spin)
            self.method_hint.setText(
                "Z-Score clássico: média e desvio calculados em demanda > 0; "
                "marca horas com |z| acima do limiar."
            )
        elif method == "MAD":
            self.param_layout.addWidget(QLabel("Z mod. >"))
            self.param_layout.addWidget(self.mad_spin)
            self.method_hint.setText(
                "MAD (Z-score modificado): mediana e MAD em demanda > 0; "
                "robusto a assimetria (limiar típico 3,5)."
            )
        else:
            self.param_layout.addWidget(QLabel("Janela:"))
            self.param_layout.addWidget(self.hampel_window)
            self.param_layout.addWidget(QLabel("Sigmas:"))
            self.param_layout.addWidget(self.hampel_sigma)
            self.method_hint.setText(
                "Hampel: mediana/MAD em janela móvel; detecta picos locais na série."
            )

    def _get_params(self) -> dict:
        method = self.method_combo.currentText()
        if method == "IQR":
            return {"multiplier": self.iqr_spin.value()}
        if method == "Z-Score":
            return {"threshold": self.zscore_spin.value()}
        if method == "MAD":
            return {"threshold": self.mad_spin.value()}
        return {"window": self.hampel_window.value(), "n_sigmas": self.hampel_sigma.value()}

    def _apply(self) -> None:
        sku = self.state.selected_sku
        if sku:
            self.controller.apply_outliers(sku, self.method_combo.currentText(), self._get_params())

    def _restore(self) -> None:
        sku = self.state.selected_sku
        if sku:
            self.controller.restore_outliers(sku)

    def refresh(self) -> None:
        sku = self.state.selected_sku
        if not sku:
            return
        self.sku_label.setText(f"SKU: {sku}")
        hs = self.state.hourly_series.get(sku)
        if hs is None:
            return

        outlier = self.state.outlier_results.get(sku)
        treated = self.state.treated_series.get(sku)
        ds = compute_descriptive(hs)

        cards = [self.card_obs, self.card_outliers, self.card_pct, self.card_max_orig, self.card_max_treat]
        cards[0][1].setText(str(ds.n))
        if outlier:
            cards[1][1].setText(str(outlier.n_outliers))
            cards[2][1].setText(f"{outlier.outlier_ratio*100:.1f}%")
            if outlier.method == "IQR" and "upper" in outlier.params:
                self.method_hint.setText(
                    f"IQR aplicado — Q1={outlier.params.get('q1', 0):.2f}, "
                    f"Q3={outlier.params.get('q3', 0):.2f}, "
                    f"IQR={outlier.params.get('iqr', 0):.2f}, "
                    f"limite sup.={outlier.params.get('upper', 0):.2f}"
                )
        else:
            cards[1][1].setText("0")
            cards[2][1].setText("0%")
        cards[3][1].setText(f"{ds.max_:.1f}")
        if treated:
            cards[4][1].setText(f"{treated.quantities.max():.1f}")
        else:
            cards[4][1].setText(f"{ds.max_:.1f}")

        plot_time_series(self.plot.canvas, hs, outlier.mask if outlier else None)

        if outlier:
            outlier_rows = hs.timestamps[outlier.mask.values]
            self.table.setRowCount(len(outlier_rows))
            idx = 0
            for ts, qty, score, reason, is_out in zip(
                hs.timestamps, hs.quantities, outlier.scores, outlier.reasons, outlier.mask,
                strict=False,
            ):
                if not is_out:
                    continue
                self.table.setItem(idx, 0, QTableWidgetItem(str(ts)))
                self.table.setItem(idx, 1, QTableWidgetItem(f"{qty:.1f}"))
                self.table.setItem(idx, 2, QTableWidgetItem(outlier.method))
                self.table.setItem(idx, 3, QTableWidgetItem(f"{score:.3f}"))
                self.table.setItem(idx, 4, QTableWidgetItem(str(reason)))
                self.table.setItem(idx, 5, QTableWidgetItem("Sim"))
                idx += 1
