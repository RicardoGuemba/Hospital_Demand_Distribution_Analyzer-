"""Monte Carlo TempQMin simulation view."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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
from hospital_demand_analyzer.app.workers import MonteCarloWorker
from hospital_demand_analyzer.domain.models import TempQMinResult
from hospital_demand_analyzer.plots.canvas import PlotWidget
from hospital_demand_analyzer.plots.monte_carlo_plot import plot_monte_carlo_trajectories
from hospital_demand_analyzer.statistics.descriptive import METRIC_TOOLTIPS
from hospital_demand_analyzer.statistics.distribution_moments import theoretical_expected_value
from hospital_demand_analyzer.statistics.fitter import top_n
from hospital_demand_analyzer.statistics.monte_carlo import (
    default_qmin,
    default_saldo_inicial,
    format_tempqmin,
    normalize_inventory_level,
)
from hospital_demand_analyzer.ui.nav import STEP_MONTE_CARLO
from hospital_demand_analyzer.ui.widgets.active_view import ActiveViewMixin
from hospital_demand_analyzer.ui.widgets.workflow_bar import WorkflowBar


class MonteCarloView(ActiveViewMixin, QWidget):
    def __init__(self, state: AppState, controller: AppController, parent=None) -> None:
        super().__init__(parent)
        self.__init_active_view__()
        self.state = state
        self.controller = controller
        self._mc_worker: MonteCarloWorker | None = None
        self._last_result: TempQMinResult | None = None
        self._last_run_params: tuple[str, int, int, int] | None = None
        self._pending_run = False
        self._loading_defaults = False
        self._param_timer = QTimer(self)
        self._param_timer.setSingleShot(True)
        self._param_timer.setInterval(400)
        self._param_timer.timeout.connect(self._auto_refresh_simulation)
        self._build_ui()
        state.sku_changed.connect(self._sync_product_combo)
        state.data_loaded.connect(self._populate_products)
        state.cluster_changed.connect(self._populate_products)
        state.fit_completed.connect(self._on_fit_completed)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(WorkflowBar(STEP_MONTE_CARLO))

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Produto:"))
        self.product_combo = QComboBox()
        self.product_combo.setMinimumWidth(280)
        self.product_combo.currentIndexChanged.connect(self._on_product_changed)
        controls.addWidget(self.product_combo, 1)

        self.btn_run = QPushButton("Monte Carlo")
        self.btn_run.setToolTip("Iniciar simulação com N trajetórias horárias até atingir Qmin")
        self.btn_run.clicked.connect(self._start_simulation)
        controls.addWidget(self.btn_run)
        layout.addLayout(controls)

        params = QHBoxLayout()
        params.addWidget(QLabel("Saldo inicial:"))
        self.saldo_spin = QSpinBox()
        self.saldo_spin.setRange(1, 1_000_000)
        self.saldo_spin.setToolTip("Inteiro positivo — padrão: Qmin × fator (5–10)")
        params.addWidget(self.saldo_spin)

        params.addWidget(QLabel("Qmin:"))
        self.qmin_spin = QSpinBox()
        self.qmin_spin.setRange(1, 1_000_000)
        self.qmin_spin.setToolTip("Inteiro positivo — padrão: média horária × 24 h")
        params.addWidget(self.qmin_spin)

        params.addWidget(QLabel("N trajetórias:"))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 50_000)
        self.n_spin.setValue(100)
        self.n_spin.setToolTip("Número de simulações Monte Carlo (padrão: 100)")
        params.addWidget(self.n_spin)
        params.addStretch()
        layout.addLayout(params)

        for spin in (self.saldo_spin, self.qmin_spin, self.n_spin):
            spin.valueChanged.connect(self._on_simulation_params_changed)

        self.info_label = QLabel(
            "Selecione um produto, ajuste saldo/Qmin se necessário e clique em Monte Carlo."
        )
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.dist_label = QLabel("Distribuição: —")
        layout.addWidget(self.dist_label)

        self.product_stats_group = QGroupBox("Estatísticas do produto (demanda/h)")
        self.product_stats_grid = QGridLayout(self.product_stats_group)
        self._product_stat_labels: dict[str, QLabel] = {}
        product_stat_fields = [
            ("mean", "Média aritmética"),
            ("expected_empirical", "E[X] empírico"),
            ("expected_conditional", "E[X|X>0]"),
            ("expected_theoretical", "E[X] teórico (ajuste)"),
            ("std", "Desvio padrão"),
            ("min", "Mínimo"),
            ("max", "Máximo"),
            ("p90", "P90"),
            ("p95", "P95"),
            ("p99", "P99"),
            ("non_zero_hours", "Horas c/ demanda"),
            ("zero_ratio", "% Horas zero"),
            ("sum", "Demanda total"),
        ]
        for i, (key, label) in enumerate(product_stat_fields):
            row, col = divmod(i, 4)
            name_lbl = QLabel(f"{label}:")
            if key in METRIC_TOOLTIPS:
                name_lbl.setToolTip(METRIC_TOOLTIPS[key])
            self.product_stats_grid.addWidget(name_lbl, row, col * 2)
            val_lbl = QLabel("—")
            if key in METRIC_TOOLTIPS:
                val_lbl.setToolTip(METRIC_TOOLTIPS[key])
            self._product_stat_labels[key] = val_lbl
            self.product_stats_grid.addWidget(val_lbl, row, col * 2 + 1)
        layout.addWidget(self.product_stats_group)

        splitter = QSplitter()
        self.traj_plot = PlotWidget(8, 5)
        splitter.addWidget(self.traj_plot)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        stats_group = QGroupBox("TempQMin — estatísticas")
        stats_grid = QGridLayout(stats_group)
        self._stat_labels: dict[str, QLabel] = {}
        stat_names = [
            ("mean", "Média"), ("median", "Mediana (P50)"), ("p5", "P5"),
            ("p10", "P10"), ("p90", "P90"), ("p95", "P95"),
            ("std", "Desvio padrão"), ("min_", "Mínimo"), ("max_", "Máximo"),
        ]
        for i, (key, label) in enumerate(stat_names):
            stats_grid.addWidget(QLabel(f"{label}:"), i, 0)
            val_lbl = QLabel("—")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._stat_labels[key] = val_lbl
            stats_grid.addWidget(val_lbl, i, 1)
        right_layout.addWidget(stats_group)

        self.traj_table = QTableWidget(0, 2)
        self.traj_table.setHorizontalHeaderLabels(["Trajetória", "TempQMin"])
        self.traj_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.traj_table)
        splitter.addWidget(right)
        splitter.setSizes([700, 350])
        layout.addWidget(splitter)

    def populate_light(self) -> None:
        self._populate_products()

    def refresh(self) -> None:
        self._populate_products()
        self._load_defaults_for_product()

    def _populate_products(self) -> None:
        current = self.product_combo.currentData()
        self.product_combo.blockSignals(True)
        self.product_combo.clear()
        skus = self.state.filtered_skus()
        df = self.state.descriptive_df
        for sku in sorted(skus):
            desc = ""
            if df is not None and not df.empty:
                match = df[df["produto"].astype(str) == sku]
                if not match.empty:
                    desc = str(match.iloc[0].get("descricao", ""))[:40]
            label = f"{sku} — {desc}" if desc else sku
            self.product_combo.addItem(label, sku)
        if current:
            idx = self.product_combo.findData(current)
            if idx >= 0:
                self.product_combo.setCurrentIndex(idx)
        elif self.state.selected_sku:
            idx = self.product_combo.findData(self.state.selected_sku)
            if idx >= 0:
                self.product_combo.setCurrentIndex(idx)
        self.product_combo.blockSignals(False)
        self._load_defaults_for_product()

    def _sync_product_combo(self, sku: str) -> None:
        if not sku:
            return
        idx = self.product_combo.findData(sku)
        if idx >= 0:
            self.product_combo.setCurrentIndex(idx)

    def _on_product_changed(self, _index: int) -> None:
        sku = self.product_combo.currentData()
        if sku:
            self.controller.select_sku(str(sku))
        self._load_defaults_for_product()

    def _load_defaults_for_product(self) -> None:
        sku = self.product_combo.currentData()
        if not sku:
            return
        df = self.state.descriptive_df
        if df is None or df.empty:
            return
        match = df[df["produto"].astype(str) == str(sku)]
        if match.empty:
            return
        row = match.iloc[0]
        mean_h = float(row.get("mean", 0))
        qmin = default_qmin(mean_h)
        rng = np.random.default_rng(
            hash(str(sku)) % (2**32) ^ self.state.config.random_state
        )
        saldo = default_saldo_inicial(qmin, rng)

        self._loading_defaults = True
        self._fill_product_stats(row)
        self._clear_simulation_stats()

        self.saldo_spin.blockSignals(True)
        self.qmin_spin.blockSignals(True)
        self.n_spin.blockSignals(True)
        self.saldo_spin.setValue(saldo)
        self.qmin_spin.setValue(qmin)
        self.saldo_spin.blockSignals(False)
        self.qmin_spin.blockSignals(False)
        self.n_spin.blockSignals(False)
        self._loading_defaults = False
        self._last_run_params = None
        self._update_dist_label(str(sku))

    def _current_run_params(self) -> tuple[str, int, int, int] | None:
        sku = self.product_combo.currentData()
        if not sku:
            return None
        return (
            str(sku),
            self.saldo_spin.value(),
            self.qmin_spin.value(),
            self.n_spin.value(),
        )

    def _params_match_last_run(self) -> bool:
        current = self._current_run_params()
        return current is not None and current == self._last_run_params

    def _on_simulation_params_changed(self, *_args) -> None:
        if self._loading_defaults or self._last_run_params is None:
            return
        if self._params_match_last_run():
            return
        self._param_timer.stop()
        self.traj_plot.clear()
        self.info_label.setText("Parâmetros alterados — recalculando simulação...")
        self._param_timer.start()

    def _auto_refresh_simulation(self) -> None:
        if self._loading_defaults or self._params_match_last_run():
            return
        if self._mc_worker is not None and self._mc_worker.isRunning():
            return
        sku = self.product_combo.currentData()
        if not sku:
            return
        if self._get_best_fit(str(sku)) is None:
            self.info_label.setText(
                "Parâmetros alterados — aguardando ajuste de distribuição."
            )
            self._pending_run = True
            self.controller.request_fit(
                str(sku), positive_only=False, value_mode=self.state.demand_value_mode,
            )
            return
        self._start_simulation()

    def _fill_product_stats(self, row) -> None:
        mapping = {
            "mean": ("mean", "{:.3f}"),
            "expected_empirical": ("expected_empirical", "{:.3f}"),
            "expected_conditional": ("expected_conditional", "{:.3f}"),
            "std": ("std", "{:.3f}"),
            "min": ("min", "{:.1f}"),
            "max": ("max", "{:.1f}"),
            "p90": ("p90", "{:.1f}"),
            "p95": ("p95", "{:.1f}"),
            "p99": ("p99", "{:.1f}"),
            "non_zero_hours": ("non_zero_hours", "{:.0f}"),
            "zero_ratio": ("zero_ratio", "{:.1%}"),
            "sum": ("sum", "{:.1f}"),
        }
        for key, (col, fmt) in mapping.items():
            val = row.get(col, 0)
            if key == "zero_ratio":
                text = fmt.format(float(val))
            else:
                text = fmt.format(float(val))
            self._product_stat_labels[key].setText(text)
        self._refresh_theoretical_product_stat()

    def _refresh_theoretical_product_stat(self) -> None:
        sku = self.product_combo.currentData()
        if not sku or "expected_theoretical" not in self._product_stat_labels:
            return
        fit = self._get_best_fit(str(sku))
        theo = theoretical_expected_value(fit) if fit else None
        self._product_stat_labels["expected_theoretical"].setText(
            f"{theo:.3f}" if theo is not None else "—"
        )

    def _clear_simulation_stats(self) -> None:
        self._param_timer.stop()
        for lbl in self._stat_labels.values():
            lbl.setText("—")
        self.traj_table.setRowCount(0)
        self.traj_plot.clear()
        self._last_result = None
        self._last_run_params = None

    def _update_dist_label(self, sku: str) -> None:
        fits = self.controller.get_fit_results(
            sku, positive_only=False, value_mode=self.state.demand_value_mode,
        )
        if not fits:
            self.dist_label.setText(
                f"Distribuição: será obtida ao clicar Monte Carlo (ajuste em background)"
            )
            return
        best = top_n(fits, 1)
        if best:
            p = best[0].params
            params_str = ", ".join(f"{k}={v:.4g}" for k, v in p.items())
            self.dist_label.setText(f"Distribuição (melhor AIC): {best[0].name} — {params_str}")
        else:
            self.dist_label.setText("Distribuição: nenhum ajuste convergente disponível")

    def _get_best_fit(self, sku: str):
        fits = self.controller.get_fit_results(
            sku, positive_only=False, value_mode=self.state.demand_value_mode,
        )
        if fits:
            best = top_n(fits, 1)
            if best:
                return best[0]
        return None

    def _start_simulation(self) -> None:
        sku = self.product_combo.currentData()
        if not sku:
            self.info_label.setText("Selecione um produto antes de simular.")
            return
        sku = str(sku)
        fit = self._get_best_fit(sku)
        if fit is None:
            self._pending_run = True
            self.info_label.setText("Calculando ajuste de distribuição antes da simulação...")
            self.btn_run.setEnabled(False)
            self.controller.request_fit(
                sku, positive_only=False, value_mode=self.state.demand_value_mode,
            )
            return
        self._run_with_fit(sku, fit)

    def _on_fit_completed(self, produto: str) -> None:
        sku = self.product_combo.currentData()
        if sku and str(sku) == produto:
            self._refresh_theoretical_product_stat()
            self._update_dist_label(str(produto))
        if not self._pending_run:
            return
        sku = self.product_combo.currentData()
        if sku and str(sku) != produto:
            return
        self._pending_run = False
        self.btn_run.setEnabled(True)
        fit = self._get_best_fit(produto)
        if fit:
            self._run_with_fit(produto, fit)
        else:
            self.info_label.setText("Ajuste indisponível — não foi possível iniciar a simulação.")

    def _run_with_fit(self, sku: str, fit) -> None:
        if self._mc_worker is not None and self._mc_worker.isRunning():
            return

        saldo = normalize_inventory_level(self.saldo_spin.value())
        qmin = normalize_inventory_level(self.qmin_spin.value())
        self.saldo_spin.blockSignals(True)
        self.qmin_spin.blockSignals(True)
        self.saldo_spin.setValue(saldo)
        self.qmin_spin.setValue(qmin)
        self.saldo_spin.blockSignals(False)
        self.qmin_spin.blockSignals(False)
        n = self.n_spin.value()

        self.btn_run.setEnabled(False)
        self.info_label.setText(
            f"Simulando {n} trajetórias para {sku} (passo=60 min, demanda>0, {fit.name})..."
        )
        self._update_dist_label(sku)

        worker = MonteCarloWorker(sku, saldo, qmin, fit, n, seed=self.state.config.random_state)
        worker.finished_ok.connect(
            self._on_simulation_done,
            Qt.ConnectionType.QueuedConnection,
        )
        worker.finished_err.connect(
            self._on_simulation_error,
            Qt.ConnectionType.QueuedConnection,
        )
        if self.controller.parent is not None:
            worker.setParent(self.controller.parent)
        worker.finished.connect(worker.deleteLater)
        self._mc_worker = worker
        worker.start()

    def _on_simulation_done(self, result: TempQMinResult) -> None:
        self._mc_worker = None
        self.btn_run.setEnabled(True)
        self._last_result = result
        self._last_run_params = (
            result.produto,
            int(result.saldo_inicial),
            int(result.qmin),
            result.n_trajetorias,
        )
        self.saldo_spin.blockSignals(True)
        self.qmin_spin.blockSignals(True)
        self.n_spin.blockSignals(True)
        self.saldo_spin.setValue(int(result.saldo_inicial))
        self.qmin_spin.setValue(int(result.qmin))
        self.n_spin.setValue(result.n_trajetorias)
        self.saldo_spin.blockSignals(False)
        self.qmin_spin.blockSignals(False)
        self.n_spin.blockSignals(False)
        self.info_label.setText(
            f"Simulação concluída — {result.produto} | "
            f"{result.distribuicao} | N={result.n_trajetorias} | "
            f"Saldo={result.saldo_inicial} | Qmin={result.qmin}"
        )
        self._fill_stats(result)
        self._fill_traj_table(result)
        QTimer.singleShot(0, lambda: self._render_plot(result))

    def _render_plot(self, result: TempQMinResult) -> None:
        try:
            plot_monte_carlo_trajectories(self.traj_plot.canvas, result)
            self.traj_plot.canvas.update()
        except Exception as exc:
            self.info_label.setText(
                f"{self.info_label.text()}\nErro ao exibir gráfico: {exc}"
            )

    def _on_simulation_error(self, message: str) -> None:
        self._mc_worker = None
        self.btn_run.setEnabled(True)
        self._pending_run = False
        self.info_label.setText(f"Erro na simulação: {message}")

    def _fill_stats(self, result: TempQMinResult) -> None:
        mapping = {
            "mean": result.mean, "median": result.median, "p5": result.p5,
            "p10": result.p10, "p90": result.p90, "p95": result.p95,
            "std": result.std, "min_": result.min_, "max_": result.max_,
        }
        for key, val in mapping.items():
            self._stat_labels[key].setText(format_tempqmin(val))

    def _fill_traj_table(self, result: TempQMinResult) -> None:
        times = result.temp_qmin_times
        self.traj_table.setRowCount(len(times))
        for i, t in enumerate(times):
            self.traj_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.traj_table.setItem(i, 1, QTableWidgetItem(format_tempqmin(float(t))))
