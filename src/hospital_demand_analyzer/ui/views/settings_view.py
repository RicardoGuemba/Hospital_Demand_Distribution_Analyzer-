"""Settings view."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.domain.demand_values import VALUE_MODE_LABELS


class SettingsView(QWidget):
    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        layout = QVBoxLayout(self)

        demand_group = QGroupBox("Demanda")
        demand_form = QFormLayout(demand_group)
        self.value_mode_combo = QComboBox()
        for label in VALUE_MODE_LABELS.values():
            self.value_mode_combo.addItem(label)
        self.value_mode_combo.setCurrentIndex(
            0 if state.demand_value_mode == "discrete" else 1
        )
        self.value_mode_combo.setToolTip(
            "Padrão: discretos inteiros ≥ 0 (inclui zeros na grade).\n"
            "Alternativa: tratamento contínuo ≥ 0."
        )
        self.value_mode_combo.currentIndexChanged.connect(self._on_value_mode)
        demand_form.addRow("Representação:", self.value_mode_combo)
        layout.addWidget(demand_group)
        state.value_mode_changed.connect(self._sync_value_mode)

        cluster_group = QGroupBox("Clustering")
        form = QFormLayout(cluster_group)
        self.scaler_combo = QComboBox()
        self.scaler_combo.addItems(["robust", "standard"])
        self.scaler_combo.setCurrentText(state.config.default_scaler)
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 8)
        self.k_spin.setValue(state.config.kmeans_n_clusters)
        self.rs_spin = QSpinBox()
        self.rs_spin.setRange(0, 9999)
        self.rs_spin.setValue(state.config.random_state)
        form.addRow("Scaler:", self.scaler_combo)
        form.addRow("K (operacional):", self.k_spin)
        form.addRow("Random State:", self.rs_spin)
        layout.addWidget(cluster_group)

        filter_group = QGroupBox("Filtro Global de Cluster")
        filter_layout = QFormLayout(filter_group)
        self.cluster_filter = QComboBox()
        self.cluster_filter.addItems(["Todos", "Alto Giro", "Medio Giro", "Baixo Giro"])
        self.cluster_filter.currentTextChanged.connect(self._on_filter)
        filter_layout.addRow("Cluster:", self.cluster_filter)
        layout.addWidget(filter_group)
        layout.addStretch()

    def _on_value_mode(self, index: int) -> None:
        mode = "discrete" if index == 0 else "continuous"
        self.state.set_demand_value_mode(mode)

    def _sync_value_mode(self, mode: str) -> None:
        self.value_mode_combo.blockSignals(True)
        self.value_mode_combo.setCurrentIndex(0 if mode == "discrete" else 1)
        self.value_mode_combo.blockSignals(False)

    def _on_filter(self, text: str) -> None:
        self.state.cluster_filter = text
        self.state.cluster_changed.emit(text)
