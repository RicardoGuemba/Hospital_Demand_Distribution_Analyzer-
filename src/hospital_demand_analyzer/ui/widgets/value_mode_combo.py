"""Combo box for discrete vs continuous demand representation."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox

from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.domain.demand_values import CONTINUOUS, DISCRETE, VALUE_MODE_LABELS


class ValueModeCombo(QComboBox):
    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._syncing = False
        for mode in (DISCRETE, CONTINUOUS):
            self.addItem(VALUE_MODE_LABELS[mode], mode)
        self.setCurrentIndex(0 if state.demand_value_mode == DISCRETE else 1)
        self.setToolTip(
            "Discreto: barras inteiras e Poisson/NB (padrão, demanda ≥ 0).\n"
            "Contínuo: histograma contínuo e Gamma/Weibull/etc. (demanda ≥ 0)."
        )
        self.currentIndexChanged.connect(self._on_changed)
        state.value_mode_changed.connect(self._on_state_changed)

    def _on_changed(self, _index: int) -> None:
        if self._syncing:
            return
        self._state.set_demand_value_mode(self.currentData())

    def _on_state_changed(self, mode: str) -> None:
        self._syncing = True
        self.setCurrentIndex(0 if mode == DISCRETE else 1)
        self._syncing = False
