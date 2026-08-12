"""Reusable workflow hint + next-step navigation bar."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QWidget

from hospital_demand_analyzer.ui.nav import NEXT_STEP, WORKFLOW_HINTS


class WorkflowBar(QFrame):
    def __init__(self, step_key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.step_key = step_key
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet("background: #f4f6f8; padding: 4px;")
        layout = QHBoxLayout(self)
        self.hint = QLabel(WORKFLOW_HINTS.get(step_key, ""))
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint, 1)
        next_info = NEXT_STEP.get(step_key)
        if next_info:
            next_key, label = next_info
            btn = QPushButton(label)
            btn.clicked.connect(lambda: self._go(next_key))
            layout.addWidget(btn)

    def _go(self, step_key: str) -> None:
        win = self.window()
        if hasattr(win, "navigate_to"):
            win.navigate_to(step_key)
