"""Mixin: refresh heavy views only when visible."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QWidget


class ActiveViewMixin:
    """Avoid blocking the UI at startup by skipping refresh while hidden."""

    _view_active: bool

    def __init_active_view__(self) -> None:
        self._view_active = False

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)  # type: ignore[misc]
        self._view_active = True
        QTimer.singleShot(0, self.refresh_if_active)

    def hideEvent(self, event) -> None:
        super().hideEvent(event)  # type: ignore[misc]
        self._view_active = False

    def refresh_if_active(self) -> None:
        if getattr(self, "_view_active", False):
            self.refresh()  # type: ignore[attr-defined]
