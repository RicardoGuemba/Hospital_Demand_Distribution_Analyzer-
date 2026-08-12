"""Reusable matplotlib canvas widget."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, width: float = 8, height: float = 4, dpi: int = 100) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


class PlotWidget(QWidget):
    def __init__(self, width: float = 8, height: float = 4, parent=None) -> None:
        super().__init__(parent)
        self.canvas = MplCanvas(width, height)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def clear(self) -> None:
        self.canvas.ax.clear()
        self.canvas.draw()
