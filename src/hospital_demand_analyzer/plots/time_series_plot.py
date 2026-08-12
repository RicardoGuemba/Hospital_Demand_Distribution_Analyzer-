"""Time series plot."""

from __future__ import annotations

import pandas as pd

from hospital_demand_analyzer.domain.models import HourlySeries
from hospital_demand_analyzer.plots.canvas import MplCanvas


def plot_time_series(
    canvas: MplCanvas,
    hs: HourlySeries,
    outlier_mask: pd.Series | None = None,
    title: str = "",
) -> None:
    ax = canvas.ax
    ax.clear()
    ax.plot(hs.timestamps, hs.quantities, linewidth=0.8, label="Demanda")
    if outlier_mask is not None and outlier_mask.any():
        outlier_ts = hs.timestamps[outlier_mask.values]
        outlier_qty = hs.quantities[outlier_mask.values]
        ax.scatter(outlier_ts, outlier_qty, color="red", s=20, zorder=5, label="Outliers")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Quantidade")
    ax.set_title(title or f"Série Temporal — {hs.produto}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    canvas.fig.tight_layout()
    canvas.draw()
