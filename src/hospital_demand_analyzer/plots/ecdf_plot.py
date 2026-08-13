"""ECDF / FDC plots."""

from __future__ import annotations

import numpy as np
from scipy import stats

from hospital_demand_analyzer.domain.models import DistributionFitResult
from hospital_demand_analyzer.plots.canvas import MplCanvas
from hospital_demand_analyzer.statistics.ecdf import compute_ecdf
from hospital_demand_analyzer.statistics.distributions.zero_inflated import ZINB_NAME, ZIP_NAME, zinb_dist, zip_dist
from hospital_demand_analyzer.statistics.fit_params import params_tuple_for_fit

DIST_MAP = {
    "Poisson": stats.poisson,
    "Negative Binomial": stats.nbinom,
    ZIP_NAME: zip_dist,
    ZINB_NAME: zinb_dist,
    "Exponential": stats.expon,
    "Gamma": stats.gamma,
    "Lognormal": stats.lognorm,
    "Weibull": stats.weibull_min,
}


def plot_fdc_empirical(
    canvas: MplCanvas,
    data: np.ndarray,
    title: str = "",
) -> None:
    """Plot empirical FDC (ECDF) only."""
    ax = canvas.ax
    ax.clear()
    data = data.astype(float)
    if len(data) == 0:
        canvas.draw()
        return

    x, y = compute_ecdf(data)
    ax.step(x, y, where="post", label="FDC empírica", linewidth=1.8, color="#2c3e50")
    ax.set_xlabel("Demanda (quantidade)")
    ax.set_ylabel("F(x)")
    ax.set_title(title or "FDC Empírica")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    canvas.fig.tight_layout()
    canvas.draw()


def _conditional_cdf(dist, x: np.ndarray, params: tuple, positive_only: bool) -> np.ndarray:
    cdf = dist.cdf(x, *params)
    if not positive_only:
        return cdf
    try:
        c0 = float(dist.cdf(0, *params))
    except Exception:
        c0 = 0.0
    denom = max(1.0 - c0, 1e-12)
    return np.clip((cdf - c0) / denom, 0.0, 1.0)


def plot_fdc_adjusted(
    canvas: MplCanvas,
    data: np.ndarray,
    fits: list[DistributionFitResult] | None = None,
    show_empirical_reference: bool = True,
    title: str = "",
    positive_only: bool = False,
) -> None:
    """Plot adjusted FDC (fitted CDFs) with optional empirical reference."""
    ax = canvas.ax
    ax.clear()
    data = data.astype(float)
    if len(data) == 0:
        canvas.draw()
        return

    if show_empirical_reference:
        x, y = compute_ecdf(data)
        ax.step(
            x, y, where="post", label="FDC empírica (ref.)",
            linewidth=1.2, color="#95a5a6", linestyle="--",
        )

    colors = ["#e74c3c", "#3498db"]
    if fits:
        x_line = np.linspace(max(float(data.min()), 0.01), max(data.max() * 1.1, 0.01), 300)
        for i, fit in enumerate(fits[:3]):
            if not fit.converged:
                continue
            dist = DIST_MAP.get(fit.name)
            if dist is None:
                continue
            params = params_tuple_for_fit(fit.name, fit.params)
            try:
                cdf = _conditional_cdf(dist, x_line, params, positive_only)
                ax.plot(
                    x_line, cdf, linewidth=1.8, color=colors[i % len(colors)],
                    label=f"FDC ajustada — {fit.name}",
                )
            except Exception:
                pass

    ax.set_xlabel("Demanda (quantidade)")
    ax.set_ylabel("F(x)")
    ax.set_title(title or "FDC Ajustada")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    canvas.fig.tight_layout()
    canvas.draw()


def plot_ecdf(
    canvas: MplCanvas,
    data: np.ndarray,
    fits: list[DistributionFitResult] | None = None,
    tail_focus: bool = False,
    title: str = "",
    positive_only: bool = False,
) -> None:
    """Combined ECDF + fitted CDFs (legacy helper)."""
    plot_fdc_empirical(canvas, data, title=title or "ECDF / FDC")
    if fits:
        ax = canvas.ax
        x_line = np.linspace(max(float(data.min()), 0.01), max(data.max() * 1.1, 0.01), 300)
        for fit in fits[:3]:
            if not fit.converged:
                continue
            dist = DIST_MAP.get(fit.name)
            if dist is None:
                continue
            params = params_tuple_for_fit(fit.name, fit.params)
            try:
                cdf = _conditional_cdf(dist, x_line, params, positive_only)
                ax.plot(x_line, cdf, label=f"CDF {fit.name}")
            except Exception:
                pass
        if tail_focus:
            p90 = np.percentile(data, 90)
            ax.set_xlim(left=max(p90 * 0.9, 0))
        ax.legend(fontsize=8)
        canvas.draw()
