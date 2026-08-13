"""Bulk CDF plots — all candidate distributions per SKU."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from hospital_demand_analyzer.domain.models import DistributionFitResult
from hospital_demand_analyzer.plots.canvas import MplCanvas
from hospital_demand_analyzer.statistics.distributions.zero_inflated import ZINB_NAME, ZIP_NAME, zinb_dist, zip_dist
from hospital_demand_analyzer.statistics.ecdf import compute_ecdf
from hospital_demand_analyzer.statistics.fit_params import best_fit, format_fit_params, params_tuple_for_fit

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


def _x_line_for_data(data: np.ndarray) -> np.ndarray:
    data = data.astype(float)
    if len(data) == 0:
        return np.array([0.01, 1.0])
    lo = max(float(data.min()), 0.0)
    hi = max(float(data.max()) * 1.1, lo + 0.01, 0.01)
    return np.linspace(lo if lo > 0 else 0.01, hi, 300)


def _plot_fits_on_ax(ax, data: np.ndarray, fits: list[DistributionFitResult] | None, *, positive_only: bool) -> None:
    x_emp, y_emp = compute_ecdf(data)
    ax.step(
        x_emp, y_emp, where="post", label="FDC empírica",
        linewidth=1.5, color="#2c3e50", linestyle="--",
    )

    chosen = best_fit(fits or [])
    x_line = _x_line_for_data(data)
    palette = matplotlib.colormaps["tab10"].colors

    for i, fit in enumerate(fits or []):
        if not fit.converged:
            continue
        dist = DIST_MAP.get(fit.name)
        if dist is None:
            continue
        params = params_tuple_for_fit(fit.name, fit.params)
        try:
            cdf = _conditional_cdf(dist, x_line, params, positive_only)
        except Exception:
            continue
        is_best = chosen is not None and fit.name == chosen.name
        lw = 2.8 if is_best else 1.4
        label = format_fit_params(fit) if is_best else fit.name
        ax.plot(x_line, cdf, linewidth=lw, color=palette[i % len(palette)], label=label)


def plot_sku_cdf_all_candidates(
    canvas: MplCanvas,
    data: np.ndarray,
    fits: list[DistributionFitResult] | None,
    *,
    highlight_best: bool = True,
    positive_only: bool = False,
    title: str = "",
) -> None:
    """Plot empirical FDC and all converged candidate CDFs; highlight best AIC fit."""
    ax = canvas.ax
    ax.clear()
    data = data.astype(float)
    if len(data) == 0:
        canvas.draw()
        return

    _plot_fits_on_ax(ax, data, fits, positive_only=positive_only)
    ax.set_xlabel("Demanda (quantidade)")
    ax.set_ylabel("F(x)")
    ax.set_title(title or "CDF — candidatas e escolhida (AIC)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    canvas.fig.tight_layout()
    canvas.draw()


def save_sku_cdf_png(
    path: Path,
    data: np.ndarray,
    fits: list[DistributionFitResult] | None,
    *,
    title: str = "",
    positive_only: bool = False,
) -> None:
    """Render SKU CDF to PNG file (off-screen)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = data.astype(float)
    if len(data) == 0:
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return

    _plot_fits_on_ax(ax, data, fits, positive_only=positive_only)
    ax.set_xlabel("Demanda (quantidade)")
    ax.set_ylabel("F(x)")
    ax.set_title(title or "CDF por SKU")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
