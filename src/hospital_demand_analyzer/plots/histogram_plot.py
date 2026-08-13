"""Histogram plot."""

from __future__ import annotations

import numpy as np
from scipy import stats

from hospital_demand_analyzer.domain.demand_values import CONTINUOUS, DISCRETE, sanitize_demand
from hospital_demand_analyzer.plots.canvas import MplCanvas
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


def _compute_bins(data: np.ndarray, method: str, manual: int = 30) -> int | np.ndarray:
    if method == "fd":
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        if iqr == 0:
            return max(min(10, len(np.unique(data))), 1)
        h = 2 * iqr / (len(data) ** (1 / 3))
        return max(int((data.max() - data.min()) / h), 1) if h > 0 else 10
    if method == "sturges":
        return max(int(np.ceil(np.log2(len(data)) + 1)), 1)
    return manual


def _conditional_mass(dist, params: tuple, positive_only: bool) -> float:
    """Probability mass/weight above zero for normalizing overlays."""
    if not positive_only:
        return 1.0
    try:
        if hasattr(dist, "pmf"):
            p0 = float(dist.pmf(0, *params))
        else:
            p0 = float(dist.cdf(0, *params))
        return max(1.0 - p0, 1e-12)
    except Exception:
        return 1.0


def _plot_discrete_bars(ax, data: np.ndarray) -> float:
    vals, counts = np.unique(data, return_counts=True)
    ax.bar(vals, counts, width=0.85, alpha=0.75, label="Empírico", color="#4a90d9", edgecolor="white")
    return 1.0


def _plot_continuous_hist(ax, data: np.ndarray, bin_method: str) -> float:
    bins = _compute_bins(data, bin_method) if bin_method != "auto" else "auto"
    _, bin_edges, _ = ax.hist(
        data, bins=bins, density=False, alpha=0.75, label="Empírico", color="#4a90d9",
    )
    if len(bin_edges) > 1:
        return float(np.mean(np.diff(bin_edges)))
    return 1.0


def _overlay_fits(
    ax,
    data: np.ndarray,
    fits: list[DistributionFitResult],
    n: int,
    bin_width: float,
    positive_only: bool,
) -> None:
    x_line = np.linspace(max(float(data.min()), 0.01), float(data.max()) * 1.05 + 0.01, 300)
    k_max = min(int(data.max()), 120)

    for fit in fits[:3]:
        if not fit.converged:
            continue
        dist = DIST_MAP.get(fit.name)
        if dist is None:
            continue
        params = params_tuple_for_fit(fit.name, fit.params)
        norm = _conditional_mass(dist, params, positive_only)

        try:
            if fit.dist_type == "discrete":
                k_min = max(1, int(data.min())) if positive_only else 0
                x_d = np.arange(k_min, k_max + 1)
                pmf = dist.pmf(x_d, *params) / norm
                ax.plot(
                    x_d, pmf * n, "o-", ms=4, linewidth=1.3,
                    label=fit.name,
                )
            else:
                pdf = dist.pdf(x_line, *params) / norm
                ax.plot(
                    x_line, pdf * n * bin_width, linewidth=1.6,
                    label=fit.name,
                )
        except Exception:
            pass


def plot_histogram(
    canvas: MplCanvas,
    data: np.ndarray,
    bin_method: str = "auto",
    fits: list[DistributionFitResult] | None = None,
    title: str = "",
    exclude_zeros: bool = False,
    positive_only: bool | None = None,
    value_mode: str = DISCRETE,
) -> None:
    """Plot demand histogram.

    Valid demand is demanda ≥ 0. When ``positive_only``/``exclude_zeros`` is True,
    only demand > 0 is shown and fitted curves are scaled conditionally.
    """
    ax = canvas.ax
    ax.clear()
    data = sanitize_demand(np.asarray(data, dtype=float))

    if positive_only is None:
        positive_only = exclude_zeros
    if positive_only:
        data = data[data > 0]

    n = len(data)
    if n == 0:
        ax.set_title(title or "Histograma (sem observações válidas)")
        canvas.draw()
        return

    if value_mode == CONTINUOUS:
        use_discrete_bars = False
    else:
        is_discrete = np.all(data == np.floor(data))
        use_discrete_bars = is_discrete and len(np.unique(data)) <= 80

    if use_discrete_bars:
        bin_width = _plot_discrete_bars(ax, data)
    else:
        bin_width = _plot_continuous_hist(ax, data, bin_method)

    if fits:
        _overlay_fits(ax, data, fits, n, bin_width, positive_only)

    ylabel = "Frequência (horas com demanda > 0)" if positive_only else "Frequência (demanda ≥ 0)"
    ax.set_xlabel("Demanda (quantidade)")
    ax.set_ylabel(ylabel)
    default_title = "Histograma de Demanda (sem zeros)" if positive_only else "Histograma de Demanda (≥ 0)"
    ax.set_title(title or default_title)
    if positive_only and use_discrete_bars:
        ax.set_xlim(left=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    canvas.fig.tight_layout()
    canvas.draw()
