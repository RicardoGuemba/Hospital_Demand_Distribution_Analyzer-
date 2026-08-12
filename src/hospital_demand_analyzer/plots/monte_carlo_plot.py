"""Plot Monte Carlo inventory trajectories."""

from __future__ import annotations

import numpy as np

from hospital_demand_analyzer.domain.models import TempQMinResult
from hospital_demand_analyzer.plots.canvas import MplCanvas
from hospital_demand_analyzer.statistics.monte_carlo import format_tempqmin


def plot_monte_carlo_trajectories(
    canvas: MplCanvas,
    result: TempQMinResult,
    *,
    title: str = "",
    max_trajectories_display: int = 100,
) -> None:
    """Plot simulated inventory paths (time vs saldo) with Qmin reference line."""
    ax = canvas.ax
    ax.clear()

    trajectories = result.trajectories[:max_trajectories_display]
    if not trajectories:
        ax.set_title(title or "Simulação Monte Carlo — TempQMin")
        canvas.draw()
        return

    cmap = plt_colormap(len(trajectories))
    for i, traj in enumerate(trajectories):
        ax.plot(
            traj.times, traj.balances,
            color=cmap(i), alpha=0.35, linewidth=0.9,
        )

    ax.axhline(
        result.qmin, color="#c0392b", linestyle="--", linewidth=1.5,
        label=f"Qmin = {result.qmin:.1f}",
    )
    ax.axhline(
        result.saldo_inicial, color="#27ae60", linestyle=":", linewidth=1.0,
        label=f"Saldo inicial = {result.saldo_inicial:.1f}",
    )

    ax.set_xlabel("Tempo (minutos)")
    ax.set_ylabel("Saldo simulado")
    ax.set_title(
        title or (
            f"Simulação Monte Carlo — {result.produto}\n"
            f"{result.distribuicao} | N={result.n_trajetorias} | "
            f"TempQMin mediano={format_tempqmin(result.median)} | P5={format_tempqmin(result.p5)}"
        )
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    if trajectories:
        t_max = max(float(traj.times[-1]) for traj in trajectories)
        y_vals = np.concatenate([traj.balances for traj in trajectories])
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))
        pad = max((y_max - y_min) * 0.05, 1.0)
        ax.set_xlim(left=0.0, right=max(t_max * 1.02, 1.0))
        ax.set_ylim(bottom=max(0.0, y_min - pad), top=y_max + pad)
    canvas.fig.tight_layout()
    canvas.draw_idle()
    canvas.flush_events()


def plt_colormap(n: int):
    """Return a callable index→color using matplotlib tab20 cycle."""
    import matplotlib
    import matplotlib.colors as mcolors

    if n <= 1:
        return lambda _i: "#4a90d9"
    base = matplotlib.colormaps["tab20"]
    return lambda i: mcolors.to_hex(base(i % 20))
