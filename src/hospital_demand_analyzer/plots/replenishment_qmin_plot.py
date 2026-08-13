"""Plots for replenishment Qmin Monte Carlo — same style as TempQMin Monte Carlo."""

from __future__ import annotations

import math

import numpy as np

from hospital_demand_analyzer.domain.models import ReplenishmentQminResult
from hospital_demand_analyzer.plots.canvas import MplCanvas
from hospital_demand_analyzer.plots.monte_carlo_plot import plt_colormap


def _fmt_qty(value: float) -> str:
    if not math.isfinite(value):
        return "—"
    return str(int(round(value)))


def plot_demand_accumulation_trajectories(
    canvas: MplCanvas,
    result: ReplenishmentQminResult,
    *,
    max_trajectories: int = 100,
    title: str = "",
) -> None:
    """Plot simulated saldo paths (Monte Carlo style) until replenishment time ends."""
    ax = canvas.ax
    ax.clear()

    trajectories = result.trajectories[:max_trajectories]
    if not trajectories:
        ax.set_title(title or "Qmin reabastecimento — Monte Carlo")
        canvas.draw()
        return

    cmap = plt_colormap(len(trajectories))
    for i, traj in enumerate(trajectories):
        ax.plot(
            traj.times,
            traj.balances,
            color=cmap(i),
            alpha=0.35,
            linewidth=0.9,
        )

    qmin = int(round(result.qmin_proposed)) if math.isfinite(result.qmin_proposed) else 1
    saldo_inicial = int(result.saldo_inicial)
    ax.axhline(
        qmin,
        color="#c0392b",
        linestyle="--",
        linewidth=1.5,
        label=f"Qmin = {_fmt_qty(qmin)}",
    )
    ax.axhline(
        saldo_inicial,
        color="#27ae60",
        linestyle=":",
        linewidth=1.0,
        label=f"Saldo inicial = {_fmt_qty(saldo_inicial)}",
    )

    ax.set_xlabel("Tempo (minutos)")
    ax.set_ylabel("Saldo simulado")
    ax.set_title(
        title
        or (
            f"Simulação Qmin reabastecimento — {result.produto}\n"
            f"{result.distribuicao} | N={result.n_trajetorias} | "
            f"T={int(round(result.replenishment_minutes))} min | "
            f"Qmin P95={_fmt_qty(qmin)}"
        )
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    t_max = max(float(traj.times[-1]) for traj in trajectories)
    y_vals = np.concatenate([traj.balances for traj in trajectories])
    finite_y = y_vals[np.isfinite(y_vals)]
    if len(finite_y) == 0:
        finite_y = np.array([0.0, float(saldo_inicial)])
    y_min = float(np.min(finite_y))
    y_max = float(np.max(finite_y))
    pad = max((y_max - y_min) * 0.05, 1.0)
    ax.set_xlim(left=0.0, right=max(t_max * 1.02, 1.0))
    ax.set_ylim(bottom=max(0.0, y_min - pad), top=y_max + pad)
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: _fmt_qty(y)))
    canvas.fig.tight_layout()
    canvas.draw_idle()
    canvas.flush_events()


def save_replenishment_qmin_png(
    path,
    result: ReplenishmentQminResult,
    *,
    title: str = "",
    max_trajectories: int = 100,
) -> None:
    """Save Monte Carlo-style replenishment plot to PNG."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(8, 5))
    trajectories = result.trajectories[:max_trajectories]
    cmap = plt_colormap(len(trajectories) or 1)
    for i, traj in enumerate(trajectories):
        ax.plot(traj.times, traj.balances, color=cmap(i), alpha=0.35, linewidth=0.9)

    qmin = int(round(result.qmin_proposed)) if math.isfinite(result.qmin_proposed) else 1
    saldo_inicial = int(result.saldo_inicial)
    ax.axhline(qmin, color="#c0392b", linestyle="--", linewidth=1.5, label=f"Qmin = {_fmt_qty(qmin)}")
    ax.axhline(
        saldo_inicial, color="#27ae60", linestyle=":", linewidth=1.0,
        label=f"Saldo inicial = {_fmt_qty(saldo_inicial)}",
    )
    ax.set_xlabel("Tempo (minutos)")
    ax.set_ylabel("Saldo simulado")
    ax.set_title(title or f"Simulação Qmin reabastecimento — {result.produto}")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: _fmt_qty(y)))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
