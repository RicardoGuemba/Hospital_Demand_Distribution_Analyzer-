"""Cluster scatter plot."""

from __future__ import annotations

import pandas as pd

from hospital_demand_analyzer.clustering.cutoffs import CUTOFF_FIELDS
from hospital_demand_analyzer.domain.models import ClusterAssignment, ClusterResult
from hospital_demand_analyzer.plots.canvas import MplCanvas

CLUSTER_COLORS = {"Baixo Giro": "#3498db", "Medio Giro": "#f39c12", "Alto Giro": "#e74c3c"}


def _assignment_xy(
    a: ClusterAssignment,
    x_feat: str,
    y_feat: str,
    cutoff_field: str | None = None,
) -> tuple[float, float]:
    if cutoff_field and x_feat == cutoff_field:
        x = a.cutoff_metric_value
    else:
        feat_map = {
            "mean_hourly_demand": a.mean_hourly_demand,
            "demand_per_observed_hour": a.cutoff_metric_value,
            "non_zero_hour_ratio": a.non_zero_hour_ratio,
            "p95_hourly_demand": a.p95,
            "p99_hourly_demand": a.p99,
            "coefficient_of_variation": a.cv,
        }
        x = feat_map.get(x_feat, a.cutoff_metric_value)
    y_map = {
        "non_zero_hour_ratio": a.non_zero_hour_ratio,
        "mean_hourly_demand": a.mean_hourly_demand,
        "demand_per_observed_hour": a.cutoff_metric_value,
        "p95_hourly_demand": a.p95,
        "p99_hourly_demand": a.p99,
        "coefficient_of_variation": a.cv,
    }
    return x, y_map.get(y_feat, a.non_zero_hour_ratio)


def plot_cluster_scatter(
    canvas: MplCanvas,
    assignments: list[ClusterAssignment],
    cluster_result: ClusterResult | None = None,
    selected_produto: str | None = None,
    x_feat: str | None = None,
    y_feat: str = "non_zero_hour_ratio",
) -> None:
    ax = canvas.ax
    ax.clear()
    if not assignments:
        canvas.draw()
        return

    if cluster_result is not None and cluster_result.method == "threshold":
        x_feat = x_feat or cluster_result.cutoff_field
    else:
        x_feat = x_feat or "mean_hourly_demand"

    cutoff_field = cluster_result.cutoff_field if cluster_result else None
    rows = []
    for a in assignments:
        x_val, y_val = _assignment_xy(a, x_feat, y_feat, cutoff_field)
        rows.append({
            "produto": a.produto,
            "cluster_name": a.cluster_name,
            "x": x_val,
            "y": y_val,
        })
    df = pd.DataFrame(rows)

    for name, group in df.groupby("cluster_name"):
        color = CLUSTER_COLORS.get(name, "#999")
        ax.scatter(group["x"], group["y"], c=color, label=name, alpha=0.6, s=30)

    if cluster_result is not None and cluster_result.method == "threshold":
        x_is_cutoff = x_feat == cluster_result.cutoff_field
        if x_is_cutoff and cluster_result.alto_cutoff is not None:
            ax.axvline(
                cluster_result.alto_cutoff,
                color="#e74c3c",
                linestyle="--",
                linewidth=1.2,
                label=f"Alto ≥ {cluster_result.alto_cutoff:.3f}",
            )
        if x_is_cutoff and cluster_result.baixo_cutoff is not None:
            ax.axvline(
                cluster_result.baixo_cutoff,
                color="#3498db",
                linestyle="--",
                linewidth=1.2,
                label=f"Baixo < {cluster_result.baixo_cutoff:.3f}",
            )

    if cluster_result is not None and not cluster_result.centroids.empty:
        cx = cluster_result.centroids.get(x_feat, cluster_result.centroids.get("mean_hourly_demand"))
        cy = cluster_result.centroids.get(y_feat, cluster_result.centroids.get("non_zero_hour_ratio"))
        if cx is not None and cy is not None:
            ax.scatter(cx, cy, c="black", marker="X", s=120, zorder=5, label="Centroides")

    if selected_produto:
        sel = df[df["produto"] == selected_produto]
        if not sel.empty:
            ax.scatter(sel["x"], sel["y"], c="lime", edgecolors="black", s=100, zorder=6, label="Selecionado")

    x_label = CUTOFF_FIELDS.get(x_feat, x_feat)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_feat)
    ax.set_title("Clusters de Giro — eixo X = critério de corte")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    canvas.fig.tight_layout()
    canvas.draw()
