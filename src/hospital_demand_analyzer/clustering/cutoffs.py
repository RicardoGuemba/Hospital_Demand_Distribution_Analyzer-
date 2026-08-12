"""Threshold cutoff helpers and previews."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Métricas disponíveis para definir Alto / Médio / Baixo giro por demanda/h.
CUTOFF_FIELDS: dict[str, str] = {
    "mean_hourly_demand": "Demanda média/h (total ÷ horas da grade)",
    "demand_per_observed_hour": "Demanda média/h (só horas com consumo > 0)",
}

DEFAULT_CUTOFF_FIELD = "mean_hourly_demand"


def assign_cluster_name(metric_value: float, alto_cutoff: float, baixo_cutoff: float) -> str:
    """Assign cluster from a single demand/h metric and two cutoffs."""
    if metric_value >= alto_cutoff:
        return "Alto Giro"
    if metric_value < baixo_cutoff:
        return "Baixo Giro"
    return "Medio Giro"


def suggest_threshold_cutoffs(
    feature_df: pd.DataFrame,
    field: str = DEFAULT_CUTOFF_FIELD,
    baixo_pct: float = 25.0,
    alto_pct: float = 75.0,
) -> tuple[float, float]:
    """Data-driven default cutoffs (P25 / P75 of the chosen metric)."""
    if field not in feature_df.columns:
        field = DEFAULT_CUTOFF_FIELD
    vals = feature_df[field].dropna().astype(float)
    if len(vals) == 0:
        return 0.05, 0.5

    baixo = float(np.percentile(vals, baixo_pct))
    alto = float(np.percentile(vals, alto_pct))
    if alto <= baixo:
        alto = baixo + max(0.01, abs(baixo) * 0.5 + 0.01)
    return max(0.0, baixo), float(alto)


def assign_skus_by_cutoff(
    feature_df: pd.DataFrame,
    alto_cutoff: float,
    baixo_cutoff: float,
    field: str = DEFAULT_CUTOFF_FIELD,
) -> dict[str, str]:
    """Return produto → cluster_name using the cutoff criterion only."""
    if field not in feature_df.columns:
        field = DEFAULT_CUTOFF_FIELD
    mapping: dict[str, str] = {}
    for _, row in feature_df.iterrows():
        value = float(row[field])
        mapping[str(row["produto"])] = assign_cluster_name(value, alto_cutoff, baixo_cutoff)
    return mapping


def preview_cluster_counts(
    feature_df: pd.DataFrame,
    alto_cutoff: float,
    baixo_cutoff: float,
    field: str = DEFAULT_CUTOFF_FIELD,
) -> dict[str, int]:
    """Fast synchronous preview — same rule as full clustering."""
    mapping = assign_skus_by_cutoff(feature_df, alto_cutoff, baixo_cutoff, field)
    counts = {"Alto Giro": 0, "Medio Giro": 0, "Baixo Giro": 0}
    for name in mapping.values():
        counts[name] = counts.get(name, 0) + 1
    return counts
