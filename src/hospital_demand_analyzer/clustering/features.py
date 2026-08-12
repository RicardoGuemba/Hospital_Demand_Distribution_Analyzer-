"""Turnover feature engineering for clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hospital_demand_analyzer.domain.models import HourlySeries

DEFAULT_FEATURES = [
    "mean_hourly_demand",
    "non_zero_hour_ratio",
    "p95_hourly_demand",
    "coefficient_of_variation",
]

ALL_FEATURES = [
    "mean_hourly_demand",
    "median_hourly_demand",
    "p95_hourly_demand",
    "p99_hourly_demand",
    "non_zero_hour_ratio",
    "demand_per_observed_hour",
    "std_hourly_demand",
    "coefficient_of_variation",
    "observation_hours",
    "total_demand",
    "zero_hour_ratio",
]


def compute_features(hs: HourlySeries) -> dict[str, float]:
    values = hs.quantities.values.astype(float)
    n = len(values)
    if n == 0:
        return {f: 0.0 for f in ALL_FEATURES}

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    non_zero = int((values > 0).sum())
    total = float(np.sum(values))

    return {
        "mean_hourly_demand": mean,
        "median_hourly_demand": float(np.median(values)),
        "p95_hourly_demand": float(np.percentile(values, 95)),
        "p99_hourly_demand": float(np.percentile(values, 99)),
        "non_zero_hour_ratio": non_zero / n,
        "demand_per_observed_hour": total / non_zero if non_zero > 0 else 0.0,
        "std_hourly_demand": std,
        "coefficient_of_variation": std / mean if mean > 0 else 0.0,
        "observation_hours": n,
        "total_demand": total,
        "zero_hour_ratio": (n - non_zero) / n,
    }


def build_feature_matrix(
    series_dict: dict[str, HourlySeries],
    features: list[str] | None = None,
) -> pd.DataFrame:
    features = features or DEFAULT_FEATURES
    rows = []
    for produto, hs in series_dict.items():
        feat = compute_features(hs)
        feat["produto"] = produto
        feat["descricao"] = hs.descricao
        rows.append(feat)
    df = pd.DataFrame(rows)
    return df
