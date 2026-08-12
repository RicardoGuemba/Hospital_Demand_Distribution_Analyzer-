"""Shared helpers for outlier detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hospital_demand_analyzer.domain.models import OutlierResult


def reference_values(values: np.ndarray, positive_only: bool) -> np.ndarray:
    """Reference sample for robust statistics (demand: exclude structural zeros)."""
    if positive_only:
        ref = values[values > 0]
        return ref if len(ref) > 0 else values
    return values


def evaluation_mask(values: np.ndarray, positive_only: bool) -> np.ndarray:
    """Which observations are eligible to be flagged as outliers."""
    if positive_only:
        return values > 0
    return np.ones(len(values), dtype=bool)


def empty_result(method: str, series: pd.Series, params: dict) -> OutlierResult:
    return OutlierResult(
        method=method,
        params=params,
        mask=pd.Series(False, index=series.index),
        scores=pd.Series(0.0, index=series.index),
        reasons=pd.Series("", index=series.index),
    )


def tukey_fences(
    ref: np.ndarray,
    multiplier: float = 1.5,
) -> tuple[float, float, float, float, float]:
    """Classic Tukey boxplot fences on a reference sample."""
    q1, q3 = np.percentile(ref, [25, 75])
    iqr = float(q3 - q1)
    if iqr > 0:
        lower = float(q1 - multiplier * iqr)
        upper = float(q3 + multiplier * iqr)
    else:
        # IQR = 0: mild outliers are points outside [Q1, Q3]
        lower = float(q1)
        upper = float(q3)
    return float(q1), float(q3), iqr, lower, upper


def fence_scores(values: np.ndarray, lower: float, upper: float, iqr: float) -> np.ndarray:
    scale = iqr if iqr > 0 else max(abs(upper - lower), 1e-9)
    scores = np.zeros(len(values))
    above = values > upper
    below = values < lower
    scores[above] = (values[above] - upper) / scale
    scores[below] = (lower - values[below]) / scale
    return scores


def reason_labels(values: np.ndarray, lower: float, upper: float) -> list[str]:
    reasons: list[str] = []
    for v in values:
        if v > upper:
            reasons.append("acima_limite_superior")
        elif v < lower:
            reasons.append("abaixo_limite_inferior")
        else:
            reasons.append("")
    return reasons
