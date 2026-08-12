"""MAD / modified Z-Score outlier detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hospital_demand_analyzer.domain.models import OutlierResult
from hospital_demand_analyzer.outliers.common import empty_result, evaluation_mask, reference_values

MAD_SCALE = 0.6745  # consistency with normal distribution


def detect_mad(
    series: pd.Series,
    threshold: float = 3.5,
    *,
    positive_only: bool = True,
) -> OutlierResult:
    """Modified Z-Score based on median and MAD (Iglewicz & Hoaglin, 3.5 default)."""
    values = series.values.astype(float)
    params = {
        "threshold": threshold,
        "positive_only": positive_only,
        "rule": "modified_zscore_mad",
    }

    ref = reference_values(values, positive_only)
    eligible = evaluation_mask(values, positive_only)

    if len(ref) < 2:
        return empty_result("MAD", series, {**params, "note": "insufficient_positive_observations"})

    median = float(np.median(ref))
    mad = float(np.median(np.abs(ref - median)))
    if mad == 0:
        mad = 1e-9
    params.update({"median": median, "mad": mad})

    modified_z = MAD_SCALE * np.abs(values - median) / mad
    mask_arr = (modified_z > threshold) & eligible
    reasons = ["mad_excede_limite" if m else "" for m in mask_arr]

    return OutlierResult(
        method="MAD",
        params=params,
        mask=pd.Series(mask_arr, index=series.index),
        scores=pd.Series(modified_z, index=series.index),
        reasons=pd.Series(reasons, index=series.index),
    )
