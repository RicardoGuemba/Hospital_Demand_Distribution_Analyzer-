"""Classic Z-Score outlier detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hospital_demand_analyzer.domain.models import OutlierResult
from hospital_demand_analyzer.outliers.common import empty_result, evaluation_mask, reference_values


def detect_zscore(
    series: pd.Series,
    threshold: float = 3.0,
    *,
    positive_only: bool = True,
) -> OutlierResult:
    """Flag observations with |z| > threshold (default 3.0).

    Mean and standard deviation are computed on the reference sample
    (by default, hours with demand > 0).
    """
    values = series.values.astype(float)
    params = {
        "threshold": threshold,
        "positive_only": positive_only,
        "rule": "zscore",
    }

    ref = reference_values(values, positive_only)
    eligible = evaluation_mask(values, positive_only)

    if len(ref) < 2:
        return empty_result("Z-Score", series, {**params, "note": "insufficient_positive_observations"})

    mean = float(np.mean(ref))
    std = float(np.std(ref, ddof=1)) if len(ref) > 1 else 0.0
    if std == 0:
        std = 1e-9
    params.update({"mean": mean, "std": std})

    z = np.abs(values - mean) / std
    mask_arr = (z > threshold) & eligible
    reasons = ["zscore_alto" if m else "" for m in mask_arr]

    return OutlierResult(
        method="Z-Score",
        params=params,
        mask=pd.Series(mask_arr, index=series.index),
        scores=pd.Series(z, index=series.index),
        reasons=pd.Series(reasons, index=series.index),
    )
