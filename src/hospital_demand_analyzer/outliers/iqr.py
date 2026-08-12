"""IQR / Tukey boxplot outlier detection (classic 1.5×IQR)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hospital_demand_analyzer.domain.models import OutlierResult
from hospital_demand_analyzer.outliers.common import (
    empty_result,
    evaluation_mask,
    fence_scores,
    reason_labels,
    reference_values,
    tukey_fences,
)


def detect_iqr(
    series: pd.Series,
    multiplier: float = 1.5,
    *,
    positive_only: bool = True,
) -> OutlierResult:
    """Tukey fences: Q1 ± k·IQR (default k=1.5).

    For hourly demand, quartiles are computed on demand > 0 so structural
    zero-hours are not treated as part of the bulk distribution.
    """
    values = series.values.astype(float)
    params = {
        "multiplier": multiplier,
        "positive_only": positive_only,
        "rule": "tukey_1.5xIQR",
    }

    ref = reference_values(values, positive_only)
    eligible = evaluation_mask(values, positive_only)

    if len(ref) < 4:
        return empty_result("IQR", series, {**params, "note": "insufficient_positive_observations"})

    q1, q3, iqr, lower, upper = tukey_fences(ref, multiplier)
    params.update({"q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper})

    outside = (values < lower) | (values > upper)
    mask_arr = outside & eligible
    scores = fence_scores(values, lower, upper, iqr)

    return OutlierResult(
        method="IQR",
        params=params,
        mask=pd.Series(mask_arr, index=series.index),
        scores=pd.Series(scores, index=series.index),
        reasons=pd.Series(reason_labels(values, lower, upper), index=series.index),
    )
