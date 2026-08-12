"""Hampel filter outlier detection (rolling robust z-score)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hospital_demand_analyzer.domain.models import OutlierResult

MAD_TO_SIGMA = 1.4826


def detect_hampel(
    series: pd.Series, window: int = 5, n_sigmas: float = 3.0
) -> OutlierResult:
    """Rolling median/MAD filter; suited to local spikes in time series."""
    values = series.values.astype(float)
    n = len(values)
    window = max(3, window | 1)  # odd window >= 3
    mask_arr = np.zeros(n, dtype=bool)
    scores = np.zeros(n)
    half = window // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_vals = values[lo:hi]
        # Ignore structural zeros when estimating local scale
        positive = window_vals[window_vals > 0]
        ref = positive if len(positive) >= 2 else window_vals

        median = float(np.median(ref))
        mad = float(np.median(np.abs(ref - median)))
        if mad == 0:
            mad = 1e-9
        score = abs(values[i] - median) / (MAD_TO_SIGMA * mad)
        scores[i] = score
        # Only flag non-zero demand spikes (zero hours are not Hampel outliers)
        if score > n_sigmas and values[i] > 0:
            mask_arr[i] = True

    mask = pd.Series(mask_arr, index=series.index)
    score_s = pd.Series(scores, index=series.index)
    reasons = pd.Series(["hampel_local" if m else "" for m in mask_arr], index=series.index)
    return OutlierResult(
        method="Hampel",
        params={"window": window, "n_sigmas": n_sigmas, "rule": "rolling_hampel"},
        mask=mask,
        scores=score_s,
        reasons=reasons,
    )
