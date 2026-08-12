"""Outlier detection dispatcher and treatment."""

from __future__ import annotations

import numpy as np

from hospital_demand_analyzer.domain.models import HourlySeries, OutlierResult
from hospital_demand_analyzer.outliers.hampel import detect_hampel
from hospital_demand_analyzer.outliers.iqr import detect_iqr
from hospital_demand_analyzer.outliers.mad import detect_mad
from hospital_demand_analyzer.outliers.zscore import detect_zscore

DEFAULT_PARAMS: dict[str, dict] = {
    "IQR": {"multiplier": 1.5},
    "Z-SCORE": {"threshold": 3.0},
    "MAD": {"threshold": 3.5},
    "HAMPEL": {"window": 5, "n_sigmas": 3.0},
}


def apply_outlier_method(
    hs: HourlySeries,
    method: str,
    params: dict | None = None,
) -> OutlierResult:
    params = {**DEFAULT_PARAMS.get(method.upper(), {}), **(params or {})}
    series = hs.quantities
    method_upper = method.upper().replace("_", "-")

    if method_upper == "IQR":
        return detect_iqr(series, multiplier=float(params.get("multiplier", 1.5)))
    if method_upper in ("Z-SCORE", "ZSCORE"):
        return detect_zscore(series, threshold=float(params.get("threshold", 3.0)))
    if method_upper == "MAD":
        return detect_mad(series, threshold=float(params.get("threshold", 3.5)))
    if method_upper == "HAMPEL":
        return detect_hampel(
            series,
            window=int(params.get("window", 5)),
            n_sigmas=float(params.get("n_sigmas", 3.0)),
        )
    raise ValueError(f"Unknown outlier method: {method}")


def _replacement_value(value: float, outlier: OutlierResult) -> float:
    """Winsorize to fence or robust center — never zero a demand spike by default."""
    lower = outlier.params.get("lower")
    upper = outlier.params.get("upper")

    if upper is not None and value > upper:
        return float(upper)
    if lower is not None and value < lower:
        return float(max(lower, 0.0))

    median = outlier.params.get("median")
    if median is not None:
        return float(median)

    mean = outlier.params.get("mean")
    if mean is not None:
        return float(mean)

    q3 = outlier.params.get("q3")
    if q3 is not None:
        return float(q3)

    return float(value)


def apply_treatment(hs: HourlySeries, outlier: OutlierResult) -> HourlySeries:
    """Cap/winsorize outlier hours to robust limits (preserves demand scale)."""
    treated_qty = hs.quantities.copy()
    mask = outlier.mask.values
    if not mask.any():
        return HourlySeries(
            produto=hs.produto,
            descricao=hs.descricao,
            timestamps=hs.timestamps,
            quantities=treated_qty,
            gap_types=hs.gap_types,
            first_record_at=hs.first_record_at,
            last_record_at=hs.last_record_at,
        )

    values = treated_qty.values.astype(float)
    for i, is_out in enumerate(mask):
        if is_out:
            values[i] = _replacement_value(values[i], outlier)
    treated_qty = treated_qty.copy()
    treated_qty.iloc[:] = values

    return HourlySeries(
        produto=hs.produto,
        descricao=hs.descricao,
        timestamps=hs.timestamps,
        quantities=treated_qty,
        gap_types=hs.gap_types,
        first_record_at=hs.first_record_at,
        last_record_at=hs.last_record_at,
    )
