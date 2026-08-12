"""Tests for outlier detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hospital_demand_analyzer.domain.models import HourlySeries
from hospital_demand_analyzer.outliers.engine import apply_outlier_method, apply_treatment
from hospital_demand_analyzer.outliers.hampel import detect_hampel
from hospital_demand_analyzer.outliers.iqr import detect_iqr
from hospital_demand_analyzer.outliers.mad import detect_mad
from hospital_demand_analyzer.outliers.zscore import detect_zscore


def _series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="h")
    return HourlySeries("123", "Test", idx, pd.Series(values, index=idx))


def test_iqr_detects_extreme_spike():
    values = [1, 2, 2, 3, 2, 1, 2, 100]
    result = detect_iqr(pd.Series(values))
    assert result.n_outliers >= 1
    assert result.mask.iloc[-1]
    assert result.params["multiplier"] == 1.5


def test_iqr_does_not_flag_zeros_in_zero_inflated_series():
    """Many zero hours + moderate demand should not mark all demand as outlier."""
    values = [0.0] * 200 + [1.0, 2.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0]
    result = detect_iqr(pd.Series(values))
    assert result.n_outliers == 0
    assert not result.mask[:200].any()


def test_iqr_flags_spike_among_zeros():
    values = [0.0] * 200 + [1.0, 2.0, 2.0, 3.0, 2.0, 1.0, 2.0, 500.0]
    result = detect_iqr(pd.Series(values))
    assert result.n_outliers >= 1
    assert result.mask.iloc[-1]


def test_iqr_tukey_fences_stored():
    values = [1, 2, 2, 3, 2, 1, 2, 100]
    result = detect_iqr(pd.Series(values), multiplier=1.5)
    assert "lower" in result.params
    assert "upper" in result.params
    assert result.params["upper"] < 100


def test_zscore_detects_outlier():
    values = [2.0] * 30 + [100.0]
    result = detect_zscore(pd.Series(values))
    assert result.n_outliers >= 1
    assert result.mask.iloc[-1]


def test_zscore_ignores_zeros():
    values = [0.0] * 50 + [2.0, 2.0, 3.0, 2.0, 2.0]
    result = detect_zscore(pd.Series(values), threshold=3.0)
    assert result.n_outliers == 0


def test_mad_detects_outlier():
    values = [1, 2, 2, 3, 2, 1, 2, 100]
    result = detect_mad(pd.Series(values))
    assert result.n_outliers >= 1


def test_mad_does_not_flag_all_positive_when_many_zeros():
    values = [0.0] * 100 + [1.0, 2.0, 2.0, 3.0, 2.0]
    result = detect_mad(pd.Series(values))
    assert result.n_outliers < 5
    assert not result.mask[:100].any()


def test_hampel_detects_outlier():
    values = [1, 2, 2, 3, 2, 1, 2, 100]
    result = detect_hampel(pd.Series(values))
    assert result.n_outliers >= 1


def test_engine_default_iqr_multiplier():
    hs = _series([0.0] * 20 + [1.0, 2.0, 2.0, 3.0, 50.0])
    result = apply_outlier_method(hs, "IQR")
    assert result.params["multiplier"] == 1.5


def test_engine_zscore_method():
    hs = _series([1, 2, 2, 3, 2, 1, 2, 100])
    result = apply_outlier_method(hs, "Z-Score")
    assert result.method == "Z-Score"


def test_treatment_winsorizes_to_upper_fence():
    hs = _series([0.0] * 20 + [1.0, 2.0, 2.0, 3.0, 50.0])
    outlier = apply_outlier_method(hs, "IQR")
    treated = apply_treatment(hs, outlier)
    if outlier.n_outliers > 0:
        assert treated.quantities.max() < hs.quantities.max()
        assert treated.quantities.max() <= outlier.params["upper"] + 1e-9


def test_treatment_reversible_max_reduced():
    hs = _series([1, 2, 2, 3, 2, 1, 2, 100])
    outlier = apply_outlier_method(hs, "IQR")
    treated = apply_treatment(hs, outlier)
    assert treated.quantities.max() <= hs.quantities.max()
