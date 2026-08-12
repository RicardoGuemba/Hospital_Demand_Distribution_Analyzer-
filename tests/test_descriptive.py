"""Tests for descriptive statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hospital_demand_analyzer.domain.models import HourlySeries
from hospital_demand_analyzer.statistics.descriptive import (
    DESCRIPTIVE_TABLE_COLUMNS,
    compute_all_descriptive,
    compute_descriptive,
    descriptive_to_dict,
    empirical_pmf_expected,
)


def _sample_series(values: list[float], produto: str = "SKU1") -> HourlySeries:
    ts = pd.date_range("2024-01-01", periods=len(values), freq="h")
    return HourlySeries(
        produto=produto,
        descricao="Test item",
        timestamps=ts,
        quantities=pd.Series(values, index=ts),
    )


def test_compute_descriptive_classic_metrics():
    hs = _sample_series([0.0, 1.0, 2.0, 3.0, 4.0])
    ds = compute_descriptive(hs)
    assert ds.n == 5
    assert ds.n_zeros == 1
    assert ds.non_zero_hours == 4
    assert ds.mean == 2.0
    assert ds.expected_empirical == 2.0
    assert ds.expected_conditional == 2.5
    assert ds.median == 2.0
    assert ds.min_ == 0.0
    assert ds.max_ == 4.0
    assert ds.q25 <= ds.q50 <= ds.q75
    assert ds.iqr == ds.q75 - ds.q25


def test_empirical_pmf_expected_matches_mean():
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert empirical_pmf_expected(values) == np.mean(values)


def test_descriptive_to_dict_has_all_table_keys():
    hs = _sample_series([1.0, 2.0, 3.0])
    row = descriptive_to_dict(compute_descriptive(hs))
    for key, _label in DESCRIPTIVE_TABLE_COLUMNS:
        if key == "cluster":
            continue
        assert key in row


def test_compute_all_descriptive_dataframe():
    series = {
        "A": _sample_series([0.0, 2.0, 4.0], "A"),
        "B": _sample_series([1.0, 1.0, 1.0], "B"),
    }
    df = compute_all_descriptive(series)
    assert len(df) == 2
    assert "expected_empirical" in df.columns
    assert "expected_conditional" in df.columns
    assert "skewness" in df.columns
    assert "kurtosis" in df.columns
    assert "var" in df.columns
    assert df.loc[df["produto"] == "B", "std"].iloc[0] == 0.0


def test_expected_equals_mean_when_no_zeros():
    hs = _sample_series([2.0, 4.0, 6.0])
    ds = compute_descriptive(hs)
    assert ds.mean == ds.expected_empirical == ds.expected_conditional == 4.0


def test_expected_zero_when_all_zeros():
    hs = _sample_series([0.0, 0.0, 0.0])
    ds = compute_descriptive(hs)
    assert ds.mean == 0.0
    assert ds.expected_empirical == 0.0
    assert ds.expected_conditional == 0.0


def test_compute_descriptive_empty_series():
    hs = _sample_series([])
    ds = compute_descriptive(hs)
    assert ds.n == 0
    assert ds.mean == 0.0
