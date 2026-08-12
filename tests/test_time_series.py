"""Tests for hourly time series."""

from __future__ import annotations

import pandas as pd
import pytest

from hospital_demand_analyzer.services.time_series import build_hourly_series


def _make_df(rows):
    return pd.DataFrame(rows)


def test_sku_same_hour_two_files():
    df = _make_df([
        {"produto": "123", "descricao_produto": "A", "descricao_canonica": "A",
         "data_transf": pd.Timestamp("2024-01-01 10:05"), "quantidade": 5, "source_file": "A.xlsx"},
        {"produto": "123", "descricao_produto": "A", "descricao_canonica": "A",
         "data_transf": pd.Timestamp("2024-01-01 10:30"), "quantidade": 3, "source_file": "B.xlsx"},
    ])
    hs = build_hourly_series(df, "123")
    assert hs.quantities.loc[pd.Timestamp("2024-01-01 10:00")] == 8


def test_zero_fill_between_hours():
    df = _make_df([
        {"produto": "123", "descricao_produto": "A", "descricao_canonica": "A",
         "data_transf": pd.Timestamp("2024-01-01 10:00"), "quantidade": 5, "source_file": "A.xlsx"},
        {"produto": "123", "descricao_produto": "A", "descricao_canonica": "A",
         "data_transf": pd.Timestamp("2024-01-01 12:00"), "quantidade": 3, "source_file": "A.xlsx"},
    ])
    hs = build_hourly_series(df, "123")
    assert hs.quantities.loc[pd.Timestamp("2024-01-01 10:00")] == 5
    assert hs.quantities.loc[pd.Timestamp("2024-01-01 11:00")] == 0
    assert hs.quantities.loc[pd.Timestamp("2024-01-01 12:00")] == 3


def test_normalized_period_first_last_record():
    """All SKUs share global grid from dataset first/last record."""
    df = _make_df([
        {"produto": "123", "descricao_produto": "A", "descricao_canonica": "A",
         "data_transf": pd.Timestamp("2024-01-01 10:05"), "quantidade": 5, "source_file": "A.xlsx"},
        {"produto": "123", "descricao_produto": "A", "descricao_canonica": "A",
         "data_transf": pd.Timestamp("2024-01-01 14:45"), "quantidade": 3, "source_file": "A.xlsx"},
        {"produto": "456", "descricao_produto": "B", "descricao_canonica": "B",
         "data_transf": pd.Timestamp("2024-01-01 08:00"), "quantidade": 1, "source_file": "A.xlsx"},
    ])
    from hospital_demand_analyzer.services.time_series import build_all_hourly_series
    all_series = build_all_hourly_series(df)
    hs = all_series["123"]
    hs2 = all_series["456"]
    assert hs.n_hours == hs2.n_hours
    assert hs.period_start == pd.Timestamp("2024-01-01 08:00")
    assert hs.period_end == pd.Timestamp("2024-01-01 14:00")
    assert hs.quantities.loc[pd.Timestamp("2024-01-01 11:00")] == 0
    assert hs.quantities.sum() == 8
    assert hs2.quantities.loc[pd.Timestamp("2024-01-01 10:00")] == 0


def test_statistics_include_zero_hours():
    df = _make_df([
        {"produto": "123", "descricao_produto": "A", "descricao_canonica": "A",
         "data_transf": pd.Timestamp("2024-01-01 10:00"), "quantidade": 10, "source_file": "A.xlsx"},
        {"produto": "123", "descricao_produto": "A", "descricao_canonica": "A",
         "data_transf": pd.Timestamp("2024-01-01 12:00"), "quantidade": 10, "source_file": "A.xlsx"},
        {"produto": "456", "descricao_produto": "B", "descricao_canonica": "B",
         "data_transf": pd.Timestamp("2024-01-01 08:00"), "quantidade": 1, "source_file": "A.xlsx"},
    ])
    from hospital_demand_analyzer.services.time_series import build_all_hourly_series
    hs = build_all_hourly_series(df)["123"]
    from hospital_demand_analyzer.statistics.descriptive import compute_descriptive
    ds = compute_descriptive(hs)
    assert ds.n == 5
    assert ds.n_zeros == 3
    assert ds.mean == pytest.approx(20 / 5)
