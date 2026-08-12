"""Tests for data import."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from hospital_demand_analyzer.data.importer import import_directory
from hospital_demand_analyzer.data.normalizer import normalize_columns
from hospital_demand_analyzer.data.validator import validate_file_df


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Data Transf.": ["2024-01-15 10:05:00", "2024-01-15 10:30:00", "2024-01-15 12:00:00"],
        "Produto": ["123", "123", "456"],
        "Descrição Produto": ["Item A", "Item A", "Item B"],
        "Quantidade": ["5", "3", "10"],
        "Status": ["Ativo", "Ativo", "Ativo"],
    })


def test_normalize_columns(sample_df):
    norm, mapping = normalize_columns(sample_df)
    assert "data_transf" in norm.columns
    assert "produto" in norm.columns
    assert "descricao_produto" in norm.columns


def test_validate_valid(sample_df):
    norm, _ = normalize_columns(sample_df)
    valid, report, _ = validate_file_df(norm, "test.xlsx")
    assert report.rows_valid == 3
    assert report.status == "ok"
    assert len(valid) == 3


def test_validate_missing_column():
    df = pd.DataFrame({"Produto": ["1"], "Quantidade": ["5"]})
    norm, _ = normalize_columns(df)
    valid, report, inc = validate_file_df(norm, "bad.xlsx")
    assert "data_transf" in report.missing_columns
    assert len(valid) == 0


def test_validate_invalid_date(sample_df):
    df = sample_df.copy()
    df.loc[0, "Data Transf."] = "invalid"
    norm, _ = normalize_columns(df)
    valid, report, _ = validate_file_df(norm, "test.xlsx")
    assert report.invalid_dates >= 1
    assert report.rows_rejected >= 1


def test_import_real_data():
    data_dir = Path(__file__).resolve().parents[1] / "Dados_Demanda"
    if not data_dir.exists():
        pytest.skip("Real data not available")
    result = import_directory(data_dir)
    assert result.total_files >= 1
    assert result.total_valid_rows > 0
    assert result.unique_skus > 0
