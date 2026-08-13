"""Tests for CDF bulk export."""

from __future__ import annotations

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from hospital_demand_analyzer.app.controller import AppController
from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.domain.models import ClusterAssignment, DistributionFitResult, HourlySeries
from hospital_demand_analyzer.exports.cdf_report import build_cdf_summary_df


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _poisson_fit() -> DistributionFitResult:
    return DistributionFitResult(
        name="Poisson",
        dist_type="discrete",
        params={"lambda": 2.0},
        log_likelihood=-50.0,
        n_params=1,
        aic=102.0,
        aicc=102.0,
        bic=105.0,
        converged=True,
        rank=1,
    )


def test_build_cdf_summary_df(qapp):
    state = AppState()
    ts = pd.date_range("2024-01-01", periods=10, freq="h")
    hs = HourlySeries(
        produto="100",
        descricao="Test",
        timestamps=ts,
        quantities=pd.Series([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0], index=ts),
    )
    state.hourly_series = {"100": hs}
    state.cluster_map = {
        "100": ClusterAssignment(
            produto="100", descricao="Test", cluster_id=2, cluster_name="Alto Giro",
            turnover_score=1.0, distance_to_centroid=0.0, mean_hourly_demand=1.0,
        ),
    }
    state.descriptive_df = pd.DataFrame([{
        "produto": "100",
        "descricao": "Test",
        "mean": 1.0,
    }])
    key = AppController._fit_cache_key("100", False, state.demand_value_mode)
    state.fit_results[key] = [_poisson_fit()]

    controller = AppController(state)
    df = build_cdf_summary_df(state, controller)
    assert len(df) == 1
    assert df.iloc[0]["CDF escolhida"] == "Poisson"
    assert "λ=" in str(df.iloc[0]["Parametros"])
    assert df.iloc[0]["Status"] == "Ajustado"
