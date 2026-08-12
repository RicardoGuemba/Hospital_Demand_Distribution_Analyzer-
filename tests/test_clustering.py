"""Tests for threshold and K-Means clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hospital_demand_analyzer.clustering.features import build_feature_matrix, compute_features
from hospital_demand_analyzer.clustering.kmeans_service import run_kmeans
from hospital_demand_analyzer.clustering.cutoffs import assign_cluster_name
from hospital_demand_analyzer.clustering.threshold_clustering import run_threshold_clustering
from hospital_demand_analyzer.domain.models import HourlySeries
from hospital_demand_analyzer.services.time_series import build_hourly_series


def _make_synthetic_series():
    series = {}
    configs = [
        ("low1", 0.05, 1), ("low2", 0.08, 2), ("low3", 0.03, 1),
        ("med1", 0.3, 10), ("med2", 0.25, 8), ("med3", 0.35, 12),
        ("high1", 0.8, 50), ("high2", 0.9, 60), ("high3", 0.7, 45),
    ]
    for name, freq, scale in configs:
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        vals = np.random.poisson(scale, n).astype(float)
        mask = np.random.random(n) > freq
        vals[mask] = 0
        series[name] = HourlySeries(name, name, idx, pd.Series(vals, index=idx))
    return series


def test_features():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    hs = HourlySeries("1", "T", idx, pd.Series([0, 1, 0, 2, 0, 3, 0, 4, 0, 5], index=idx))
    feat = compute_features(hs)
    assert feat["mean_hourly_demand"] == pytest.approx(1.5)
    assert feat["non_zero_hour_ratio"] == pytest.approx(0.5)


def test_kmeans_reproducible():
    series = _make_synthetic_series()
    feat_df = build_feature_matrix(series)
    r1 = run_kmeans(feat_df, random_state=42)
    r2 = run_kmeans(feat_df, random_state=42)
    ids1 = sorted(a.cluster_id for a in r1.assignments)
    ids2 = sorted(a.cluster_id for a in r2.assignments)
    assert ids1 == ids2


def test_kmeans_three_clusters():
    series = _make_synthetic_series()
    feat_df = build_feature_matrix(series)
    result = run_kmeans(feat_df, random_state=42)
    assert result.n_clusters == 3
    names = {a.cluster_name for a in result.assignments}
    assert "Alto Giro" in names
    assert "Baixo Giro" in names


def test_assign_cluster_name():
    assert assign_cluster_name(3.0, alto_cutoff=2.0, baixo_cutoff=0.5) == "Alto Giro"
    assert assign_cluster_name(0.2, alto_cutoff=2.0, baixo_cutoff=0.5) == "Baixo Giro"
    assert assign_cluster_name(1.0, alto_cutoff=2.0, baixo_cutoff=0.5) == "Medio Giro"


def test_threshold_clustering_alto_cutoff():
    series = _make_synthetic_series()
    feat_df = build_feature_matrix(series)
    result = run_threshold_clustering(feat_df, alto_cutoff=2.0, baixo_cutoff=0.5)
    assert result.method == "threshold"
    for a in result.assignments:
        if a.cluster_name == "Alto Giro":
            assert a.mean_hourly_demand >= 2.0
        elif a.cluster_name == "Baixo Giro":
            assert a.mean_hourly_demand < 0.5


def test_threshold_cutoff_expands_alto():
    series = _make_synthetic_series()
    feat_df = build_feature_matrix(series)
    strict = run_threshold_clustering(feat_df, alto_cutoff=50.0, baixo_cutoff=0.5)
    relaxed = run_threshold_clustering(feat_df, alto_cutoff=2.0, baixo_cutoff=0.5)
    n_strict = strict.cluster_summaries.get("Alto Giro", {}).get("n_skus", 0)
    n_relaxed = relaxed.cluster_summaries.get("Alto Giro", {}).get("n_skus", 0)
    assert n_relaxed >= n_strict


def test_threshold_invalid_cutoffs():
    feat_df = pd.DataFrame({"produto": ["1"], "mean_hourly_demand": [1.0]})
    with pytest.raises(ValueError):
        run_threshold_clustering(feat_df, alto_cutoff=1.0, baixo_cutoff=2.0)


def test_threshold_clustering_respects_cutoff_field():
    feat_df = pd.DataFrame({
        "produto": ["a", "b", "c"],
        "mean_hourly_demand": [0.1, 0.5, 2.0],
        "demand_per_observed_hour": [1.0, 5.0, 20.0],
        "non_zero_hour_ratio": [0.1, 0.3, 0.8],
        "p95_hourly_demand": [1.0, 6.0, 25.0],
        "coefficient_of_variation": [1.0, 1.2, 0.8],
    })
    result = run_threshold_clustering(
        feat_df, alto_cutoff=10.0, baixo_cutoff=3.0, cutoff_field="demand_per_observed_hour",
    )
    by_name = {a.produto: a.cluster_name for a in result.assignments}
    assert by_name["a"] == "Baixo Giro"
    assert by_name["b"] == "Medio Giro"
    assert by_name["c"] == "Alto Giro"


def test_alto_cutoff_monotonically_changes_alto_skus():
    from hospital_demand_analyzer.clustering.cutoffs import assign_skus_by_cutoff

    series = _make_synthetic_series()
    feat_df = build_feature_matrix(series)
    baixo = 0.5
    prev_alto_skus: set[str] | None = None
    for alto in [0.8, 1.5, 3.0, 10.0, 50.0]:
        mapping = assign_skus_by_cutoff(feat_df, alto, baixo)
        alto_skus = {p for p, c in mapping.items() if c == "Alto Giro"}
        for produto, cluster in mapping.items():
            value = float(feat_df.loc[feat_df["produto"] == produto, "mean_hourly_demand"].iloc[0])
            if cluster == "Alto Giro":
                assert value >= alto
            elif cluster == "Baixo Giro":
                assert value < baixo
            else:
                assert baixo <= value < alto
        if prev_alto_skus is not None:
            assert alto_skus.issubset(prev_alto_skus)
        prev_alto_skus = alto_skus


def test_suggest_and_preview_cutoffs():
    from hospital_demand_analyzer.clustering.cutoffs import preview_cluster_counts, suggest_threshold_cutoffs

    feat_df = pd.DataFrame({
        "produto": ["a", "b", "c", "d", "e"],
        "mean_hourly_demand": [0.01, 0.05, 0.2, 0.8, 3.0],
    })
    baixo, alto = suggest_threshold_cutoffs(feat_df)
    assert baixo < alto
    counts = preview_cluster_counts(feat_df, alto, baixo)
    assert sum(counts.values()) == 5
