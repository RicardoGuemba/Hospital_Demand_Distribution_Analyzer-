"""K-Means clustering for turnover segmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

from hospital_demand_analyzer.domain.models import ClusterAssignment, ClusterResult

CLUSTER_NAMES = ["Baixo Giro", "Medio Giro", "Alto Giro"]


def turnover_score(row: pd.Series) -> float:
    """Weighted index for cluster ordering. See docs/TURNOVER_SCORE.md."""
    return (
        0.4 * row.get("mean_hourly_demand", 0)
        + 0.35 * row.get("non_zero_hour_ratio", 0)
        + 0.25 * row.get("p95_hourly_demand", 0)
    )


def run_kmeans(
    feature_df: pd.DataFrame,
    features: list[str] | None = None,
    scaler_name: str = "robust",
    n_clusters: int = 3,
    random_state: int = 42,
) -> ClusterResult:
    from hospital_demand_analyzer.clustering.features import DEFAULT_FEATURES

    features = features or DEFAULT_FEATURES
    work = feature_df.dropna(subset=features).copy()
    if len(work) < n_clusters:
        raise ValueError(f"Need at least {n_clusters} SKUs for clustering, got {len(work)}")

    X = work[features].values.astype(float)
    scaler = RobustScaler() if scaler_name == "robust" else StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    work["cluster_id_raw"] = labels

    # Compute turnover score and reorder clusters
    work["turnover_score"] = work.apply(turnover_score, axis=1)
    cluster_means = work.groupby("cluster_id_raw")["turnover_score"].mean().sort_values()
    id_map = {old: new for new, old in enumerate(cluster_means.index)}
    name_map = {new: CLUSTER_NAMES[new] for new in range(n_clusters)}
    work["cluster_id"] = work["cluster_id_raw"].map(id_map)
    work["cluster_name"] = work["cluster_id"].map(name_map)

    centroids_raw = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=features,
    )
    centroids_raw["cluster_id_raw"] = range(n_clusters)
    centroids_raw["cluster_id"] = centroids_raw["cluster_id_raw"].map(id_map)
    centroids_raw["cluster_name"] = centroids_raw["cluster_id"].map(name_map)
    centroids_raw["turnover_score"] = centroids_raw.apply(turnover_score, axis=1)

    distances = np.linalg.norm(X_scaled - km.cluster_centers_[labels], axis=1)

    sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0

    assignments = []
    for i, (_, row) in enumerate(work.iterrows()):
        assignments.append(
            ClusterAssignment(
                produto=str(row["produto"]),
                descricao=str(row.get("descricao", "")),
                cluster_id=int(row["cluster_id"]),
                cluster_name=str(row["cluster_name"]),
                turnover_score=float(row["turnover_score"]),
                distance_to_centroid=float(distances[i]),
                mean_hourly_demand=float(row.get("mean_hourly_demand", 0)),
                cutoff_metric_value=float(row.get("mean_hourly_demand", 0)),
                non_zero_hour_ratio=float(row.get("non_zero_hour_ratio", 0)),
                p95=float(row.get("p95_hourly_demand", 0)),
                p99=float(row.get("p99_hourly_demand", 0)),
                cv=float(row.get("coefficient_of_variation", 0)),
                observation_hours=int(row.get("observation_hours", 0)),
            )
        )

    summaries: dict[str, dict[str, float]] = {}
    for name in CLUSTER_NAMES:
        subset = work[work["cluster_name"] == name]
        if len(subset) == 0:
            continue
        summaries[name] = {
            "n_skus": len(subset),
            "pct": len(subset) / len(work) * 100,
            "mean_demand": float(subset["mean_hourly_demand"].mean()),
            "mean_frequency": float(subset["non_zero_hour_ratio"].mean()),
            "mean_p95": float(subset["p95_hourly_demand"].mean()),
        }

    return ClusterResult(
        assignments=assignments,
        centroids=centroids_raw,
        features_used=features,
        scaler_name=scaler_name,
        inertia=float(km.inertia_),
        silhouette=float(sil),
        random_state=random_state,
        n_clusters=n_clusters,
        cluster_summaries=summaries,
        method="kmeans",
    )
