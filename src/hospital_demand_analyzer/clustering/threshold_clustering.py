"""Threshold-based turnover clustering by hourly demand cutoffs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from hospital_demand_analyzer.clustering.cutoffs import DEFAULT_CUTOFF_FIELD, assign_cluster_name
from hospital_demand_analyzer.clustering.kmeans_service import CLUSTER_NAMES, turnover_score
from hospital_demand_analyzer.domain.models import ClusterAssignment, ClusterResult

CLUSTER_ID = {"Baixo Giro": 0, "Medio Giro": 1, "Alto Giro": 2}


def run_threshold_clustering(
    feature_df: pd.DataFrame,
    alto_cutoff: float = 2.0,
    baixo_cutoff: float = 0.5,
    cutoff_field: str = DEFAULT_CUTOFF_FIELD,
) -> ClusterResult:
    """Segment SKUs strictly by demand/h cutoffs on ``cutoff_field``."""
    if baixo_cutoff >= alto_cutoff:
        raise ValueError(
            f"Corte Baixo Giro ({baixo_cutoff}) deve ser menor que Corte Alto Giro ({alto_cutoff})."
        )

    if cutoff_field not in feature_df.columns:
        cutoff_field = DEFAULT_CUTOFF_FIELD

    work = feature_df.dropna(subset=[cutoff_field]).copy()
    if work.empty:
        raise ValueError("Nenhum SKU com features válidas para clustering.")

    work["_cutoff_metric"] = work[cutoff_field].astype(float)
    work["cluster_name"] = work["_cutoff_metric"].apply(
        lambda v: assign_cluster_name(float(v), alto_cutoff, baixo_cutoff)
    )
    work["cluster_id"] = work["cluster_name"].map(CLUSTER_ID)
    work["turnover_score"] = work.apply(turnover_score, axis=1)

    feature_cols = [
        c for c in [
            "mean_hourly_demand", "non_zero_hour_ratio", "p95_hourly_demand",
            "coefficient_of_variation",
        ] if c in work.columns
    ]

    centroids = work.groupby("cluster_name")[feature_cols].mean().reset_index()
    centroids["cluster_id"] = centroids["cluster_name"].map(CLUSTER_ID)
    centroids["turnover_score"] = centroids.apply(turnover_score, axis=1)

    centroid_lookup = {
        row["cluster_name"]: row[feature_cols].values.astype(float)
        for _, row in centroids.iterrows()
    }

    assignments: list[ClusterAssignment] = []
    for _, row in work.iterrows():
        cname = str(row["cluster_name"])
        feat_vec = row[feature_cols].values.astype(float)
        center = centroid_lookup.get(cname, feat_vec)
        dist = float(np.linalg.norm(feat_vec - center))
        cutoff_val = float(row["_cutoff_metric"])

        assignments.append(
            ClusterAssignment(
                produto=str(row["produto"]),
                descricao=str(row.get("descricao", "")),
                cluster_id=int(row["cluster_id"]),
                cluster_name=cname,
                turnover_score=float(row["turnover_score"]),
                distance_to_centroid=dist,
                mean_hourly_demand=float(row.get("mean_hourly_demand", 0)),
                cutoff_metric_value=cutoff_val,
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
        summaries[name] = {
            "n_skus": len(subset),
            "pct": len(subset) / len(work) * 100 if len(work) else 0,
            "mean_demand": float(subset["_cutoff_metric"].mean()) if len(subset) else 0,
            "mean_frequency": float(subset["non_zero_hour_ratio"].mean()) if len(subset) else 0,
            "mean_p95": float(subset["p95_hourly_demand"].mean()) if len(subset) else 0,
        }

    labels = work["cluster_id"].values
    sil = 0.0
    if len(set(labels)) > 1 and len(work) > len(set(labels)):
        sil_feats = work[["_cutoff_metric", "non_zero_hour_ratio"]].values.astype(float)
        sil = float(silhouette_score(sil_feats, labels))

    inertia = 0.0
    for _, row in work.iterrows():
        cname = str(row["cluster_name"])
        feat_vec = row[feature_cols].values.astype(float)
        center = centroid_lookup.get(cname, feat_vec)
        inertia += float(np.sum((feat_vec - center) ** 2))

    return ClusterResult(
        assignments=assignments,
        centroids=centroids,
        features_used=feature_cols,
        scaler_name="n/a",
        inertia=inertia,
        silhouette=sil,
        random_state=0,
        n_clusters=3,
        cluster_summaries=summaries,
        method="threshold",
        alto_cutoff=alto_cutoff,
        baixo_cutoff=baixo_cutoff,
        cutoff_field=cutoff_field,
    )
