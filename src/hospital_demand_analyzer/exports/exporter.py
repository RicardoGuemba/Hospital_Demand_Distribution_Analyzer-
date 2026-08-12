"""Export service."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hospital_demand_analyzer.domain.models import ClusterAssignment, DistributionFitResult


def export_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def export_excel(df: pd.DataFrame, path: Path, sheet_name: str = "Dados") -> None:
    df.to_excel(path, index=False, sheet_name=sheet_name)


def export_time_series(long_df: pd.DataFrame, path: Path, fmt: str = "csv") -> None:
    if fmt == "xlsx":
        export_excel(long_df, path, "SerieTemporal")
    else:
        export_csv(long_df, path)


def export_cluster_results(assignments: list[ClusterAssignment], path: Path, fmt: str = "csv") -> None:
    rows = [{
        "Produto": a.produto,
        "Descricao": a.descricao,
        "Cluster": a.cluster_name,
        "Turnover Score": a.turnover_score,
        "Media/h": a.mean_hourly_demand,
        "Frequencia": a.non_zero_hour_ratio,
        "P95": a.p95,
        "P99": a.p99,
        "CV": a.cv,
        "Distancia Centroide": a.distance_to_centroid,
    } for a in assignments]
    df = pd.DataFrame(rows)
    if fmt == "xlsx":
        export_excel(df, path, "Clusters")
    else:
        export_csv(df, path)


def export_fit_ranking(results: list[DistributionFitResult], path: Path, fmt: str = "csv") -> None:
    rows = [{
        "Rank": r.rank,
        "Distribuicao": r.name,
        "Tipo": r.dist_type,
        "AIC": r.aic,
        "Delta AIC": r.delta_aic,
        "BIC": r.bic,
        "Delta BIC": r.delta_bic,
        "KS": r.ks_stat,
        "CvM": r.cvm_stat,
        "AD": r.ad_stat,
        "Tail Score": r.tail_fit_score,
        "Convergiu": r.converged,
    } for r in results if r.converged]
    df = pd.DataFrame(rows)
    if fmt == "xlsx":
        export_excel(df, path, "Ranking")
    else:
        export_csv(df, path)
