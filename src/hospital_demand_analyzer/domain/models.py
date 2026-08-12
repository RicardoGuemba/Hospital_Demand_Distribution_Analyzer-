"""Domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd
import numpy as np


class DataGapType(str, Enum):
    ZERO_DEMAND = "zero_demand"
    MISSING_DATA = "missing_data"


@dataclass
class FileInfo:
    name: str
    path: str
    extension: str
    size_bytes: int
    modified: datetime


@dataclass
class FileQualityReport:
    file_name: str
    rows_total: int
    rows_valid: int
    rows_rejected: int
    columns: int
    skus: int
    date_min: datetime | None
    date_max: datetime | None
    missing_columns: list[str] = field(default_factory=list)
    invalid_dates: int = 0
    invalid_quantities: int = 0
    null_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    status: str = "ok"

    @property
    def error_count(self) -> int:
        return len(self.errors) + self.invalid_dates + self.invalid_quantities


@dataclass
class GlobalInconsistencies:
    missing_columns: list[str] = field(default_factory=list)
    invalid_dates: int = 0
    invalid_quantities: int = 0
    skus_without_description: list[str] = field(default_factory=list)
    divergent_descriptions: dict[str, list[str]] = field(default_factory=dict)
    duplicate_rows: int = 0


@dataclass
class ImportResult:
    normalized_df: pd.DataFrame
    file_reports: list[FileQualityReport]
    inconsistencies: GlobalInconsistencies
    data_dir: str
    imported_at: datetime = field(default_factory=datetime.now)

    @property
    def total_files(self) -> int:
        return len(self.file_reports)

    @property
    def total_valid_rows(self) -> int:
        return sum(r.rows_valid for r in self.file_reports)

    @property
    def total_rejected(self) -> int:
        return sum(r.rows_rejected for r in self.file_reports)

    @property
    def unique_skus(self) -> int:
        if self.normalized_df.empty:
            return 0
        return int(self.normalized_df["produto"].nunique())

    @property
    def date_min(self) -> datetime | None:
        if self.normalized_df.empty:
            return None
        return pd.Timestamp(self.normalized_df["data_transf"].min()).to_pydatetime()

    @property
    def date_max(self) -> datetime | None:
        if self.normalized_df.empty:
            return None
        return pd.Timestamp(self.normalized_df["data_transf"].max()).to_pydatetime()


@dataclass
class HourlySeries:
    produto: str
    descricao: str
    timestamps: pd.DatetimeIndex
    quantities: pd.Series
    gap_types: pd.Series | None = None
    first_record_at: datetime | None = None
    last_record_at: datetime | None = None

    @property
    def period_start(self) -> datetime | None:
        if len(self.timestamps) == 0:
            return None
        return self.timestamps.min().to_pydatetime()

    @property
    def period_end(self) -> datetime | None:
        if len(self.timestamps) == 0:
            return None
        return self.timestamps.max().to_pydatetime()

    @property
    def n_hours(self) -> int:
        return len(self.quantities)

    @property
    def total_demand(self) -> float:
        return float(self.quantities.sum())

    @property
    def n_observations(self) -> int:
        return len(self.quantities)

    @property
    def zero_ratio(self) -> float:
        if self.n_observations == 0:
            return 0.0
        return float((self.quantities == 0).sum()) / self.n_observations


@dataclass
class DescriptiveStats:
    produto: str
    descricao: str
    n: int
    n_zeros: int
    zero_ratio: float
    sum_: float
    mean: float
    expected_empirical: float
    expected_conditional: float
    median: float
    std: float
    var: float
    cv: float
    min_: float
    max_: float
    q25: float
    q50: float
    q75: float
    iqr: float
    p90: float
    p95: float
    p975: float
    p99: float
    skewness: float
    kurtosis: float
    date_start: datetime | None = None
    date_end: datetime | None = None
    non_zero_hours: int = 0


@dataclass
class DistributionFitResult:
    name: str
    dist_type: str  # discrete / continuous
    params: dict[str, float]
    log_likelihood: float
    n_params: int
    aic: float
    aicc: float
    bic: float
    delta_aic: float = 0.0
    delta_bic: float = 0.0
    ks_stat: float | None = None
    ks_pvalue: float | None = None
    cvm_stat: float | None = None
    ad_stat: float | None = None
    tail_fit_score: float | None = None
    tail_errors: dict[str, float] = field(default_factory=dict)
    converged: bool = True
    warnings: list[str] = field(default_factory=list)
    fit_time_ms: float = 0.0
    rank: int = 0
    rank_tail: int = 0


@dataclass
class MonteCarloTrajectory:
    """Single simulated inventory path until Qmin is reached."""

    times: np.ndarray
    balances: np.ndarray
    temp_qmin: float


@dataclass
class TempQMinResult:
    """Aggregated Monte Carlo TempQMin simulation output."""

    produto: str
    saldo_inicial: float
    qmin: float
    n_trajetorias: int
    delta_t_minutos: float
    distribuicao: str
    parametros: dict[str, float]
    trajectories: list[MonteCarloTrajectory]
    temp_qmin_times: np.ndarray
    mean: float
    median: float
    p5: float
    p10: float
    p90: float
    p95: float
    std: float
    min_: float
    max_: float


@dataclass
class OutlierResult:
    method: str
    params: dict[str, Any]
    mask: pd.Series
    scores: pd.Series
    reasons: pd.Series

    @property
    def n_outliers(self) -> int:
        return int(self.mask.sum())

    @property
    def outlier_ratio(self) -> float:
        if len(self.mask) == 0:
            return 0.0
        return self.n_outliers / len(self.mask)


@dataclass
class ClusterAssignment:
    produto: str
    descricao: str
    cluster_id: int
    cluster_name: str
    turnover_score: float
    distance_to_centroid: float
    mean_hourly_demand: float
    cutoff_metric_value: float = 0.0
    non_zero_hour_ratio: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    cv: float = 0.0
    observation_hours: int = 0


@dataclass
class ClusterResult:
    assignments: list[ClusterAssignment]
    centroids: pd.DataFrame
    features_used: list[str]
    scaler_name: str
    inertia: float
    silhouette: float
    random_state: int
    n_clusters: int = 3
    cluster_summaries: dict[str, dict[str, float]] = field(default_factory=dict)
    method: str = "threshold"
    alto_cutoff: float | None = None
    baixo_cutoff: float | None = None
    cutoff_field: str = "mean_hourly_demand"
