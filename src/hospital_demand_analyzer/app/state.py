"""Global application state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QObject, Signal

from hospital_demand_analyzer.clustering.cutoffs import DEFAULT_CUTOFF_FIELD
from hospital_demand_analyzer.domain.models import (
    ClusterAssignment,
    ClusterResult,
    DistributionFitResult,
    HourlySeries,
    ImportResult,
    OutlierResult,
    ReplenishmentQminResult,
)
from hospital_demand_analyzer.domain.demand_values import DEFAULT_VALUE_MODE, normalize_value_mode
from hospital_demand_analyzer.utils.config import AppConfig


@dataclass
class SkuContext:
    produto: str = ""
    descricao: str = ""
    cluster_name: str = ""
    outlier_method: str = ""
    outlier_params: dict = field(default_factory=dict)


class AppState(QObject):
    data_loaded = Signal()
    sku_changed = Signal(str)
    cluster_changed = Signal(str)
    clusters_updated = Signal()
    outliers_changed = Signal()
    fit_completed = Signal(str)
    value_mode_changed = Signal(str)
    session_skus_changed = Signal()

    def __init__(self, config: AppConfig | None = None) -> None:
        super().__init__()
        self.config = config or AppConfig.load()
        self.data_dir: Path = self.config.data_path()
        self.import_result: ImportResult | None = None
        self.hourly_series: dict[str, HourlySeries] = {}
        self.treated_series: dict[str, HourlySeries] = {}
        self.descriptive_df: pd.DataFrame = pd.DataFrame()
        self.cluster_result: ClusterResult | None = None
        self.cluster_map: dict[str, ClusterAssignment] = {}
        self.fit_results: dict[str, list[DistributionFitResult]] = {}
        self.replenishment_qmin_results: dict[str, ReplenishmentQminResult] = {}
        self.outlier_results: dict[str, OutlierResult] = {}
        self.session_selected_skus: set[str] = set()
        self.session_sku_order: list[str] = []
        self.selected_sku: str = ""
        self.selected_cluster: str = "Todos"
        self.cluster_filter: str = "Todos"
        self.last_updated: datetime | None = None
        self.use_treated: bool = False
        self.cluster_alto_cutoff: float = self.config.cluster_alto_cutoff
        self.cluster_baixo_cutoff: float = self.config.cluster_baixo_cutoff
        self.cluster_cutoff_field: str = DEFAULT_CUTOFF_FIELD
        self.feature_df: pd.DataFrame = pd.DataFrame()
        self.clustering_method: str = self.config.clustering_method
        self.global_period_start: datetime | None = None
        self.global_period_end: datetime | None = None
        self.global_n_hours: int = 0
        self.demand_value_mode: str = normalize_value_mode(
            getattr(self.config, "demand_value_mode", DEFAULT_VALUE_MODE)
        )

    def set_demand_value_mode(self, mode: str) -> None:
        normalized = normalize_value_mode(mode)
        if self.demand_value_mode == normalized:
            return
        self.demand_value_mode = normalized
        self.value_mode_changed.emit(normalized)

    def get_active_series(self, produto: str | None = None) -> HourlySeries | None:
        sku = produto or self.selected_sku
        if not sku:
            return None
        if self.use_treated and sku in self.treated_series:
            return self.treated_series[sku]
        return self.hourly_series.get(sku)

    def get_cluster_for_sku(self, produto: str) -> str:
        a = self.cluster_map.get(produto)
        return a.cluster_name if a else ""

    def filtered_skus(self) -> list[str]:
        if self.cluster_filter == "Todos":
            return sorted(self.hourly_series.keys())
        return sorted(
            p for p, a in self.cluster_map.items()
            if a.cluster_name == self.cluster_filter and p in self.hourly_series
        )

    def processing_skus(self) -> list[str]:
        """SKUs in scope for batch CDF fit/export (respects cluster filter)."""
        return self.filtered_skus()
