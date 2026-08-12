"""Configuration loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hospital_demand_analyzer.utils.paths import project_root


@dataclass
class OutlierDefaults:
    iqr_multiplier: float = 1.5
    mad_threshold: float = 3.5
    hampel_window: int = 5
    hampel_n_sigmas: float = 3.0


@dataclass
class AppConfig:
    data_dir: str = "Dados_Demanda"
    cache_dir: str = "cache"
    logs_dir: str = "logs"
    random_state: int = 42
    kmeans_n_clusters: int = 3
    default_scaler: str = "robust"
    cluster_alto_cutoff: float = 2.0
    cluster_baixo_cutoff: float = 0.5
    clustering_method: str = "threshold"
    demand_value_mode: str = "discrete"
    outlier_defaults: OutlierDefaults = field(default_factory=OutlierDefaults)

    @classmethod
    def load(cls, path: Path | None = None) -> AppConfig:
        config_path = path or project_root() / "config" / "default.json"
        if not config_path.exists():
            return cls()
        with config_path.open(encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)
        od = raw.pop("outlier_defaults", {})
        cfg = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        cfg.outlier_defaults = OutlierDefaults(**od)
        return cfg

    def data_path(self) -> Path:
        return project_root() / self.data_dir

    def cache_path(self) -> Path:
        p = project_root() / self.cache_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def logs_path(self) -> Path:
        p = project_root() / self.logs_dir
        p.mkdir(parents=True, exist_ok=True)
        return p
