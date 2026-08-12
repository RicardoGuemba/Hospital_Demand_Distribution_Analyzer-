"""Demand value preparation and analysis mode (discrete vs continuous)."""

from __future__ import annotations

import numpy as np

DISCRETE = "discrete"
CONTINUOUS = "continuous"
DEFAULT_VALUE_MODE = DISCRETE

VALUE_MODE_LABELS: dict[str, str] = {
    DISCRETE: "Discreto (demanda ≥ 0)",
    CONTINUOUS: "Contínuo (demanda ≥ 0)",
}


def normalize_value_mode(mode: str | None) -> str:
    if mode == CONTINUOUS:
        return CONTINUOUS
    return DISCRETE


def sanitize_demand(values: np.ndarray) -> np.ndarray:
    """Keep only valid demand observations (demanda ≥ 0)."""
    arr = np.asarray(values, dtype=float)
    return arr[arr >= 0]


def prepare_demand_values(values: np.ndarray, *, include_zeros: bool = True) -> np.ndarray:
    """Sanitize to ≥ 0; optionally restrict to demand > 0 only."""
    arr = sanitize_demand(values)
    if not include_zeros:
        arr = arr[arr > 0]
    return arr


def is_continuous_mode(mode: str | None) -> bool:
    return normalize_value_mode(mode) == CONTINUOUS
