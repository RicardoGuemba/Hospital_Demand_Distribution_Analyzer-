"""Monte Carlo Qmin from accumulated demand over a replenishment interval."""

from __future__ import annotations

import math

import numpy as np

from hospital_demand_analyzer.domain.models import (
    DemandAccumulationTrajectory,
    DistributionFitResult,
    ReplenishmentQminResult,
)
from hospital_demand_analyzer.statistics.demand_sampler import sample_hourly_demand
from hospital_demand_analyzer.statistics.monte_carlo import MINUTES_PER_STEP, normalize_inventory_level

DEFAULT_REPLENISHMENT_MINUTES = 120.0
DEFAULT_SALDO_FACTOR = 7.0


def _hourly_demand_units(fit: DistributionFitResult, rng: np.random.Generator) -> int:
    """Non-negative integer demand for one hour."""
    try:
        raw = float(sample_hourly_demand(fit, rng))
    except (ValueError, TypeError, OverflowError):
        return 0
    if not math.isfinite(raw):
        return 0
    return max(0, int(round(raw)))


def _partial_hour_demand(units: int, fraction: float) -> int:
    if units <= 0 or fraction <= 0:
        return 0
    return max(0, int(round(units * fraction)))


def _finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _safe_percentile(values: np.ndarray, pct: float, default: float = 0.0) -> float:
    clean = _finite_values(values)
    if len(clean) == 0:
        return default
    return float(np.percentile(clean, pct))


def _sanitize_fit_params(params: dict) -> dict[str, float]:
    clean: dict[str, float] = {}
    for key, val in params.items():
        try:
            number = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean[str(key)] = number
    return clean


def simulate_demand_accumulation_trajectory(
    replenishment_minutes: float,
    fit: DistributionFitResult,
    *,
    rng: np.random.Generator,
    saldo_inicial: int | None = None,
) -> DemandAccumulationTrajectory:
    """Simulate integer demand accumulation until replenishment time ends."""
    remaining = float(replenishment_minutes)
    elapsed = 0.0
    cumulative = 0
    times = [0.0]
    cum_demands = [0.0]

    while remaining > 1e-9:
        demand = _hourly_demand_units(fit, rng)
        if remaining >= MINUTES_PER_STEP:
            cumulative += demand
            elapsed += MINUTES_PER_STEP
            remaining -= MINUTES_PER_STEP
        else:
            partial = _partial_hour_demand(demand, remaining / MINUTES_PER_STEP)
            cumulative += partial
            elapsed += remaining
            remaining = 0.0
        times.append(elapsed)
        cum_demands.append(float(cumulative))

    times_arr = np.asarray(times, dtype=float)
    cum_arr = np.asarray(cum_demands, dtype=float)
    if saldo_inicial is None:
        balances = np.zeros_like(cum_arr)
    else:
        balances = np.asarray(saldo_inicial, dtype=float) - cum_arr

    return DemandAccumulationTrajectory(
        times=times_arr,
        cumulative_demand=cum_arr,
        balances=balances,
        total_demand=int(cumulative),
    )


def compute_accumulated_demand_stats(values: np.ndarray) -> dict[str, float]:
    clean = _finite_values(values)
    if len(clean) == 0:
        return {
            "mean": 0.0, "median": 0.0, "q1": 0.0, "q3": 0.0,
            "p5": 0.0, "p10": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0,
            "std": 0.0, "min_": 0.0, "max_": 0.0,
        }
    return {
        "mean": float(np.mean(clean)),
        "median": _safe_percentile(clean, 50),
        "q1": _safe_percentile(clean, 25),
        "q3": _safe_percentile(clean, 75),
        "p5": _safe_percentile(clean, 5),
        "p10": _safe_percentile(clean, 10),
        "p90": _safe_percentile(clean, 90),
        "p95": _safe_percentile(clean, 95),
        "p99": _safe_percentile(clean, 99),
        "std": float(np.std(clean)) if len(clean) > 1 else 0.0,
        "min_": float(np.min(clean)),
        "max_": float(np.max(clean)),
    }


def _proposed_qmin_from_demands(accumulated: np.ndarray) -> int:
    """P95 of accumulated demand as a positive integer Qmin."""
    p95 = _safe_percentile(accumulated, 95, default=0.0)
    if p95 <= 0:
        peak = float(np.max(_finite_values(accumulated))) if len(_finite_values(accumulated)) else 0.0
        if peak <= 0:
            return 1
        return normalize_inventory_level(math.ceil(peak))
    return normalize_inventory_level(math.ceil(p95))


def _default_saldo_inicial(qmin: int, stats: dict[str, float]) -> int:
    p99 = stats.get("p99", 0.0)
    baseline = max(qmin * DEFAULT_SALDO_FACTOR, p99, float(qmin))
    return normalize_inventory_level(math.ceil(baseline))


def run_replenishment_qmin_monte_carlo(
    produto: str,
    replenishment_minutes: float,
    fit: DistributionFitResult,
    *,
    n_trajetorias: int = 100,
    seed: int | None = None,
) -> ReplenishmentQminResult:
    """Run N trajectories; proposed Qmin = ceil(P95) of integer accumulated demands."""
    rng = np.random.default_rng(seed)
    trajectories: list[DemandAccumulationTrajectory] = []

    for _ in range(n_trajetorias):
        trajectories.append(
            simulate_demand_accumulation_trajectory(
                replenishment_minutes, fit, rng=rng,
            )
        )

    accumulated = np.array([t.total_demand for t in trajectories], dtype=float)
    stats = compute_accumulated_demand_stats(accumulated)
    qmin = _proposed_qmin_from_demands(accumulated)
    stats["p95"] = float(qmin)
    saldo_inicial = _default_saldo_inicial(qmin, stats)

    enriched: list[DemandAccumulationTrajectory] = []
    for traj in trajectories:
        enriched.append(
            DemandAccumulationTrajectory(
                times=traj.times,
                cumulative_demand=traj.cumulative_demand,
                balances=np.asarray(saldo_inicial, dtype=float) - traj.cumulative_demand,
                total_demand=traj.total_demand,
            )
        )

    return ReplenishmentQminResult(
        produto=produto,
        replenishment_minutes=float(replenishment_minutes),
        n_trajetorias=n_trajetorias,
        distribuicao=fit.name,
        parametros=_sanitize_fit_params(fit.params),
        trajectories=enriched,
        accumulated_demands=accumulated,
        saldo_inicial=saldo_inicial,
        qmin_proposed=float(qmin),
        **stats,
    )
