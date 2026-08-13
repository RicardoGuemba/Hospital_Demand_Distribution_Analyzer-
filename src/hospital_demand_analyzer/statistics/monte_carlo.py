"""Monte Carlo simulation of TempQMin (time until inventory reaches Qmin)."""

from __future__ import annotations

import numpy as np

from hospital_demand_analyzer.domain.models import (
    DistributionFitResult,
    MonteCarloTrajectory,
    TempQMinResult,
)
from hospital_demand_analyzer.statistics.demand_sampler import sample_hourly_demand

DEFAULT_MAX_HORIZON_HOURS = 8760
MINUTES_PER_STEP = 60.0
SALDO_FACTOR_MIN = 5.0
SALDO_FACTOR_MAX = 10.0


def default_qmin(mean_hourly: float) -> int:
    """Qmin padrão: demanda média horária × 24 h (inteiro positivo)."""
    return max(1, int(round(mean_hourly * 24.0)))


def sample_saldo_factor(rng: np.random.Generator) -> float:
    """Fator multiplicador aleatório para saldo inicial (5–10)."""
    return float(rng.uniform(SALDO_FACTOR_MIN, SALDO_FACTOR_MAX))


def default_saldo_inicial(qmin: int, rng: np.random.Generator) -> int:
    """Saldo inicial = Qmin × Fator_Multiplicador (inteiro positivo)."""
    return max(1, int(round(qmin * sample_saldo_factor(rng))))


def format_tempqmin(minutes: float) -> str:
    """Formata TempQMin em minutos com equivalente em horas (1 casa decimal)."""
    hours = minutes / MINUTES_PER_STEP
    return f"{minutes:.1f} min ({hours:.1f} h)"


def normalize_inventory_level(value: float) -> int:
    """Garante saldo/Qmin como inteiro estritamente positivo."""
    return max(1, int(round(value)))


def simulate_single_trajectory(
    saldo_inicial: float,
    qmin: float,
    fit: DistributionFitResult,
    *,
    rng: np.random.Generator,
    max_horizon_hours: float = DEFAULT_MAX_HORIZON_HOURS,
) -> MonteCarloTrajectory:
    """Simulate one inventory path with hourly steps; times and TempQMin in minutes."""
    if saldo_inicial <= qmin:
        return MonteCarloTrajectory(
            times=np.array([0.0]),
            balances=np.array([saldo_inicial]),
            temp_qmin=0.0,
        )

    saldo = float(saldo_inicial)
    tempo_min = 0.0
    times = [0.0]
    balances = [saldo]
    max_horizon_min = max_horizon_hours * MINUTES_PER_STEP

    while saldo > qmin and tempo_min < max_horizon_min:
        demanda = sample_hourly_demand(fit, rng)

        if demanda == 0:
            tempo_min += MINUTES_PER_STEP
            times.append(tempo_min)
            balances.append(saldo)
            continue

        saldo_anterior = saldo
        novo_saldo = saldo_anterior - demanda

        if novo_saldo <= qmin:
            fracao = (saldo_anterior - qmin) / demanda
            tempo_min += fracao * MINUTES_PER_STEP
            times.append(tempo_min)
            balances.append(qmin)
            saldo = qmin
            break

        saldo = novo_saldo
        tempo_min += MINUTES_PER_STEP
        times.append(tempo_min)
        balances.append(saldo)

    if saldo > qmin:
        tempo_min = max_horizon_min

    return MonteCarloTrajectory(
        times=np.asarray(times, dtype=float),
        balances=np.asarray(balances, dtype=float),
        temp_qmin=float(tempo_min),
    )


def compute_tempqmin_stats(times: np.ndarray) -> dict[str, float]:
    if len(times) == 0:
        return {
            "mean": 0.0, "median": 0.0, "p5": 0.0, "p10": 0.0,
            "p90": 0.0, "p95": 0.0, "std": 0.0, "min_": 0.0, "max_": 0.0,
        }
    return {
        "mean": float(np.mean(times)),
        "median": float(np.median(times)),
        "p5": float(np.percentile(times, 5)),
        "p10": float(np.percentile(times, 10)),
        "p90": float(np.percentile(times, 90)),
        "p95": float(np.percentile(times, 95)),
        "std": float(np.std(times)),
        "min_": float(np.min(times)),
        "max_": float(np.max(times)),
    }


def run_tempqmin_monte_carlo(
    produto: str,
    saldo_inicial: float,
    qmin: float,
    fit: DistributionFitResult,
    *,
    n_trajetorias: int = 100,
    seed: int | None = None,
    max_horizon_hours: float = DEFAULT_MAX_HORIZON_HOURS,
) -> TempQMinResult:
    """Run N Monte Carlo trajectories; TempQMin reported in minutes."""
    saldo_inicial = normalize_inventory_level(saldo_inicial)
    qmin = normalize_inventory_level(qmin)
    rng = np.random.default_rng(seed)
    trajectories: list[MonteCarloTrajectory] = []

    for _ in range(n_trajetorias):
        trajectories.append(
            simulate_single_trajectory(
                saldo_inicial, qmin, fit,
                rng=rng, max_horizon_hours=max_horizon_hours,
            )
        )

    temp_times = np.array([t.temp_qmin for t in trajectories], dtype=float)
    stats = compute_tempqmin_stats(temp_times)

    return TempQMinResult(
        produto=produto,
        saldo_inicial=saldo_inicial,
        qmin=qmin,
        n_trajetorias=n_trajetorias,
        delta_t_minutos=MINUTES_PER_STEP,
        distribuicao=fit.name,
        parametros={k: float(v) for k, v in fit.params.items()},
        trajectories=trajectories,
        temp_qmin_times=temp_times,
        **stats,
    )
