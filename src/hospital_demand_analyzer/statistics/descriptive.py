"""Descriptive statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from hospital_demand_analyzer.domain.models import DescriptiveStats, HourlySeries


def empirical_pmf_expected(values: np.ndarray) -> float:
    """E[X] from the empirical PMF: sum(x * P_hat(x)) over the hourly grid."""
    values = values.astype(float)
    n = len(values)
    if n == 0:
        return 0.0
    unique, counts = np.unique(values, return_counts=True)
    probs = counts / n
    return float(np.sum(unique * probs))


def compute_descriptive(hs: HourlySeries) -> DescriptiveStats:
    """Compute descriptive statistics for a normalized hourly series.

    All metrics include zero-filled hours (null demand intervals count as 0).

    ``mean`` is the arithmetic average over the full hourly grid (zeros included).
    ``expected_empirical`` is E[X] = sum(x * P_hat(x)) using the empirical PMF.
    ``expected_conditional`` is E[X | X > 0]: average demand in hours with consumption.
    """
    values = hs.quantities.values.astype(float)
    n = len(values)
    if n == 0:
        return DescriptiveStats(
            produto=hs.produto, descricao=hs.descricao,
            n=0, n_zeros=0, zero_ratio=0, sum_=0, mean=0,
            expected_empirical=0, expected_conditional=0, median=0,
            std=0, var=0, cv=0, min_=0, max_=0,
            q25=0, q50=0, q75=0, iqr=0,
            p90=0, p95=0, p975=0, p99=0, skewness=0, kurtosis=0,
        )

    n_zeros = int((values == 0).sum())
    non_zero_hours = int((values > 0).sum())
    total = float(np.sum(values))
    mean = float(np.mean(values))
    expected_empirical = empirical_pmf_expected(values)
    expected_conditional = total / non_zero_hours if non_zero_hours > 0 else 0.0
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    var = float(np.var(values, ddof=1)) if n > 1 else 0.0
    cv = std / mean if mean != 0 else 0.0

    return DescriptiveStats(
        produto=hs.produto,
        descricao=hs.descricao,
        n=n,
        n_zeros=n_zeros,
        zero_ratio=n_zeros / n,
        sum_=total,
        mean=mean,
        expected_empirical=expected_empirical,
        expected_conditional=expected_conditional,
        median=float(np.median(values)),
        std=std,
        var=var,
        cv=cv,
        min_=float(np.min(values)),
        max_=float(np.max(values)),
        q25=float(np.percentile(values, 25)),
        q50=float(np.percentile(values, 50)),
        q75=float(np.percentile(values, 75)),
        iqr=float(np.percentile(values, 75) - np.percentile(values, 25)),
        p90=float(np.percentile(values, 90)),
        p95=float(np.percentile(values, 95)),
        p975=float(np.percentile(values, 97.5)),
        p99=float(np.percentile(values, 99)),
        skewness=float(stats.skew(values)) if n > 2 else 0.0,
        kurtosis=float(stats.kurtosis(values)) if n > 3 else 0.0,
        date_start=hs.timestamps.min().to_pydatetime() if n else None,
        date_end=hs.timestamps.max().to_pydatetime() if n else None,
        non_zero_hours=non_zero_hours,
    )


def descriptive_to_dict(ds: DescriptiveStats) -> dict:
    """Serialize descriptive stats to a flat row dict."""
    return {
        "produto": ds.produto,
        "descricao": ds.descricao,
        "n": ds.n,
        "n_zeros": ds.n_zeros,
        "zero_ratio": ds.zero_ratio,
        "non_zero_hours": ds.non_zero_hours,
        "sum": ds.sum_,
        "mean": ds.mean,
        "expected_empirical": ds.expected_empirical,
        "expected_conditional": ds.expected_conditional,
        "median": ds.median,
        "std": ds.std,
        "var": ds.var,
        "cv": ds.cv,
        "min": ds.min_,
        "max": ds.max_,
        "q25": ds.q25,
        "q50": ds.q50,
        "q75": ds.q75,
        "iqr": ds.iqr,
        "p90": ds.p90,
        "p95": ds.p95,
        "p975": ds.p975,
        "p99": ds.p99,
        "skewness": ds.skewness,
        "kurtosis": ds.kurtosis,
        "date_start": ds.date_start,
        "date_end": ds.date_end,
    }


DESCRIPTIVE_TABLE_COLUMNS: list[tuple[str, str]] = [
    ("produto", "Produto"),
    ("descricao", "Descrição"),
    ("cluster", "Cluster"),
    ("n", "N (horas)"),
    ("non_zero_hours", "Horas c/ Demanda"),
    ("n_zeros", "Horas Zero"),
    ("zero_ratio", "% Zero"),
    ("sum", "Soma"),
    ("mean", "Média aritmética"),
    ("expected_empirical", "E[X] empírico"),
    ("expected_conditional", "E[X|X>0]"),
    ("median", "Mediana"),
    ("std", "Desvio Padrão"),
    ("var", "Variância"),
    ("cv", "CV"),
    ("min", "Mínimo"),
    ("max", "Máximo"),
    ("q25", "Q25"),
    ("q50", "Q50"),
    ("q75", "Q75"),
    ("iqr", "IQR"),
    ("p90", "P90"),
    ("p95", "P95"),
    ("p975", "P97.5"),
    ("p99", "P99"),
    ("skewness", "Assimetria"),
    ("kurtosis", "Curtose"),
]

METRIC_TOOLTIPS: dict[str, str] = {
    "mean": "Média aritmética: soma da demanda ÷ N horas da grade (zeros incluídos).",
    "expected_empirical": (
        "E[X] empírico: Σ x·P̂(x), com P̂(x) = frequência relativa na grade horária. "
        "Coincide com a média aritmética na mesma amostra."
    ),
    "expected_conditional": (
        "E[X|X>0]: demanda esperada nas horas em que houve consumo "
        "(soma ÷ horas com demanda > 0)."
    ),
    "expected_theoretical": (
        "E[X] teórico: valor esperado da melhor distribuição ajustada (menor AIC). "
        "Disponível após ajuste de distribuição."
    ),
}


def compute_all_descriptive(series_dict: dict[str, HourlySeries]) -> pd.DataFrame:
    rows = [descriptive_to_dict(compute_descriptive(hs)) for hs in series_dict.values()]
    return pd.DataFrame(rows)
