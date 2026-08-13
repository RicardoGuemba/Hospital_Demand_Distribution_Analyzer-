"""Export replenishment Qmin Monte Carlo summary."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.domain.models import DistributionFitResult, ReplenishmentQminResult
from hospital_demand_analyzer.exports.exporter import export_csv, export_excel
from hospital_demand_analyzer.statistics.fit_params import format_fit_params


def _descricao_for(state: AppState, produto: str) -> str:
    desc = state.descriptive_df
    if desc is None or desc.empty:
        return ""
    match = desc[desc["produto"].astype(str) == str(produto)]
    if match.empty:
        return ""
    return str(match.iloc[0].get("descricao", ""))


def _fmt_nonneg_int(value) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return str(max(0, int(round(number))))


def build_replenishment_summary_row(
    state: AppState,
    produto: str,
    result: ReplenishmentQminResult | None = None,
) -> dict:
    if result is None:
        result = state.replenishment_qmin_results.get(produto)
    if result is None:
        return {
            "Produto": produto,
            "Descricao": _descricao_for(state, produto),
            "Cluster": state.get_cluster_for_sku(str(produto)),
            "Status": "Pendente",
        }
    params_text = "—"
    if result.parametros:
        stub = DistributionFitResult(
            name=result.distribuicao,
            dist_type="discrete",
            params=result.parametros,
            log_likelihood=0.0,
            n_params=len(result.parametros),
            aic=0.0,
            aicc=0.0,
            bic=0.0,
            converged=True,
        )
        params_text = format_fit_params(stub)
    return {
        "Produto": produto,
        "Descricao": _descricao_for(state, produto),
        "Cluster": state.get_cluster_for_sku(str(produto)),
        "CDF escolhida": result.distribuicao,
        "Parametros": params_text,
        "Tempo reabastecimento (min)": _fmt_nonneg_int(result.replenishment_minutes),
        "Saldo inicial": _fmt_nonneg_int(result.saldo_inicial),
        "Qmin proposta P95": _fmt_nonneg_int(result.qmin_proposed),
        "Mediana": _fmt_nonneg_int(result.median),
        "Q1": _fmt_nonneg_int(result.q1),
        "Q3": _fmt_nonneg_int(result.q3),
        "Media": _fmt_nonneg_int(result.mean),
        "P5": _fmt_nonneg_int(result.p5),
        "P90": _fmt_nonneg_int(result.p90),
        "P99": _fmt_nonneg_int(result.p99),
        "Desvio": _fmt_nonneg_int(result.std),
        "Min": _fmt_nonneg_int(result.min_),
        "Max": _fmt_nonneg_int(result.max_),
        "N trajetorias": _fmt_nonneg_int(result.n_trajetorias),
        "Status": "Simulado",
    }


def build_replenishment_summary_df(state: AppState) -> pd.DataFrame:
    rows: list[dict] = []
    for produto in state.session_sku_order:
        if produto not in state.session_selected_skus:
            continue
        rows.append(build_replenishment_summary_row(state, produto))
    return pd.DataFrame(rows)


def export_replenishment_summary(df: pd.DataFrame, path: Path, fmt: str = "csv") -> None:
    if fmt == "xlsx":
        export_excel(df, path, "Qmin_Reabastecimento")
    else:
        export_csv(df, path)
