"""Export CDF summary, rankings and PNG plots for all SKUs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hospital_demand_analyzer.app.controller import AppController
from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.domain.demand_values import prepare_demand_values
from hospital_demand_analyzer.exports.exporter import export_csv, export_excel
from hospital_demand_analyzer.plots.cdf_bulk_plot import save_sku_cdf_png
from hospital_demand_analyzer.statistics.fit_params import best_fit, format_fit_params


def _series_dict(state: AppState, use_treated: bool) -> dict:
    if use_treated and state.treated_series:
        merged = dict(state.hourly_series)
        merged.update(state.treated_series)
        return merged
    return state.hourly_series


def _processing_produtos(state: AppState, use_treated: bool) -> list[str]:
    allowed = set(state.processing_skus())
    return sorted(p for p in _series_dict(state, use_treated).keys() if p in allowed)


def _values_for_sku(state: AppState, produto: str, use_treated: bool):
    hs = state.treated_series.get(produto) if use_treated else None
    if hs is None:
        hs = state.hourly_series.get(produto)
    if hs is None:
        return None
    return prepare_demand_values(hs.quantities.values.astype(float), include_zeros=True)


def build_cdf_summary_row(
    state: AppState,
    controller: AppController,
    produto: str,
    *,
    use_treated: bool = False,
    fits: list | None = None,
) -> dict:
    """Build one summary row for a SKU (for incremental table updates)."""
    mode = state.demand_value_mode
    if fits is None:
        fits = controller.get_fit_results(produto, positive_only=False, value_mode=mode)
    chosen = best_fit(fits or [])

    desc = state.descriptive_df
    desc_row = None
    if desc is not None and not desc.empty:
        match = desc[desc["produto"].astype(str) == str(produto)]
        if not match.empty:
            desc_row = match.iloc[0]

    n_converged = len([f for f in (fits or []) if f.converged])
    status = "Ajustado" if chosen else ("Pendente" if fits is None else "Sem convergência")

    return {
        "Produto": produto,
        "Descricao": desc_row.get("descricao", "") if desc_row is not None else "",
        "Cluster": state.get_cluster_for_sku(str(produto)),
        "CDF escolhida": chosen.name if chosen else "",
        "Parametros": format_fit_params(chosen) if chosen else "",
        "AIC": chosen.aic if chosen else "",
        "BIC": chosen.bic if chosen else "",
        "Tail Score": chosen.tail_fit_score if chosen else "",
        "KS": chosen.ks_stat if chosen else "",
        "Candidatas convergentes": n_converged,
        "Status": status,
        "Modo valor": mode,
    }


def build_cdf_summary_df(
    state: AppState,
    controller: AppController,
    *,
    use_treated: bool = False,
) -> pd.DataFrame:
    """One row per SKU with chosen CDF and representative parameters."""
    rows: list[dict] = []

    for produto in _processing_produtos(state, use_treated):
        rows.append(
            build_cdf_summary_row(
                state, controller, produto, use_treated=use_treated,
            )
        )

    return pd.DataFrame(rows)


def export_cdf_summary(df: pd.DataFrame, path: Path, fmt: str = "csv") -> None:
    if fmt == "xlsx":
        export_excel(df, path, "CDF_Resumo")
    else:
        export_csv(df, path)


def build_cdf_rankings_long_df(
    state: AppState,
    controller: AppController,
    *,
    use_treated: bool = False,
) -> pd.DataFrame:
    mode = state.demand_value_mode
    rows: list[dict] = []
    for produto in sorted(_series_dict(state, use_treated).keys()):
        fits = controller.get_fit_results(produto, positive_only=False, value_mode=mode) or []
        for fit in sorted([f for f in fits if f.converged], key=lambda f: f.aic):
            rows.append({
                "Produto": produto,
                "Rank": fit.rank,
                "Distribuicao": fit.name,
                "Tipo": fit.dist_type,
                "Parametros": format_fit_params(fit),
                "AIC": fit.aic,
                "Delta AIC": fit.delta_aic,
                "BIC": fit.bic,
                "Tail Score": fit.tail_fit_score,
                "KS": fit.ks_stat,
            })
    return pd.DataFrame(rows)


def export_cdf_rankings_workbook(
    state: AppState,
    controller: AppController,
    path: Path,
    *,
    use_treated: bool = False,
) -> None:
    """Export long-format rankings CSV or single-sheet XLSX."""
    df = build_cdf_rankings_long_df(state, controller, use_treated=use_treated)
    if path.suffix.lower() == ".xlsx":
        export_excel(df, path, "Rankings_CDF")
    else:
        export_csv(df, path)


def export_all_cdf_pngs(
    state: AppState,
    controller: AppController,
    out_dir: Path,
    *,
    use_treated: bool = False,
) -> int:
    """Save one PNG per SKU with all candidate CDFs. Returns count written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = state.demand_value_mode
    count = 0
    for produto in sorted(_series_dict(state, use_treated).keys()):
        values = _values_for_sku(state, produto, use_treated)
        if values is None or len(values) == 0:
            continue
        fits = controller.get_fit_results(produto, positive_only=False, value_mode=mode)
        if not fits:
            continue
        safe_name = str(produto).replace("/", "_").replace("\\", "_")
        path = out_dir / f"{safe_name}_cdf.png"
        save_sku_cdf_png(
            path,
            values,
            fits,
            title=f"CDF — {produto}",
            positive_only=False,
        )
        count += 1
    return count
