"""Hourly time series construction."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from hospital_demand_analyzer.domain.models import DataGapType, HourlySeries


def floor_to_hour(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.floor("h")


def compute_global_hour_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Global hourly index shared by all SKUs (first to last record in dataset)."""
    if df.empty:
        return pd.DatetimeIndex([])
    first_record = pd.Timestamp(df["data_transf"].min())
    last_record = pd.Timestamp(df["data_transf"].max())
    period_start = first_record.floor("h")
    period_end = last_record.floor("h")
    return pd.date_range(start=period_start, end=period_end, freq="h")


def global_period_bounds(df: pd.DataFrame) -> tuple[datetime | None, datetime | None, int]:
    index = compute_global_hour_index(df)
    if len(index) == 0:
        return None, None, 0
    return index.min().to_pydatetime(), index.max().to_pydatetime(), len(index)


def build_hourly_series(
    df: pd.DataFrame,
    produto: str,
    global_index: pd.DatetimeIndex | None = None,
) -> HourlySeries:
    """Build normalized hourly series for a single SKU on the global hourly grid.

    All SKUs share the same ``global_index`` (1-hour steps). Hours without
    demand for this SKU are zero-filled and count in statistics.
    """
    if global_index is None or len(global_index) == 0:
        global_index = compute_global_hour_index(df)

    sku_df = df[df["produto"] == produto].copy()
    if sku_df.empty or len(global_index) == 0:
        return HourlySeries(
            produto=produto,
            descricao="",
            timestamps=global_index,
            quantities=pd.Series(0.0, index=global_index),
            gap_types=pd.Series(DataGapType.ZERO_DEMAND.value, index=global_index),
        )

    desc = (
        sku_df["descricao_canonica"].iloc[0]
        if "descricao_canonica" in sku_df.columns
        else sku_df["descricao_produto"].iloc[0]
    )

    first_record = pd.Timestamp(sku_df["data_transf"].min())
    last_record = pd.Timestamp(sku_df["data_transf"].max())

    sku_df["hour"] = floor_to_hour(sku_df["data_transf"])
    hourly = sku_df.groupby("hour")["quantidade"].sum()
    reindexed = hourly.reindex(global_index, fill_value=0.0).astype(float)

    gap_types = pd.Series(DataGapType.ZERO_DEMAND.value, index=global_index)

    return HourlySeries(
        produto=str(produto),
        descricao=str(desc),
        timestamps=global_index,
        quantities=reindexed,
        gap_types=gap_types,
        first_record_at=first_record.to_pydatetime(),
        last_record_at=last_record.to_pydatetime(),
    )


def build_all_hourly_series(df: pd.DataFrame) -> dict[str, HourlySeries]:
    """Build hourly series for all SKUs on a shared global grid."""
    if df.empty:
        return {}
    global_index = compute_global_hour_index(df)
    result: dict[str, HourlySeries] = {}
    for produto in df["produto"].unique():
        result[str(produto)] = build_hourly_series(df, str(produto), global_index)
    return result


def hourly_series_to_dataframe(series_dict: dict[str, HourlySeries]) -> pd.DataFrame:
    """Convert all hourly series to a long dataframe."""
    rows = []
    for produto, hs in series_dict.items():
        for ts, qty in zip(hs.timestamps, hs.quantities, strict=False):
            rows.append({
                "timestamp": ts,
                "produto": produto,
                "descricao": hs.descricao,
                "quantidade": qty,
            })
    return pd.DataFrame(rows)
