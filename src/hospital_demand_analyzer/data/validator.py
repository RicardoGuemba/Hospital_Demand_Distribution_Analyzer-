"""Data validation."""

from __future__ import annotations

import pandas as pd

from hospital_demand_analyzer.data.normalizer import ESSENTIAL_COLUMNS
from hospital_demand_analyzer.domain.models import FileQualityReport, GlobalInconsistencies


def validate_file_df(
    df: pd.DataFrame, file_name: str
) -> tuple[pd.DataFrame, FileQualityReport, GlobalInconsistencies]:
    """Validate and clean a single normalized dataframe."""
    report = FileQualityReport(
        file_name=file_name,
        rows_total=len(df),
        rows_valid=0,
        rows_rejected=0,
        columns=len(df.columns),
        skus=0,
        date_min=None,
        date_max=None,
    )
    inconsistencies = GlobalInconsistencies()

    missing = [c for c in ESSENTIAL_COLUMNS if c not in df.columns]
    report.missing_columns = missing
    if missing:
        report.status = "error"
        report.errors.append(f"Missing essential columns: {missing}")
        inconsistencies.missing_columns.extend(missing)
        return df.iloc[0:0], report, inconsistencies

    work = df.copy()
    report.null_counts = {c: int(work[c].isna().sum()) for c in ESSENTIAL_COLUMNS}

    # Parse dates
    parsed_dates = pd.to_datetime(work["data_transf"], errors="coerce")
    invalid_date_mask = parsed_dates.isna() & work["data_transf"].notna()
    report.invalid_dates = int(invalid_date_mask.sum())
    inconsistencies.invalid_dates += report.invalid_dates

    # Parse quantities
    qty = pd.to_numeric(work["quantidade"], errors="coerce")
    invalid_qty_mask = qty.isna() & work["quantidade"].notna()
    report.invalid_quantities = int(invalid_qty_mask.sum())
    inconsistencies.invalid_quantities += report.invalid_quantities

    reject_mask = invalid_date_mask | invalid_qty_mask | parsed_dates.isna() | qty.isna()
    report.rows_rejected = int(reject_mask.sum())

    valid = work[~reject_mask].copy()
    valid["data_transf"] = parsed_dates[~reject_mask].values
    valid["quantidade"] = qty[~reject_mask].values

    # Negative quantities flagged but kept
    neg = valid["quantidade"] < 0
    if neg.any():
        report.errors.append(f"{int(neg.sum())} negative quantities found")

    valid["produto"] = valid["produto"].astype(str).str.strip()
    valid["descricao_produto"] = valid["descricao_produto"].fillna("").astype(str).str.strip()

    no_desc = valid["descricao_produto"] == ""
    if no_desc.any():
        skus = valid.loc[no_desc, "produto"].unique().tolist()
        inconsistencies.skus_without_description.extend(skus)

    report.rows_valid = len(valid)
    report.skus = int(valid["produto"].nunique()) if len(valid) else 0
    if len(valid):
        report.date_min = pd.Timestamp(valid["data_transf"].min()).to_pydatetime()
        report.date_max = pd.Timestamp(valid["data_transf"].max()).to_pydatetime()

    report.status = "ok" if report.rows_valid > 0 else "empty"
    return valid, report, inconsistencies


def detect_divergent_descriptions(df: pd.DataFrame) -> dict[str, list[str]]:
    """Find SKUs with multiple descriptions."""
    divergent: dict[str, list[str]] = {}
    for produto, group in df.groupby("produto"):
        descs = group["descricao_produto"].dropna().unique().tolist()
        descs = [d for d in descs if d]
        if len(descs) > 1:
            divergent[str(produto)] = descs
    return divergent


def canonical_description(descriptions: list[str]) -> str:
    """Pick most frequent non-empty description."""
    non_empty = [d for d in descriptions if d]
    if not non_empty:
        return ""
    from collections import Counter
    return Counter(non_empty).most_common(1)[0][0]
