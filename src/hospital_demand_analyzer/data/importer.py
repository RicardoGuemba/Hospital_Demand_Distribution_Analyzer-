"""Import orchestration service."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from hospital_demand_analyzer.data.normalizer import normalize_columns, normalize_row_values
from hospital_demand_analyzer.data.reader import read_data_file
from hospital_demand_analyzer.data.scanner import scan_directory
from hospital_demand_analyzer.data.validator import (
    canonical_description,
    detect_divergent_descriptions,
    validate_file_df,
)
from hospital_demand_analyzer.domain.models import GlobalInconsistencies, ImportResult

logger = logging.getLogger("hda")


def import_directory(data_dir: Path) -> ImportResult:
    """Import and validate all files in directory."""
    files = scan_directory(data_dir)
    all_frames: list[pd.DataFrame] = []
    file_reports = []
    global_inc = GlobalInconsistencies()

    for finfo in files:
        path = Path(finfo.path)
        try:
            raw = read_data_file(path)
            raw, _mapping = normalize_columns(raw)
            raw = normalize_row_values(raw)
            raw["source_file"] = finfo.name

            valid, report, inc = validate_file_df(raw, finfo.name)
            file_reports.append(report)
            global_inc.missing_columns = list(
                set(global_inc.missing_columns + inc.missing_columns)
            )
            global_inc.invalid_dates += inc.invalid_dates
            global_inc.invalid_quantities += inc.invalid_quantities
            global_inc.skus_without_description.extend(inc.skus_without_description)

            if len(valid):
                all_frames.append(valid)
        except Exception as exc:
            logger.exception("Failed to import %s", finfo.name)
            from hospital_demand_analyzer.domain.models import FileQualityReport
            file_reports.append(
                FileQualityReport(
                    file_name=finfo.name,
                    rows_total=0,
                    rows_valid=0,
                    rows_rejected=0,
                    columns=0,
                    skus=0,
                    date_min=None,
                    date_max=None,
                    errors=[str(exc)],
                    status="error",
                )
            )

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    if not combined.empty:
        global_inc.divergent_descriptions = detect_divergent_descriptions(combined)
        global_inc.duplicate_rows = int(combined.duplicated().sum())
        # Add canonical description
        desc_map = {}
        for produto, group in combined.groupby("produto"):
            desc_map[str(produto)] = canonical_description(
                group["descricao_produto"].tolist()
            )
        combined["descricao_canonica"] = combined["produto"].map(desc_map)

    return ImportResult(
        normalized_df=combined,
        file_reports=file_reports,
        inconsistencies=global_inc,
        data_dir=str(data_dir),
        imported_at=datetime.now(),
    )
