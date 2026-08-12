"""File readers for Excel and CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_data_file(path: Path) -> pd.DataFrame:
    """Read a single data file based on extension."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if ext in (".xlsx", ".xls"):
        engine = "openpyxl" if ext == ".xlsx" else "xlrd"
        try:
            return pd.read_excel(path, engine=engine, dtype=str)
        except Exception:
            if ext == ".xls":
                raise ValueError(f".xls not supported without xlrd: {path.name}")
            raise
    raise ValueError(f"Unsupported extension: {ext}")
