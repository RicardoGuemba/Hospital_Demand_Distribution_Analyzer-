"""Column normalization and header mapping."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

CANONICAL_COLUMNS = [
    "id",
    "transferencia",
    "data_transf",
    "produto",
    "descricao_produto",
    "status",
    "quantidade",
    "qt_barras",
    "estoque",
    "estoque_descricao",
    "abastecido",
    "qt_abastecida",
]

ESSENTIAL_COLUMNS = ["data_transf", "produto", "descricao_produto", "quantidade"]

HEADER_ALIASES: dict[str, str] = {
    "id": "id",
    "transferência": "transferencia",
    "transferencia": "transferencia",
    "data transf.": "data_transf",
    "data transf": "data_transf",
    "data_transf": "data_transf",
    "produto": "produto",
    "descrição produto": "descricao_produto",
    "descricao produto": "descricao_produto",
    "descricao_produto": "descricao_produto",
    "status": "status",
    "quantidade": "quantidade",
    "qt. barras": "qt_barras",
    "qt barras": "qt_barras",
    "qt_barras": "qt_barras",
    "estoque": "estoque",
    "estoque descrição": "estoque_descricao",
    "estoque descricao": "estoque_descricao",
    "estoque_descricao": "estoque_descricao",
    "abastecido?": "abastecido",
    "abastecido": "abastecido",
    "qt. abastecida": "qt_abastecida",
    "qt abastecida": "qt_abastecida",
    "qt_abastecida": "qt_abastecida",
}


def _normalize_header(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(name).strip())
    key = cleaned.lower()
    return HEADER_ALIASES.get(key, re.sub(r"[^a-z0-9_]", "_", key))


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normalize column headers; return df and original->canonical mapping."""
    mapping: dict[str, str] = {}
    rename: dict[str, str] = {}
    for col in df.columns:
        canonical = _normalize_header(col)
        mapping[str(col)] = canonical
        rename[col] = canonical
    out = df.rename(columns=rename)
    return out, mapping


def normalize_row_values(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from string columns."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )
    return out
