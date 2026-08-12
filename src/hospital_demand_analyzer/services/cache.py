"""Parquet cache with fingerprint invalidation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def _fingerprint(paths: list[Path]) -> str:
    parts = []
    for p in sorted(paths):
        if p.exists():
            stat = p.stat()
            parts.append(f"{p}:{stat.st_size}:{stat.st_mtime}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


class ParquetCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = cache_dir / "cache_meta.json"

    def _meta(self) -> dict:
        if self._meta_path.exists():
            return json.loads(self._meta_path.read_text(encoding="utf-8"))
        return {}

    def _save_meta(self, meta: dict) -> None:
        self._meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    def is_valid(self, key: str, source_paths: list[Path]) -> bool:
        meta = self._meta()
        entry = meta.get(key)
        if not entry:
            return False
        return entry.get("fingerprint") == _fingerprint(source_paths)

    def load(self, key: str) -> pd.DataFrame | None:
        path = self.cache_dir / f"{key}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def save(self, key: str, df: pd.DataFrame, source_paths: list[Path]) -> None:
        path = self.cache_dir / f"{key}.parquet"
        df.to_parquet(path, index=False)
        meta = self._meta()
        meta[key] = {
            "fingerprint": _fingerprint(source_paths),
            "saved_at": datetime.now().isoformat(),
            "rows": len(df),
        }
        self._save_meta(meta)

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink(missing_ok=True)
        if self._meta_path.exists():
            self._meta_path.unlink()
