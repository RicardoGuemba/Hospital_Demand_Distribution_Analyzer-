"""Directory scanner for demand data files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from hospital_demand_analyzer.domain.models import FileInfo

SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}
LOCK_PREFIX = "~$"


def scan_directory(directory: Path) -> list[FileInfo]:
    """Scan directory for supported data files (read-only)."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files: list[FileInfo] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith(LOCK_PREFIX):
            continue
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        stat = path.stat()
        files.append(
            FileInfo(
                name=path.name,
                path=str(path),
                extension=ext,
                size_bytes=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime),
            )
        )
    return files
