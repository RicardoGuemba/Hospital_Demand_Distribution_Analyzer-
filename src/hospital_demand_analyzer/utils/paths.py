"""Project path utilities."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return project root (where run_app.bat lives)."""
    return Path(__file__).resolve().parents[3]


def resolve_path(relative: str) -> Path:
    return project_root() / relative
