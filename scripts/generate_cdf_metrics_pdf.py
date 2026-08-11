#!/usr/bin/env python3
"""CLI: generate METRICAS_CDF.pdf from docs/METRICAS_CDF.md."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hospital_demand_analyzer.exports.cdf_metrics_pdf import generate_cdf_metrics_pdf  # noqa: E402


def main() -> int:
    out = generate_cdf_metrics_pdf()
    print(f"PDF gerado: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
