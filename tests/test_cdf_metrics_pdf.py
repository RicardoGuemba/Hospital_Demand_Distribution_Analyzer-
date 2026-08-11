"""Tests for CDF metrics PDF generation."""

from __future__ import annotations

from hospital_demand_analyzer.exports.cdf_metrics_pdf import generate_cdf_metrics_pdf
from hospital_demand_analyzer.utils.paths import project_root


def test_generate_cdf_metrics_pdf_creates_file(tmp_path):
    md = project_root() / "docs" / "METRICAS_CDF.md"
    assert md.is_file(), "docs/METRICAS_CDF.md must exist"

    out = tmp_path / "METRICAS_CDF.pdf"
    result = generate_cdf_metrics_pdf(output_path=out, markdown_path=md)

    assert result == out
    assert out.is_file()
    assert out.stat().st_size > 1000
    assert out.read_bytes()[:4] == b"%PDF"
