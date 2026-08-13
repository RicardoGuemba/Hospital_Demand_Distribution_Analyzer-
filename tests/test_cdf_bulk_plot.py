"""Tests for bulk CDF plot export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from hospital_demand_analyzer.domain.models import DistributionFitResult
from hospital_demand_analyzer.plots.cdf_bulk_plot import save_sku_cdf_png


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _poisson_fit() -> DistributionFitResult:
    return DistributionFitResult(
        name="Poisson",
        dist_type="discrete",
        params={"lambda": 3.0},
        log_likelihood=-50.0,
        n_params=1,
        aic=102.0,
        aicc=102.0,
        bic=105.0,
        converged=True,
        rank=1,
    )


def test_save_sku_cdf_png(qapp, tmp_path: Path):
    data = np.array([0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 2.0])
    out = tmp_path / "sku100_cdf.png"
    save_sku_cdf_png(out, data, [_poisson_fit()], title="CDF test")
    assert out.is_file()
    assert out.stat().st_size > 1000
