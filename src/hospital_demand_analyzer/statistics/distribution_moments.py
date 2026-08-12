"""Theoretical moments from fitted distributions."""

from __future__ import annotations

from hospital_demand_analyzer.domain.models import DistributionFitResult
from hospital_demand_analyzer.statistics.distributions.registry import get_distribution
from hospital_demand_analyzer.statistics.fitter import _resolve_dist


def theoretical_expected_value(fit: DistributionFitResult) -> float | None:
    """Return E[X] under the fitted distribution, or None if unavailable."""
    if not fit.converged or not fit.params:
        return None
    spec = get_distribution(fit.name)
    if spec is None:
        return None
    dist, params = _resolve_dist(spec, fit.params)
    if dist is None or not params:
        return None
    try:
        return float(dist.mean(*params))
    except Exception:
        return None
