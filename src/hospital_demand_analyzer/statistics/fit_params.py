"""Map fitted distribution names to scipy-like parameter tuples and display helpers."""

from __future__ import annotations

from hospital_demand_analyzer.domain.models import DistributionFitResult
from hospital_demand_analyzer.statistics.distributions.zero_inflated import ZINB_NAME, ZIP_NAME

_PARAM_LABELS = {
    "lambda": "λ",
    "n": "n",
    "p": "p",
    "pi": "π",
}


def params_tuple_for_fit(name: str, params: dict[str, float]) -> tuple[float, ...]:
    if name == "Poisson":
        return (float(params["lambda"]),)
    if name == "Negative Binomial":
        return (float(params["n"]), float(params["p"]))
    if name == ZIP_NAME:
        return (float(params["pi"]), float(params["lambda"]))
    if name == ZINB_NAME:
        return (float(params["pi"]), float(params["n"]), float(params["p"]))
    ordered = sorted(
        (int(k.split("_")[1]), v) for k, v in params.items() if k.startswith("param_")
    )
    return tuple(float(v) for _, v in ordered)


def format_fit_params(fit: DistributionFitResult) -> str:
    """Human-readable parameter string for a fitted distribution."""
    if not fit.params:
        return "—"
    parts = []
    for key, val in fit.params.items():
        label = _PARAM_LABELS.get(key, key)
        parts.append(f"{label}={float(val):.4g}")
    return f"{fit.name}: " + ", ".join(parts)


def best_fit(results: list[DistributionFitResult]) -> DistributionFitResult | None:
    """Return the best converged fit by AIC, or None."""
    valid = [r for r in results if r.converged and r.aic < float("inf")]
    if not valid:
        return None
    return min(valid, key=lambda r: r.aic)
