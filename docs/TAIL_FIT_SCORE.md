# Tail Fit Score

## Definition

Weighted mean of relative absolute errors between empirical and theoretical quantiles at tail percentiles.

## Formula

For percentiles P ∈ {90, 95, 97.5, 99} with weights w_P:

```
TailFitScore = Σ(w_P × |Q_theoretical(P) - Q_empirical(P)| / Q_empirical(P)) / Σ(w_P)
```

## Weights

| Percentile | Weight |
|---|---|
| P90 | 0.15 |
| P95 | 0.25 |
| P97.5 | 0.25 |
| P99 | 0.35 |

Lower score = better tail adherence.

When empirical quantile is zero, absolute error is used instead of relative.
