# Turnover Score (Índice de Giro)

## Purpose

Order K-Means cluster IDs into Baixo / Médio / Alto Giro labels.

## Formula

```
TurnoverScore = 0.40 × mean_hourly_demand
              + 0.35 × non_zero_hour_ratio
              + 0.25 × p95_hourly_demand
```

## Usage (threshold mode — default)

- **Alto Giro:** `mean_hourly_demand >= cluster_alto_cutoff` (default 2.0)
- **Baixo Giro:** `mean_hourly_demand < cluster_baixo_cutoff` (default 0.5)
- **Médio Giro:** between the two cutoffs

Adjust cutoffs in **Clusters de Giro** screen and click **Recalcular Clusters**.

## Usage (K-Means mode — optional)

Set `clustering_method` to `"kmeans"` in config for legacy behavior.
