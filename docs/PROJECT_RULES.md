# Project Rules — Hospital Demand Distribution Analyzer

## Development Principles

1. Incremental development — one stage at a time
2. Never modify original data files in `Dados_Demanda/`
3. Separate Presentation / Application / Domain / Infrastructure layers
4. No statistical logic in UI widgets
5. Type hints, small functions, single responsibility
6. Files should stay under ~400 lines unless justified
7. Always show empirical data alongside model results
8. Never claim a fitted distribution is the "true" distribution

## Data Rules

- `Produto` is the SKU primary key
- Hourly aggregation: floor to hour, sum quantities
- Zero-fill only within validated SKU date range
- Preserve `source_file` for traceability

## Statistical Rules

- AIC/BIC comparable only on same observation set
- Tail fit (P90–P99) is critical for policy decisions
- Outlier treatment is non-destructive (keep original)
- Zeros are not automatically outliers

## Clustering Rules

- K-Means with K=3 for operational use
- Labels Alto/Médio/Baixo assigned after sorting by Turnover Score
- Cluster similarity ≠ identical distribution
