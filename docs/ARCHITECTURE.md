# Architecture

## Layers

- **Presentation** (`ui/`): PySide6 windows, views, widgets
- **Application** (`app/`): AppState, AppController, QThread workers
- **Domain** (`domain/`): dataclass models
- **Infrastructure** (`data/`, `services/`, `exports/`): I/O, cache, time series
- **Statistics** (`statistics/`): descriptive, ECDF, fitting, metrics
- **Clustering** (`clustering/`): features, K-Means
- **Outliers** (`outliers/`): IQR, MAD, Hampel

## Entry Point

- **macOS:** duplo clique em `run_app.command` (ou `./run_app.command` no Terminal)
- Windows: duplo clique em `run_app.bat`
- CLI: `python -m hospital_demand_analyzer`

## Data Flow

1. Import Excel/CSV → normalize → validate → quality report
2. Consolidate SKUs → hourly series with zero-fill
3. Descriptive stats + features → K-Means clustering
4. Distribution fitting → AIC/BIC/GoF/Tail Score ranking
5. Outlier detection → treated series → distribution fitting
6. Time series + distribution views (FDC, histograms)
7. Distribution fitting per SKU (background workers, fit cache)
8. Monte Carlo TempQMin simulation → trajectories and TempQMin stats — see [`SIMULACAO_MONTE_CARLO_TEMQMIN.md`](SIMULACAO_MONTE_CARLO_TEMQMIN.md)
9. Export CSV/XLSX
