"""Sidebar navigation layout and workflow step keys."""

from __future__ import annotations

from dataclasses import dataclass

STEP_DATA = "data"
STEP_OUTLIERS = "outliers"
STEP_CLUSTERS = "clusters"
STEP_PRODUCTS = "products"
STEP_STATISTICS = "statistics"
STEP_TIME_SERIES = "time_series"
STEP_DISTRIBUTIONS = "distributions"
STEP_MONTE_CARLO = "monte_carlo"
STEP_COMPARISON = "comparison"
STEP_EXPORT = "export"
STEP_SETTINGS = "settings"


@dataclass(frozen=True)
class NavEntry:
    label: str
    key: str | None  # None = section header


NAV_ENTRIES: list[NavEntry] = [
    NavEntry("PREPARAÇÃO", None),
    NavEntry("1. Carregar Dados", STEP_DATA),
    NavEntry("2. Outliers", STEP_OUTLIERS),
    NavEntry("3. Clusters de Giro", STEP_CLUSTERS),
    NavEntry("ANÁLISE", None),
    NavEntry("4. Produtos", STEP_PRODUCTS),
    NavEntry("5. Estatísticas por SKU", STEP_STATISTICS),
    NavEntry("6. Série Temporal", STEP_TIME_SERIES),
    NavEntry("7. Distribuições", STEP_DISTRIBUTIONS),
    NavEntry("8. Simulação Monte Carlo", STEP_MONTE_CARLO),
    NavEntry("9. Comparação", STEP_COMPARISON),
    NavEntry("SAÍDA", None),
    NavEntry("10. Exportação", STEP_EXPORT),
    NavEntry("11. Configurações", STEP_SETTINGS),
]

WORKFLOW_HINTS: dict[str, str] = {
    STEP_DATA: "Etapa 1 — Importe e valide os dados. Use «Recarregar Dados» após alterar arquivos.",
    STEP_OUTLIERS: "Etapa 2 — Selecione um produto e aplique tratamento de outliers (opcional, por SKU).",
    STEP_CLUSTERS: "Etapa 3 — Defina cortes de demanda/h e agrupe produtos por giro.",
    STEP_PRODUCTS: "Etapa 4 — Liste e selecione produtos por demanda e cluster.",
    STEP_STATISTICS: "Etapa 5 — Estatísticas descritivas clássicas por SKU (média, dispersão, percentis, forma).",
    STEP_TIME_SERIES: "Etapa 6 — Analise série temporal, histograma e FDC do produto selecionado.",
    STEP_DISTRIBUTIONS: "Etapa 7 — Compare ajustes estatísticos por SKU ou cluster agregado.",
    STEP_MONTE_CARLO: "Etapa 8 — Simule trajetórias Monte Carlo para estimar TempQMin até atingir Qmin.",
    STEP_COMPARISON: "Etapa 9 — Compare série original × tratada após outliers.",
}

NEXT_STEP: dict[str, tuple[str, str]] = {
    STEP_DATA: (STEP_OUTLIERS, "Próximo: Outliers →"),
    STEP_OUTLIERS: (STEP_CLUSTERS, "Próximo: Clusters →"),
    STEP_CLUSTERS: (STEP_PRODUCTS, "Próximo: Produtos →"),
    STEP_PRODUCTS: (STEP_STATISTICS, "Próximo: Estatísticas →"),
    STEP_STATISTICS: (STEP_TIME_SERIES, "Próximo: Série Temporal →"),
    STEP_TIME_SERIES: (STEP_DISTRIBUTIONS, "Próximo: Distribuições →"),
    STEP_DISTRIBUTIONS: (STEP_MONTE_CARLO, "Próximo: Simulação Monte Carlo →"),
    STEP_MONTE_CARLO: (STEP_COMPARISON, "Próximo: Comparação →"),
    STEP_COMPARISON: (STEP_EXPORT, "Próximo: Exportação →"),
}
