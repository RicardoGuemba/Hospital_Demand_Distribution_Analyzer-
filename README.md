# Hospital Demand Distribution Analyzer

Aplicação desktop em **Python / PySide6** para análise estatística da **demanda horária hospitalar** por SKU: importação de dados, estatísticas descritivas, clustering por giro, ajuste de distribuições de probabilidade, detecção de outliers, simulação **Monte Carlo TempQMin** e exportação.

Desenvolvido para apoiar decisões de estoque e logística em ambientes com demanda **intermitente** (muitas horas com consumo zero).

---

## Funcionalidades

| Etapa | Módulo | Descrição |
|-------|--------|-----------|
| 1 | Carregar Dados | Importação Excel/CSV, validação, normalização e grade horária com zero-fill |
| 2 | Outliers | IQR, MAD e Hampel — tratamento não destrutivo por SKU |
| 3 | Clusters de Giro | Agrupamento Alto / Médio / Baixo giro por cortes ou K-Means |
| 4 | Produtos | Listagem, busca e seleção de SKUs |
| 5 | Estatísticas | Descritivas por SKU: média aritmética, E[X] empírico, E[X\|X>0], percentis, forma |
| 6 | Série Temporal | Gráfico horário, histograma e FDC empírica/ajustada |
| 7 | Distribuições | Ajuste Poisson, Binomial Negativa, **ZIP** e **ZINB** (zero-infladas) — ranking AIC/BIC/GoF |
| 8 | Monte Carlo | Simulação de trajetórias de estoque até Qmin; estimativa de **TempQMin** (zeros respeitados conforme FDC ajustada) |
| 9 | CDF por SKU | Ajuste em lote, FDC de todas candidatas, distribuição escolhida (AIC) e parâmetros |
| 10 | Qmin reabastecimento | Qmin proposta = P95 da demanda acumulada no intervalo de reabastecimento (SKUs da sessão) |
| 11 | Exportação | CSV / XLSX |
| 12 | Configurações | Seed, clusters K-Means, diretório de dados |

---

## Requisitos

- **Python 3.11+**
- macOS, Windows ou Linux com display gráfico (PySide6)

---

## Instalação e execução

### macOS (recomendado)

Duplo clique em [`run_app.command`](run_app.command) ou, no Terminal:

```bash
./run_app.command
```

Na primeira execução o script cria `.venv`, instala dependências e abre a interface.

### Windows

Duplo clique em [`run_app.bat`](run_app.bat) ou:

```bat
run_app.bat
```

### Manual

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m hospital_demand_analyzer
```

---

## Dados de entrada

Coloque arquivos **Excel (.xlsx)** ou **CSV** em uma pasta (ex.: `Dados_Demanda/`) com colunas de produto, data/hora e quantidade. Na etapa **1. Carregar Dados**, selecione o diretório.

A aplicação:

- Consolida registros por SKU em **série horária** (intervalo fixo de 1 h)
- **Zero-fill** entre primeira e última observação do SKU
- Trata demanda **≥ 0** (zeros incluídos por padrão)

---

## Simulação Monte Carlo (TempQMin)

A etapa 8 estima o **tempo até o saldo atingir Qmin**, sorteando demanda horária da melhor distribuição ajustada (AIC):

- Passo de simulação: **60 minutos**
- Demanda sorteada: **inteiro ≥ 0** (marginal da FDC ajustada)
- Interpolação fracionária ao cruzar Qmin dentro da hora
- N trajetórias configurável (padrão: 100)

Documentação completa: [`docs/SIMULACAO_MONTE_CARLO_TEMQMIN.md`](docs/SIMULACAO_MONTE_CARLO_TEMQMIN.md)

---

## Arquitetura

```
src/hospital_demand_analyzer/
├── app/           # Estado, controller, workers assíncronos
├── data/          # Importação e validação
├── domain/        # Modelos (série horária, fits, Monte Carlo)
├── statistics/    # Descritiva, ECDF, fitting, Monte Carlo
├── clustering/    # Features e K-Means / cortes
├── outliers/      # IQR, MAD, Hampel
├── plots/         # Matplotlib / canvas Qt
├── ui/            # Views PySide6
└── exports/       # CSV, XLSX, PDF de métricas
```

Detalhes: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Documentação adicional

| Documento | Conteúdo |
|-----------|----------|
| [`docs/METRICAS_CDF.md`](docs/METRICAS_CDF.md) | Métricas de ajuste da FDC (AIC, BIC, KS, Tail Score, …) |
| [`docs/SIMULACAO_MONTE_CARLO_TEMQMIN.md`](docs/SIMULACAO_MONTE_CARLO_TEMQMIN.md) | Algoritmo saldo hora a hora e TempQMin |
| [`docs/TURNOVER_SCORE.md`](docs/TURNOVER_SCORE.md) | Score e clustering por giro |
| [`docs/TAIL_FIT_SCORE.md`](docs/TAIL_FIT_SCORE.md) | Métrica de cauda |
| [`docs/PROJECT_RULES.md`](docs/PROJECT_RULES.md) | Regras de domínio do projeto |

PDF das métricas CDF: [`METRICAS_CDF.pdf`](METRICAS_CDF.pdf) (gerado via `scripts/generate_cdf_metrics_pdf.py`)

---

## Testes

```bash
pytest
```

Suite com testes de importação, estatística, fitting, Monte Carlo, UI funcional e workers.

---

## Stack

| Componente | Uso |
|------------|-----|
| PySide6 | Interface desktop |
| pandas / numpy | Séries temporais e matrizes |
| scipy / statsmodels | Distribuições e testes de aderência |
| scikit-learn | K-Means |
| matplotlib | Gráficos embutidos na UI |

---

## Licença

Uso interno / projeto de análise hospitalar. Consulte o mantenedor para termos de redistribuição.

---

## Autor

[Ricardo Guemba](https://github.com/RicardoGuemba) — **Hospital Demand Distribution Analyzer**
