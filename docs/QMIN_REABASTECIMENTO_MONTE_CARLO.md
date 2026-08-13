# Qmin por Monte Carlo — tempo de reabastecimento

**Hospital Demand Distribution Analyzer** — etapa 11 (Qmin reabastecimento): simulação da **demanda acumulada** até um tempo de reabastecimento arbitrário; **Qmin proposta = P95** dos totais simulados.

---

## 1. Visão geral

Para cada SKU **selecionado na sessão corrente** (via `select_sku` nas etapas 4–8), o app:

1. Usa a **melhor FDC ajustada** (menor AIC no cache `fit_results`).
2. Simula **N trajetórias** de demanda acumulada até o tempo de reabastecimento **T** (padrão **120 min**).
3. Define **Qmin proposta = P95** dos valores de demanda acumulada no fim de cada trajetória.

| Aspecto | Valor |
|---------|-------|
| Passo temporal | **1 hora = 60 min** (`MINUTES_PER_STEP`) |
| Sorteio horário | `sample_demand_from_fit()` — respeita a FDC (inclui zero em discretas) |
| Fração final | Se `T` não é múltiplo de 60 min: última hora contribui `d × (T_restante / 60)` |
| Escopo | Apenas SKUs em `session_sku_order` / `session_selected_skus` |

**Implementação:** `src/hospital_demand_analyzer/statistics/replenishment_qmin.py`

**Interface:** etapa 11 — `ui/views/replenishment_qmin_view.py`

---

## 2. Algoritmo por trajetória

```
D ← 0, t ← 0, T_restante ← T
enquanto T_restante > 0:
    d ← sorteio da FDC do SKU
    se T_restante ≥ 60:
        D ← D + d
        t ← t + 60
        T_restante ← T_restante − 60
    senão:
        D ← D + d × (T_restante / 60)
        t ← T
        T_restante ← 0
```

Cada trajetória registra pontos `(tempo, demanda_acumulada)` internamente; na **UI**, o gráfico segue o estilo Monte Carlo (etapa 8):

- **Eixo Y:** saldo simulado = `Qmin proposta − demanda acumulada(t)`
- **Saldo inicial:** Qmin proposta (P95 da demanda total no intervalo)
- Curvas **decrescem** até `t = T` (tempo de reabastecimento esgotado)
- Linhas horizontais: saldo inicial, ruptura (0), percentis do **saldo final** (Q1, mediana, Q3, P5, P95, P99)

A Qmin proposta (tabela/export) continua sendo **P95 da demanda acumulada** no fim do intervalo.

---

## 3. Qmin proposta e estatísticas

Após **N** trajetórias, forma-se o vetor `[D₁, …, Dₙ]`:

| Métrica | Definição |
|---------|-----------|
| **Qmin proposta** | P95 do vetor |
| Q1 / Mediana / Q3 | P25, P50, P75 |
| P5, P90, P99 | Percentis de cauda |
| Média, desvio, min, max | Estatísticas descritivas clássicas |

---

## 4. Pré-requisitos

- SKU deve ter **ajuste convergente** no cache (etapas 7 ou 10).
- SKU deve ter sido **selecionado pelo usuário** na sessão (lista limpa ao recarregar dados).

---

## 5. Interpretação operacional

A Qmin proposta estima o estoque mínimo necessário para cobrir a **demanda acumulada esperada no ciclo de reabastecimento**, com margem estocástica no **percentil 95**, sob o modelo de demanda ajustado (FDC).

Não substitui políticas de estoque da instituição; é um indicador analítico derivado do melhor ajuste estatístico disponível no app.
