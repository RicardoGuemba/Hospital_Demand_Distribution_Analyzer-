# Métricas de Ajuste da FDC (CDF)

**Hospital Demand Distribution Analyzer** — documentação das métricas usadas para comparar a Função de Distribuição Acumulada empírica (FDC/ECDF) com CDFs teóricas ajustadas.

---

## 1. Visão geral

O sistema estima, para cada SKU ou cluster agregado, qual distribuição de probabilidade (Poisson, Binomial Negativa, Gamma, etc.) melhor descreve a demanda horária. A comparação é feita visualmente (FDC empírica × CDF ajustada) e numericamente por um conjunto de métricas.

**Valores válidos:** demanda ≥ 0 (zeros incluídos por padrão na grade horária).

**Critério principal de ranking:** AIC (menor é melhor).

**Implementação central:** `src/hospital_demand_analyzer/statistics/metrics.py`

---

## 2. FDC empírica (ECDF)

### Definição

A FDC empírica é a função escalonada que indica a proporção de observações com demanda menor ou igual a cada valor x:

```
F_n(x) = (1/n) × |{ i : x_i ≤ x }|
```

### Intuição

É a referência visual “real” dos dados. Nos gráficos de Série Temporal e Distribuições, a curva **FDC empírica** (ou ECDF) mostra como a demanda se acumula ao longo dos quantis observados.

### Interpretação

Não é uma métrica de comparação numérica; serve como baseline contra a qual as CDFs teóricas são sobrepostas.

### Onde aparece

- Gráficos **FDC Empírica** e **ECDF + CDFs** (etapas 6 e 7)
- Cálculo: `statistics/ecdf.py` → `compute_ecdf()`

---

## 3. Verossimilhança log(L)

### Definição

Logaritmo da função de verossimilhança: soma dos log-probabilidades dos dados observados sob os parâmetros ajustados do modelo.

```
log L = Σ log f(x_i | θ)
```

onde f é a PMF (discreta) ou PDF (contínua) e θ são os parâmetros estimados.

### Intuição

Mede o quão “prováveis” são os dados observados sob o modelo. Valores maiores (menos negativos) indicam melhor aderência probabilística.

### Interpretação

Não é exibida diretamente na tabela da UI, mas alimenta AIC, AICC e BIC.

### Onde aparece

- `statistics/fitter.py` → campo `log_likelihood` em `DistributionFitResult`

---

## 4. AIC — Critério de Informação de Akaike

### Definição

```
AIC = 2k − 2 log L
```

onde k = número de parâmetros livres do modelo e n = tamanho da amostra.

### Intuição

Equilibra qualidade de ajuste (via log L) e complexidade (penaliza modelos com muitos parâmetros). Evita escolher distribuições excessivamente flexíveis que se ajustam ao ruído.

### Interpretação

**Menor AIC = melhor modelo.** O sistema ordena as distribuições convergentes por AIC crescente e atribui **Rank = 1** ao menor AIC.

Regra prática (comparando ΔAIC):
- ΔAIC ≈ 0–2: suporte similar ao melhor modelo
- ΔAIC 4–7: suporte substancialmente menor
- ΔAIC > 10: suporte fraco

### Onde aparece

- Tabela **7. Distribuições** — colunas AIC e Rank
- Exportação CSV/XLSX — coluna AIC
- Código: `metrics.py` → `compute_aic()`

---

## 5. AICC — AIC corrigido (amostras pequenas)

### Definição

```
AICC = AIC + (2k(k+1)) / (n − k − 1)
```

Aplicado quando n − k − 1 > 0; caso contrário, AICC = AIC.

### Intuição

Corrige o viés do AIC quando a amostra é pequena em relação ao número de parâmetros.

### Interpretação

**Menor AICC = melhor.** Calculado internamente, mas **não exibido na tabela da UI** (disponível em `DistributionFitResult.aicc`).

### Onde aparece

- `metrics.py` → `compute_aicc()`

---

## 6. BIC — Critério de Informação Bayesiano

### Definição

```
BIC = k log(n) − 2 log L
```

### Intuição

Penaliza a complexidade mais fortemente que o AIC (via fator log(n)). Tende a favorecer modelos mais parsimoniosos quando n é grande.

### Interpretação

**Menor BIC = melhor.** Complementa o AIC; divergências entre AIC e BIC sugerem trade-off entre ajuste fino e simplicidade.

### Onde aparece

- Tabela **7. Distribuições** — colunas BIC
- Exportação — coluna BIC
- Código: `metrics.py` → `compute_bic()`

---

## 7. ΔAIC e ΔBIC

### Definição

```
ΔAIC_i = AIC_i − min(AIC)
ΔBIC_i = BIC_i − min(BIC)
```

Distância de cada modelo em relação ao melhor da lista.

### Intuição

Facilita a leitura relativa: o melhor modelo tem Δ = 0; os demais mostram o “custo” de escolher outra distribuição.

### Interpretação

**Menor Δ = melhor** (Δ = 0 para o vencedor).

### Onde aparece

- Tabela **7. Distribuições** — colunas ΔAIC, ΔBIC
- Código: `fitter.py` → `fit_all()` após calcular mínimos

---

## 8. KS — Kolmogorov-Smirnov

### Definição

Estatística D = máximo desvio vertical entre a CDF empírica e a CDF teórica ajustada:

```
D = sup_x | F_n(x) − F(x | θ) |
```

Implementado via `scipy.stats.kstest`.

### Intuição

Teste clássico de aderência de distribuição. Sensível a diferenças em qualquer ponto do eixo x, com peso uniforme.

### Interpretação

**Menor D = melhor aderência da CDF.** Valores altos indicam desvio sistemático entre empírico e teórico. O p-valor associado (ks_pvalue) está disponível internamente, mas a UI exibe apenas a estatística D.

### Onde aparece

- Tabela **7. Distribuições** — coluna KS
- Exportação — coluna KS
- Código: `metrics.py` → `ks_test()`

---

## 9. CvM — Cramér-von Mises

### Definição

Integra o quadrado da diferença entre CDF empírica e teórica em todo o domínio:

```
W² = n ∫ (F_n(x) − F(x|θ))² dF(x|θ)
```

Implementado via `scipy.stats.cramervonmises`.

### Intuição

Semelhante ao KS, mas penaliza desvios em **toda** a curva (não só no ponto de máximo). Mais sensível ao **corpo central** da distribuição.

### Interpretação

**Menor CvM = melhor aderência global da CDF.**

### Onde aparece

- Tabela **7. Distribuições** — coluna CvM
- Exportação — coluna CvM
- Código: `metrics.py` → `cvm_test()`

---

## 10. AD — Anderson-Darling

### Definição

Variante ponderada do teste de aderência, dando **maior peso às caudas** da distribuição:

```
A² = −n − (1/n) Σ (2i−1) [ ln F(x_(i)) + ln(1 − F(x_(n+1−i))) ]
```

onde x_(i) são os dados ordenados. Implementação customizada em `ad_test()` (não restrita à normal).

### Intuição

Útil quando erros nas caudas (demandas altas) são mais críticos operacionalmente do que erros no centro.

### Interpretação

**Menor AD = melhor aderência**, com ênfase nas extremidades.

### Onde aparece

- Tabela **7. Distribuições** — coluna AD
- Exportação — coluna AD
- Código: `metrics.py` → `ad_test()`

---

## 11. Tail Fit Score

### Definição

Média ponderada do erro relativo absoluto entre quantis empíricos e teóricos nas caudas:

```
TailFitScore = Σ(w_P × |Q_teórico(P) − Q_empírico(P)| / Q_empírico(P)) / Σ(w_P)
```

Percentis P ∈ {90, 95, 97.5, 99} com pesos:

| Percentil | Peso |
|-----------|------|
| P90 | 0.15 |
| P95 | 0.25 |
| P97.5 | 0.25 |
| P99 | 0.35 |

Quando Q_empírico(P) = 0, usa-se erro absoluto em vez de relativo.

### Intuição

Mede especificamente a qualidade do ajuste nas **caudas superiores** — relevante para dimensionamento de estoque e níveis de serviço (P95, P99).

### Interpretação

**Menor Tail Score = melhor aderência na cauda.** O sistema também calcula `rank_tail` (ordenação separada por Tail Score).

### Onde aparece

- Tabela **7. Distribuições** — coluna Tail Score
- Exportação — coluna Tail Score
- Código: `metrics.py` → `compute_tail_fit_score()`
- Detalhe: `docs/TAIL_FIT_SCORE.md`

---

## 12. Rank e Rank_tail

### Definição

- **Rank:** posição na ordenação por AIC crescente (1 = melhor AIC)
- **Rank_tail:** posição na ordenação por Tail Fit Score crescente (1 = melhor cauda)

### Intuição

Simplificam a leitura da tabela de ranking quando várias distribuições convergem.

### Interpretação

Rank 1 por AIC = melhor compromisso global; Rank 1 por cauda = melhor ajuste operacional nas extremidades. Podem divergir.

### Onde aparece

- Tabela **7. Distribuições** — coluna Rank
- Exportação — coluna Rank
- Código: `fitter.py` → `fit_all()`

---

## 13. CDF condicional (modo demanda > 0)

### Definição

Quando o usuário desmarca **“Incluir zeros”**, a CDF teórica exibida nos gráficos é renormalizada:

```
F_cond(x) = (F(x) − F(0)) / (1 − F(0))   para x > 0
```

### Intuição

Permite comparar a forma da distribuição **entre horas com consumo positivo**, removendo a massa de probabilidade em zero.

### Interpretação

Não é uma métrica numérica separada; é uma transformação visual e de overlay aplicada quando `positive_only=True`.

### Onde aparece

- Gráficos ECDF/FDC ajustada com sufixo “(apenas demanda > 0)”
- Código: `plots/ecdf_plot.py` → `_conditional_cdf()`
- Histogramas: `plots/histogram_plot.py` → `_conditional_mass()`

---

## 14. Como o sistema escolhe o melhor modelo

1. **Ajuste:** `fit_all()` tenta todas as distribuições do registro (discretas ou contínuas, conforme modo de representação).
2. **Ranking principal:** modelos convergentes ordenados por **AIC** → Rank 1.
3. **Overlay nos gráficos:** `select_fits_for_display()` seleciona até 3 curvas — top AIC e, se `% zero ≥ 10%`, promove Poisson ou Binomial Negativa.
4. **Diagnóstico complementar:** KS, CvM e AD avaliam aderência da CDF ponto a ponto.
5. **Caudas:** Tail Fit Score e `rank_tail` avaliam P90–P99 (crítico para logística hospitalar).

**Aviso exibido na UI:** “Melhor distribuição significa melhor modelo entre as candidatas avaliadas, não a 'verdadeira distribuição' da demanda.”

---

## 15. Tabela-resumo

| Métrica | O que mede | Melhor valor | Coluna na UI |
|---------|-----------|--------------|--------------|
| FDC empírica | Baseline visual | — | Gráfico |
| log L | Probabilidade dos dados | Maior | (interno) |
| AIC | Ajuste × complexidade | **Menor** | AIC, Rank |
| AICC | AIC corrigido (n pequeno) | Menor | (interno) |
| BIC | Parsimônia bayesiana | Menor | BIC |
| ΔAIC | Distância ao melhor AIC | **0** | ΔAIC |
| ΔBIC | Distância ao melhor BIC | 0 | ΔBIC |
| KS | Máx. desvio entre CDFs | Menor | KS |
| CvM | Integral do desvio² | Menor | CvM |
| AD | Desvio² ponderado (caudas) | Menor | AD |
| Tail Score | Erro nos quantis P90–P99 | Menor | Tail Score |
| Rank_tail | Ordem por cauda | 1 | (interno) |

---

## Referências de código

| Arquivo | Responsabilidade |
|---------|-----------------|
| `statistics/metrics.py` | AIC, BIC, KS, CvM, AD, Tail Score |
| `statistics/fitter.py` | Orquestra ajuste e ranking |
| `statistics/ecdf.py` | FDC empírica |
| `plots/ecdf_plot.py` | Visualização FDC/CDF |
| `domain/models.py` | `DistributionFitResult` |
| `ui/views/distributions_view.py` | Tabela de ranking na UI |
