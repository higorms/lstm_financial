# 📋 Planejamento — Modelo LSTM de Classificação Binária para Previsão de Direção de Preços

> **Responsável:** Etapas 1, 2 e 3  
> **Escopo:** Coleta de dados, Feature Engineering, Modelagem LSTM e Salvamento do modelo  
> **Data:** Abril/2026

---

## 0. Decisões Estratégicas de Design

| Decisão | Escolha | Justificativa |
|---|---|---|
| **Ativo** | `PETR4.SA` (Petrobras PN) | Ação brasileira com altíssima liquidez, forte presença institucional, série diária disponível desde ~2000 (>6.000 pregões) e comportamento influenciado por múltiplos fatores (commodities, câmbio, política), o que enriquece a modelagem. |
| **Granularidade** | `1d` (diária) | O yfinance limita dados intraday a no máximo 730 dias (para 1h). Com granularidade diária temos **+25 anos de histórico**, o que é fundamental para treinar uma LSTM robusta e ter um período de teste estatisticamente significativo. |
| **Tipo de problema** | Classificação binária | Target `1` = retorno do próximo período ≥ 0; Target `0` = retorno do próximo período < 0. |
| **Framework** | TensorFlow / Keras | Ecossistema maduro, fácil serialização (SavedModel/H5), API de alto nível concisa que facilita a leitura pelo colega na etapa de deploy, e compatibilidade direta com FastAPI/Flask via `tf.keras.models.load_model()`. |

---

## 1. Coleta e Pré-processamento dos Dados

### 1.1 Coleta
- Download via `yfinance` do ativo `PETR4.SA` com o maior range disponível (`period="max"`, granularidade `1d`).
- Colunas esperadas: `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.
- Download de séries auxiliares para features exógenas:
  - **Petróleo Brent** (`BZ=F`) — principal driver do ativo.
  - **Dólar/Real** (`USDBRL=X`) — exposição cambial.
  - **Ibovespa** (`^BVSP`) — contexto de mercado.

### 1.2 Limpeza
- Remoção de dias sem negociação / valores nulos.
- Tratamento de *stock splits* e proventos (usar `Adj Close` como base).
- Alinhamento temporal entre as séries (inner join por data).
- Verificação de outliers via Z-score e IQR — sinalizar mas **não remover** (em séries financeiras, outliers carregam informação).

### 1.3 Construção do Target
```
target = 1  se  (Close[t+1] - Close[t]) / Close[t] >= 0
target = 0  caso contrário
```
- Verificar balanceamento de classes; se desbalanceado, considerar `class_weight` no treinamento e/ou SMOTE apenas no treino.

---

## 2. Feature Engineering

> **Filosofia:** Criar um feature set rico e diversificado para que a LSTM tenha múltiplas "visões" da dinâmica do mercado. Features serão organizadas em **7 famílias**.

### 2.1 Features de Retorno e Volatilidade

| Feature | Descrição |
|---|---|
| `log_return` | Log-retorno: `ln(Close[t] / Close[t-1])` |
| `log_return_Nд` | Log-retorno acumulado em N dias (N = 2, 5, 10, 21) |
| `realized_vol_N` | Desvio-padrão rolling dos log-retornos (janelas 5, 10, 21, 63) |
| `garman_klass_vol` | Volatilidade Garman-Klass (usa OHLC) — mais eficiente que close-to-close |
| `parkinson_vol` | Volatilidade de Parkinson (usa High-Low) |
| `vol_ratio` | Razão entre vol curta (5d) e vol longa (21d) — captura regime de vol |

### 2.2 Indicadores Técnicos Clássicos

| Feature | Descrição |
|---|---|
| `RSI_14` | Relative Strength Index (14 períodos) |
| `RSI_7` | RSI mais curto para capturar momentum recente |
| `MACD`, `MACD_signal`, `MACD_hist` | Moving Average Convergence Divergence |
| `BB_upper`, `BB_lower`, `BB_pct` | Bollinger Bands (20, 2σ) e %B |
| `SMA_ratios` | Razões `Close/SMA_N` para N = 5, 10, 20, 50, 200 |
| `EMA_ratios` | Razões `Close/EMA_N` para N = 9, 21, 50 |
| `ATR_14` | Average True Range normalizado |
| `ADX_14` | Average Directional Index — força da tendência |
| `CCI_20` | Commodity Channel Index |
| `Williams_%R` | Williams %R (14 períodos) |
| `Stochastic_K`, `Stochastic_D` | Oscilador Estocástico |
| `OBV` | On-Balance Volume (normalizado) |
| `MFI_14` | Money Flow Index |
| `VWAP_ratio` | Razão Close/VWAP aproximado |
| `Ichimoku_*` | Tenkan-sen, Kijun-sen, Senkou Span A/B (posições relativas ao Close) |

> Biblioteca sugerida: `ta` (Technical Analysis Library) ou `ta-lib`.

### 2.3 Features Baseadas em Volume

| Feature | Descrição |
|---|---|
| `volume_sma_ratio` | `Volume / SMA(Volume, 20)` — volume relativo |
| `volume_change` | Variação percentual do volume |
| `volume_price_trend` | Correlação rolling entre variação de preço e volume (21d) |
| `accumulation_distribution` | Indicador A/D (acumulação/distribuição) |

### 2.4 Features de Microestrutura e Candlestick

| Feature | Descrição |
|---|---|
| `body_ratio` | `(Close - Open) / (High - Low)` — proporção do corpo do candle |
| `upper_shadow` | `(High - max(Open,Close)) / (High - Low)` |
| `lower_shadow` | `(min(Open,Close) - Low) / (High - Low)` |
| `gap` | `Open[t] / Close[t-1] - 1` — gap de abertura |
| `intraday_range` | `(High - Low) / Close` — amplitude intraday normalizada |

### 2.5 Features Exógenas

| Feature | Descrição |
|---|---|
| `brent_return` | Log-retorno diário do Petróleo Brent |
| `brent_return_5d` | Log-retorno acumulado 5 dias do Brent |
| `usdbrl_return` | Log-retorno do câmbio USD/BRL |
| `usdbrl_return_5d` | Log-retorno acumulado 5 dias do câmbio |
| `ibov_return` | Log-retorno diário do Ibovespa |
| `ibov_return_5d` | Log-retorno acumulado 5 dias |
| `correlation_brent_5d` | Correlação rolling 5d entre PETR4 e Brent |
| `beta_ibov_21d` | Beta rolling 21d contra o Ibovespa |

### 2.6 Features Temporais / Calendário

| Feature | Descrição |
|---|---|
| `day_of_week` | Dia da semana (sin/cos encoding cíclico) |
| `month` | Mês (sin/cos encoding cíclico) |
| `is_month_start` | Flag: primeiro pregão do mês |
| `is_month_end` | Flag: último pregão do mês |
| `is_quarter_end` | Flag: último pregão do trimestre |
| `days_since_year_start` | Dias corridos desde o início do ano (normalizado) |

### 2.7 ⭐ Features via Decomposição Wavelet

> A Transformada Wavelet permite decompor a série temporal em componentes de diferentes frequências (escalas), separando tendência de longo prazo, ciclos de médio prazo e ruído de alta frequência. Isso fornece à LSTM informações que médias móveis simples não conseguem capturar.

**Abordagem:**

1. **Wavelet escolhida:** Daubechies `db4` (boa para séries financeiras — suave e com suporte compacto).
2. **Decomposição:** Aplicar DWT (Discrete Wavelet Transform) via `pywt` nos log-retornos usando decomposição multinível (4 níveis).
3. **Componentes extraídas:**

| Feature | Descrição |
|---|---|
| `wavelet_approx_4` | Coeficiente de aproximação nível 4 — tendência de longo prazo |
| `wavelet_detail_1` | Coeficiente de detalhe nível 1 — ruído / alta frequência |
| `wavelet_detail_2` | Coeficiente de detalhe nível 2 — ciclos de curto prazo |
| `wavelet_detail_3` | Coeficiente de detalhe nível 3 — ciclos de médio prazo |
| `wavelet_detail_4` | Coeficiente de detalhe nível 4 — ciclos de longo prazo |
| `wavelet_energy_ratio` | Razão da energia entre alta e baixa frequência |
| `wavelet_denoised` | Série reconstruída sem o nível de detalhe 1 (denoised) |
| `wavelet_denoised_return` | Log-retorno da série denoised |
| `wavelet_trend_strength` | Razão `approx_4 / std(original)` — força da tendência filtrada |

4. **Aplicação rolling:** Para evitar *look-ahead bias*, a decomposição será aplicada em **janelas rolling** (ex: últimos 60 dias), usando apenas dados passados para calcular a feature no instante `t`.

5. **Wavelet adicional — MODWT:** Considerar também a Maximal Overlap Discrete Wavelet Transform (MODWT), que não exige que o comprimento da série seja potência de 2 e produz coeficientes alinhados temporalmente.

---

## 3. Preparação dos Dados para o Modelo

### 3.1 Divisão Temporal dos Dados

> ⚠️ **Sem shuffle.** Em séries temporais a divisão é sempre cronológica.

| Conjunto | Proporção | Finalidade |
|---|---|---|
| **Treino** | ~70% | Aprendizado dos parâmetros |
| **Validação** | ~15% | Ajuste de hiperparâmetros e early stopping |
| **Teste** | ~15% | Avaliação final + Backtest |

### 3.2 Normalização

- Aplicar `StandardScaler` ou `MinMaxScaler` **fitado apenas no conjunto de treino**.
- Transformar validação e teste com o scaler do treino (evitar data leakage).
- Salvar o scaler junto com o modelo para uso no deploy.

### 3.3 Construção das Sequências (Janela Temporal)

- Criar janelas deslizantes de tamanho `T` (hiperparâmetro — testaremos `T ∈ {30, 60, 90}`).
- Cada amostra: tensor `(T, num_features)` → label `{0, 1}`.
- Implementar via função customizada ou `tf.keras.utils.timeseries_dataset_from_array`.

#### Shape do Tensor de Entrada (3D)

A LSTM **não aceita DataFrames pandas**. Os dados devem ser convertidos em arrays NumPy com **3 dimensões**:

```
X.shape = (samples, timesteps, features)
```

| Dimensão | Significado | Valor esperado |
|---|---|---|
| `samples` | Nº de janelas deslizantes geradas | ~4.000+ (depende do tamanho da série − janela T) |
| `timesteps` | Tamanho da janela temporal `T` | 30, 60 ou 90 (hiperparâmetro) |
| `features` | Nº de features por timestep | ~70–80 (todas as 7 famílias de features) |

**Exemplo concreto:** com 75 features e janela de 60 dias → cada amostra tem shape `(60, 75)`. O dataset de treino completo terá shape `(~3500, 60, 75)`.

O target `y` será um vetor **1D**: shape `(samples,)` com valores `0` ou `1`.

---

## 4. Arquitetura do Modelo LSTM

### 4.1 Arquitetura Base (ponto de partida)

```
Input (T, num_features)
    │
    ▼
LSTM (128 units, return_sequences=True)
    │ + BatchNormalization + Dropout(0.3)
    ▼
LSTM (64 units, return_sequences=False)
    │ + BatchNormalization + Dropout(0.3)
    ▼
Dense (32, activation='relu')
    │ + Dropout(0.2)
    ▼
Dense (1, activation='sigmoid')
    │
    ▼
Output: P(retorno ≥ 0)
```

### 4.2 Processo de Experimentação (Passo-a-Passo)

O desenvolvimento do modelo seguirá um processo **incremental e comparativo** em 5 etapas. Cada etapa gera um modelo candidato avaliado nas mesmas métricas de classificação (AUC-ROC, F1) **e financeiras** (Sharpe Ratio via backtest no conjunto de validação). Ao final, o melhor candidato é selecionado, retreinado e submetido ao backtest final.

> 📊 **Observabilidade:** Todos os experimentos de todas as etapas serão registrados no **MLflow** (métricas, parâmetros, artefatos e backtest). As curvas de treinamento serão visualizadas no **TensorBoard**. Veja detalhes na seção 4.4.

---

#### **Etapa A — Baseline LSTM (Arquitetura Base)**

1. Treinar a arquitetura base da seção 4.1 com hiperparâmetros fixos razoáveis (lr=0.001, batch=64, T=60).
2. Registrar métricas de classificação no conjunto de validação.
3. **Executar backtest no conjunto de validação** e registrar métricas financeiras.
4. Logar tudo no MLflow (run: `baseline_lstm`).
5. Este resultado serve como **referência mínima** — todos os modelos seguintes devem superá-lo.

---

#### **Etapa B — Teste de Arquiteturas Alternativas**

Treinar **4 variações arquiteturais**, cada uma mantendo os mesmos hiperparâmetros da Etapa A para garantir comparação justa:

| # | Arquitetura | Modificação em relação à base | Hipótese |
|---|---|---|---|
| B1 | **Attention LSTM** | Adiciona camada de Self-Attention após a 1ª LSTM (`return_sequences=True` em ambas as LSTMs → Attention → GlobalAveragePooling → Dense) | A atenção permite focar nos timesteps mais informativos, melhorando a captura de eventos pontuais relevantes |
| B2 | **Conv1D + LSTM** | Adiciona `Conv1D(64, kernel=3) → MaxPool1D → BatchNorm` antes da 1ª LSTM | A convolução extrai padrões locais (ex: formações de candle) antes da LSTM capturar dependências longas |
| B3 | **Bidirectional LSTM** | Substitui as duas camadas LSTM por `Bidirectional(LSTM(...))` | Captura padrões tanto "passado→futuro" quanto "futuro→passado" dentro da janela (a janela já é passada, então não há look-ahead) |
| B4 | **LSTM + GRU Híbrido** | 1ª camada LSTM (128) + 2ª camada GRU (64) | GRU é mais leve e pode generalizar melhor na camada final, reduzindo overfitting |

**Resultado:** tabela comparativa com métricas de classificação **e financeiras (backtest)** de todas as variações. Selecionar a **melhor arquitetura** (ou as 2 melhores) para a próxima etapa.

> 💡 O backtest nas etapas A e B permite identificar se uma arquitetura com AUC-ROC ligeiramente inferior pode gerar retornos superiores (ex: acertando mais nos dias de maior volatilidade). A decisão da melhor arquitetura será baseada no **conjunto** de métricas, não apenas em uma.

---

#### **Etapa C — Otimização de Hiperparâmetros (Optuna)**

Com a arquitetura vencedora da Etapa B, executar otimização via **Optuna** (~50-100 trials):

| Hiperparâmetro | Espaço de busca |
|---|---|
| Janela temporal `T` | `{30, 45, 60, 90}` |
| Nº de unidades LSTM camada 1 | `[64, 256]` (step 32) |
| Nº de unidades LSTM camada 2 | `[32, 128]` (step 32) |
| Nº de unidades Dense | `[16, 64]` (step 16) |
| Taxa de dropout (LSTM) | `[0.1, 0.5]` |
| Taxa de dropout (Dense) | `[0.1, 0.4]` |
| Learning rate | `[1e-4, 1e-2]` (log scale) |
| Batch size | `{32, 64, 128}` |
| Usar BatchNormalization | `{True, False}` |

**Objetivo do Optuna:** maximizar `AUC-ROC` no conjunto de validação.

**Resultado:** melhor combinação de hiperparâmetros. Executar backtest na validação com o melhor trial para confirmar melhoria financeira.

> 📊 O Optuna será integrado ao MLflow: cada trial será registrado como um nested run, permitindo visualizar a evolução da busca na UI do MLflow.

---

#### **Etapa D — Modelo Final + Validação Cruzada Temporal**

1. Retreinar o modelo com a melhor arquitetura + melhores hiperparâmetros.
2. Aplicar **Walk-Forward Validation** (validação cruzada temporal): dividir o histórico em 3-5 folds sequenciais para confirmar que o desempenho é consistente ao longo de diferentes regimes de mercado.
3. Avaliar no **conjunto de teste** (nunca visto) → métricas de classificação finais.
4. Logar modelo final no MLflow como **modelo registrado** (`mlflow.register_model`).
5. Este é o modelo que será salvo para deploy.

---

#### **Etapa E — Backtest Final Detalhado**

1. Executar o `Backtester` no **conjunto de teste** com o modelo final da Etapa D.
2. Gerar relatório completo com todas as métricas financeiras (Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor, Calmar).
3. Produzir gráficos finais:
   - Equity Curve: Estratégia vs. Buy & Hold
   - Drawdown ao longo do tempo
   - Distribuição dos retornos diários da estratégia
   - Heatmap de retornos mensais
   - Sinais do modelo sobrepostos ao gráfico de preços
4. Logar todos os gráficos e métricas como artefatos no MLflow.
5. Gerar relatório final consolidado (HTML/Markdown) para apresentação aos professores.

---

### 4.3 Configuração de Treinamento (comum a todas as etapas)

| Parâmetro | Configuração |
|---|---|
| **Loss** | `binary_crossentropy` |
| **Optimizer** | Adam com learning rate scheduling (`ReduceLROnPlateau(factor=0.5, patience=5)`) |
| **Epochs** | Até 200 com `EarlyStopping(patience=15, restore_best_weights=True)` |
| **Class weight** | Ajustar se classes desbalanceadas (`sklearn.utils.class_weight.compute_class_weight`) |
| **Callbacks** | `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` |

### 4.4 Observabilidade dos Experimentos

#### 4.4.1 MLflow — Rastreamento Central de Experimentos

O **MLflow Tracking** será a ferramenta principal de observabilidade. Todos os experimentos (Etapas A–E) serão registrados em um servidor MLflow local.

**Estrutura de organização no MLflow:**

```
MLflow Experiment: "lstm_financial_petr4"
│
├── Run: etapa_A_baseline
│   ├── Params: lr, batch_size, T, architecture="baseline", ...
│   ├── Metrics: auc_roc, f1, accuracy, log_loss, sharpe, sortino, max_drawdown, ...
│   ├── Artifacts: modelo .keras, confusion_matrix.png, equity_curve.png
│   └── Tags: stage="A", status="baseline"
│
├── Run: etapa_B1_attention
│   ├── (mesma estrutura)
│   └── Tags: stage="B", variant="attention"
├── Run: etapa_B2_conv1d_lstm
├── Run: etapa_B3_bidirectional
├── Run: etapa_B4_lstm_gru
│
├── Run: etapa_C_optuna (parent)
│   ├── Nested Run: trial_001
│   ├── Nested Run: trial_002
│   └── ... (50-100 trials)
│
├── Run: etapa_D_modelo_final
│   ├── Artifacts: modelo final, walk_forward_results.json
│   └── → Modelo Registrado: "lstm_petr4_production"
│
└── Run: etapa_E_backtest_final
    ├── Metrics: todas as métricas financeiras
    └── Artifacts: relatório completo, todos os gráficos
```

**O que será logado em cada run:**

| Categoria | Itens |
|---|---|
| **Parâmetros** | Arquitetura, T, lr, batch_size, dropout, nº units, class_weight |
| **Métricas de classificação** | AUC-ROC, F1, Precision, Recall, Accuracy, Log Loss |
| **Métricas financeiras (backtest)** | Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor, Calmar, Retorno Acumulado |
| **Artefatos** | Modelo (.keras), scaler (.pkl), confusion_matrix.png, equity_curve.png, classification_report.txt |
| **Tags** | Etapa (A/B/C/D/E), variante, status |

> 🎯 **Para os professores:** a UI do MLflow permitirá comparar lado a lado todas as variações com filtros, ordenação e gráficos interativos — ideal para a apresentação.

#### 4.4.2 TensorBoard — Monitoramento do Treinamento

O **TensorBoard** será usado como complemento para visualizar o comportamento **durante** o treinamento (epoch a epoch):

| Visualização | Finalidade |
|---|---|
| **Loss curves** (train vs. val) | Detectar overfitting / underfitting |
| **Accuracy curves** (train vs. val) | Acompanhar convergência |
| **Learning rate** | Verificar atuação do `ReduceLROnPlateau` |
| **Histograma de pesos** | Verificar se há vanishing/exploding gradients |

**Integração:** callback `tf.keras.callbacks.TensorBoard(log_dir="logs/{run_name}")` em todos os treinamentos. Cada etapa/variação terá seu próprio subdiretório, permitindo sobreposição de curvas para comparação visual.

#### 4.4.3 Tabela Comparativa Consolidada

Ao final de todas as etapas, será gerada automaticamente uma **tabela-resumo** consolidando métricas de classificação e financeiras de todos os modelos:

```
| Modelo              | AUC-ROC | F1    | Sharpe | Sortino | MaxDD   | WinRate | RetAcum |
|---------------------|---------|-------|--------|---------|---------|---------|---------|
| A - Baseline        | 0.XX    | 0.XX  | X.XX   | X.XX    | -XX.X%  | XX.X%   | XX.X%   |
| B1 - Attention      | ...     | ...   | ...    | ...     | ...     | ...     | ...     |
| B2 - Conv1D+LSTM    | ...     | ...   | ...    | ...     | ...     | ...     | ...     |
| B3 - Bidirectional  | ...     | ...   | ...    | ...     | ...     | ...     | ...     |
| B4 - LSTM+GRU       | ...     | ...   | ...    | ...     | ...     | ...     | ...     |
| C - Optuna Best     | ...     | ...   | ...    | ...     | ...     | ...     | ...     |
| D - Final (teste)   | ...     | ...   | ...    | ...     | ...     | ...     | ...     |
| E - Backtest Final  | —       | —     | ...    | ...     | ...     | ...     | ...     |
```

---

## 5. Avaliação do Modelo

### 5.1 Métricas de Classificação

| Métrica | Motivo |
|---|---|
| **Accuracy** | Baseline de comparação |
| **Precision / Recall / F1-Score** | Visão balanceada por classe |
| **AUC-ROC** | Qualidade geral do ranking de probabilidades |
| **Log Loss** | Calibração das probabilidades |
| **Matriz de Confusão** | Visualização dos erros |
| **Classification Report** | Resumo completo |

### 5.2 Métricas Financeiras (via Backtest)

| Métrica | Descrição |
|---|---|
| **Retorno Acumulado** | Retorno total da estratégia vs. Buy & Hold |
| **Retorno Anualizado** | CAGR da estratégia |
| **Sharpe Ratio** | Retorno ajustado ao risco (risk-free = Selic) |
| **Sortino Ratio** | Como Sharpe, mas penaliza só volatilidade negativa |
| **Max Drawdown** | Maior queda do pico ao vale |
| **Win Rate** | % de trades com retorno positivo |
| **Profit Factor** | Soma dos ganhos / soma das perdas |
| **Calmar Ratio** | CAGR / Max Drawdown |

### 5.3 Classe de Backtest

Implementar uma classe `Backtester` com a seguinte estrutura:

```
class Backtester:
    - __init__(predictions, actual_returns, dates, initial_capital, transaction_cost)
    - run()                → executa simulação dia a dia
    - get_equity_curve()   → retorna série do patrimônio ao longo do tempo
    - get_metrics()        → retorna dict com todas as métricas financeiras
    - plot_results()       → gráfico: equity curve da estratégia vs Buy & Hold
    - plot_drawdown()      → gráfico do drawdown ao longo do tempo
    - generate_report()    → relatório completo em texto/markdown
```

**Regras da simulação:**
- Quando modelo prevê `1` → posição comprada (long).
- Quando modelo prevê `0` → fora do mercado (cash).
- Custo de transação configurável (default: 0.03% por operação — taxa B3).
- Sem alavancagem. Capital fixo investido 100% quando long.

---

## 6. Salvamento e Exportação

| Artefato | Formato | Descrição |
|---|---|---|
| Modelo treinado | `.keras` (SavedModel) | Modelo completo com pesos e arquitetura |
| Scaler | `.pkl` (joblib) | Objeto scaler fitado no treino |
| Configuração de features | `.json` | Lista ordenada das features, janela `T`, e parâmetros |
| Métricas | `.json` | Métricas de classificação e financeiras do backtest |
| Relatório do Backtest | `.html` ou `.md` | Relatório visual com gráficos |

---

## 7. Estrutura de Pastas Proposta

```
lstm_financial/
├── README.md
├── PLANEJAMENTO.md
├── requirements.txt
├── notebooks/
│   ├── 01_coleta_dados.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modelagem_lstm.ipynb
│   ├── 04_avaliacao_backtest.ipynb
│   └── 05_backtest_final.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py          # Coleta via yfinance
│   │   └── preprocessor.py       # Limpeza e alinhamento
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technical.py          # Indicadores técnicos
│   │   ├── volatility.py         # Features de volatilidade
│   │   ├── wavelet.py            # Decomposição Wavelet
│   │   ├── exogenous.py          # Features exógenas
│   │   ├── temporal.py           # Features de calendário
│   │   ├── microstructure.py     # Candlestick / microestrutura
│   │   └── pipeline.py           # Orquestra toda a eng. de features
│   ├── model/
│   │   ├── __init__.py
│   │   ├── builder.py            # Construção da arquitetura LSTM
│   │   ├── trainer.py            # Loop de treinamento + integração MLflow/TensorBoard
│   │   └── tuner.py              # Otimização de hiperparâmetros (Optuna + MLflow)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Métricas de classificação
│   │   └── backtester.py         # Classe Backtester
│   └── utils/
│       ├── __init__.py
│       ├── serialization.py      # Salvamento de modelo/scaler/config
│       └── mlflow_utils.py       # Helpers de logging no MLflow
├── models/                        # Artefatos salvos
│   ├── model.keras
│   ├── scaler.pkl
│   ├── feature_config.json
│   └── metrics.json
├── data/                          # Dados brutos e processados
│   ├── raw/
│   └── processed/
├── logs/                          # Logs do TensorBoard
├── mlruns/                        # Dados do MLflow Tracking (local)
└── reports/                       # Relatórios e gráficos do backtest
    ├── etapa_A_baseline/
    │   ├── equity_curve.png
    │   ├── drawdown.png
    │   ├── confusion_matrix.png
    │   ├── classification_report.txt
    │   └── backtest_metrics.json
    ├── etapa_B1_attention/
    │   └── (mesma estrutura)
    ├── etapa_B2_conv1d_lstm/
    ├── etapa_B3_bidirectional/
    ├── etapa_B4_lstm_gru/
    ├── etapa_C_optuna_best/
    ├── etapa_D_modelo_final/
    ├── etapa_E_backtest_final/
    │   ├── equity_curve.png
    │   ├── drawdown.png
    │   ├── retornos_distribuicao.png
    │   ├── heatmap_retornos_mensais.png
    │   ├── sinais_sobre_preco.png
    │   ├── backtest_metrics.json
    │   └── relatorio_final.md
    └── tabela_comparativa.md       # Resumo de todos os modelos
```

### Formas de Acessar os Backtests

| Canal | Quando usar | Vantagem |
|---|---|---|
| **MLflow UI** (`mlflow ui` → `localhost:5000`) | Comparar modelos lado a lado, filtrar, ordenar | Interativo, gráficos clicáveis, visão consolidada |
| **Pasta `reports/`** | Acesso rápido a gráficos e métricas sem subir servidor | Sempre disponível, versionável no Git |
| **Notebooks** | Reproduzir ou ajustar visualizações | Execução interativa, flexível |

---

## 8. Dependências Previstas

```
yfinance
pandas
numpy
scikit-learn
tensorflow
tensorboard
mlflow
pywt            # PyWavelets — decomposição wavelet
ta              # Technical Analysis library
optuna          # Otimização de hiperparâmetros
matplotlib
seaborn
plotly
joblib
```

---

## 9. Checklist de Entrega (minha parte)

- [ ] Dados coletados e pré-processados
- [ ] Feature engineering completa (7 famílias, incluindo wavelet)
- [ ] MLflow configurado e operacional
- [ ] Etapa A: baseline treinado + backtest na validação
- [ ] Etapa B: 4 variações treinadas + backtest na validação de cada uma
- [ ] Etapa C: otimização Optuna concluída (trials logados no MLflow)
- [ ] Etapa D: modelo final retreinado + Walk-Forward Validation
- [ ] Etapa E: backtest final no teste + relatório completo com gráficos
- [ ] Tabela comparativa consolidada de todos os modelos
- [ ] Modelo e artefatos salvos em `/models`
- [ ] Código modularizado em `/src`
- [ ] Notebooks documentados em `/notebooks`
- [ ] `requirements.txt` atualizado
- [ ] README.md atualizado com instruções
