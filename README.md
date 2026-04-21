# LSTM Financial — Previsão de Direção de Preços (PETR4.SA)

Projeto de **Machine Learning Engineering** — FIAP Pós-Tech (Tech Challenge Fase 4).

## Objetivo

Modelo LSTM de **classificação binária** para prever a direção (alta/baixa) do preço de fechamento da **PETR4.SA** (Petrobras PN), com pipeline completa de desenvolvimento, avaliação financeira via backtest e deploy em API.

## Estrutura do Projeto

```
lstm_financial/
├── notebooks/          # Notebooks executáveis (01 a 05)
├── src/                # Código modularizado
│   ├── data/           # Coleta e pré-processamento
│   ├── features/       # 7 famílias de features (incl. Wavelet)
│   ├── model/          # Arquiteturas LSTM, treinamento, Optuna
│   ├── evaluation/     # Métricas e Backtester
│   └── utils/          # Serialização, MLflow helpers
├── models/             # Modelo e artefatos salvos
├── data/               # Dados brutos e processados
├── reports/            # Relatórios de backtest por etapa
├── logs/               # TensorBoard logs
└── mlruns/             # MLflow tracking
```

## Etapas de Desenvolvimento

| Etapa | Descrição | Notebook |
|-------|-----------|----------|
| 1 | Coleta e pré-processamento | `01_coleta_dados.ipynb` |
| 2 | Feature Engineering (7 famílias + Wavelet) | `02_feature_engineering.ipynb` |
| 3 | Modelagem: Baseline + 4 variações + Optuna + Modelo Final | `03_modelagem_lstm.ipynb` |
| 4 | Análise comparativa dos modelos | `04_avaliacao_backtest.ipynb` |
| 5 | Backtest final detalhado | `05_backtest_final.ipynb` |

## Instalação e Execução

```bash
# Criar ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar notebooks na ordem (01 → 05)

# Visualizar experimentos no MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Visualizar treinamento no TensorBoard
tensorboard --logdir logs/
```

## Resultados

| Modelo | AUC | Sharpe | Sortino | Retorno Acum. |
|--------|-----|--------|---------|---------------|
| A – Baseline LSTM | 0.523 | 0.89 | — | 119.5% |
| B1 – Attention | 0.516 | — | — | — |
| B2 – Conv1D+LSTM | 0.473 | — | — | — |
| B3 – Bidirectional | 0.542 | — | — | — |
| **B4 – LSTM+GRU (Final)** | **0.463** | **0.99** | **1.21** | **142%** |
| C – Optuna Best | 0.557 | — | — | — |

> O modelo B4 (LSTM+GRU) foi selecionado como modelo final por apresentar as melhores métricas financeiras (Sharpe, Sortino, Calmar, Retorno Acumulado), apesar de AUC modesto.

## Observabilidade

- **MLflow**: rastreamento de todos os experimentos (métricas, parâmetros, artefatos)
- **TensorBoard**: curvas de loss/accuracy durante treinamento
- **Reports**: gráficos e métricas persistidos em `reports/`

## Tecnologias

- Python, TensorFlow/Keras, MLflow, Optuna, PyWavelets
- yfinance, pandas, scikit-learn, ta (Technical Analysis)
- matplotlib, seaborn, plotly

## Equipe

- **Etapas 1-3** (Modelo): Coleta, Feature Engineering, Modelagem LSTM, Backtest
- **Etapas 4-5** (Deploy): API FastAPI/Flask, Docker, Monitoramento
