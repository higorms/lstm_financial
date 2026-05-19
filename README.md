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
| 3b | Modelagem (regressão): Baseline + 4 variações | `03b_modelagem_lstm_regressao.ipynb` |
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

## Deploy em Produção (EC2)

O deploy foi realizado em uma instância **AWS EC2** usando **Docker Compose**, empacotando a API e suas dependências em contêineres para facilitar atualização e rollback.

```bash
# Subir a stack em background
docker-compose up -d --build

# Verificar status dos serviços
docker-compose ps

# Acompanhar logs
docker-compose logs -f
```

O arquivo `docker-compose.yml` define os serviços, variáveis de ambiente e portas expostas, permitindo reproduzir o ambiente de produção localmente com o mesmo setup.

### Serviços no Docker Compose

| Serviço | Função | Porta | Observações |
|---------|--------|-------|-------------|
| `api` | API FastAPI de regressão | 8000 | Build do `Dockerfile`, escreve logs em `inference_logs` |
| `grafana` | Dashboards e visualização de métricas | 3000 | Lê CSVs de `inference_logs` via plugin Infinity |
| `csv_server` | Servidor Nginx para CSVs | 8081 | Publica `inference_logs` para consumo do Grafana |
| `mlflow` | Tracking de experimentos | 5000 | Usa SQLite e `mlruns` montados do host |

Volumes persistentes:

- `grafana_data`: mantém dashboards e configurações do Grafana
- `inference_logs`: compartilha logs da API entre os serviços

### Endpoints da API

Base URL (local): `http://localhost:8000`

- `GET /health`
	- Verifica disponibilidade do serviço.
	- Exemplo de resposta: `{ "status": "ok" }`
- `GET /predict/regression?date=YYYY-MM-DD`
	- Retorna previsão de retorno e preço de fechamento para a data informada.
	- Campos: `predicted_date`, `predicted_return`, `predicted_close`.
	- Códigos: `200` OK, `400` data inválida, `404` data ausente no dataset, `500` artefatos não carregados.

Exemplo de chamada:

```bash
curl "http://localhost:8000/predict/regression?date=2024-01-15"
```

## Resultados

### Classificação

| Modelo | AUC | Sharpe | Sortino | Retorno Acum. |
|--------|-----|--------|---------|---------------|
| A – Baseline LSTM | 0.523 | 0.89 | — | 119.5% |
| B1 – Attention | 0.516 | — | — | — |
| B2 – Conv1D+LSTM | 0.473 | — | — | — |
| B3 – Bidirectional | 0.542 | — | — | — |
| **B4 – LSTM+GRU (Final)** | **0.463** | **0.99** | **1.21** | **142%** |
| C – Optuna Best | 0.557 | — | — | — |

> O modelo B4 (LSTM+GRU) foi selecionado como modelo final por apresentar as melhores métricas financeiras (Sharpe, Sortino, Calmar, Retorno Acumulado), apesar de AUC modesto.

### Regressão

| Modelo | MAE | RMSE | MAPE | R2 | Direction_Acc |
|--------|-----|------|------|----|---------------|
| A – Baseline | 0.2496 | 0.3477 | 1.8328 | 0.9932 | 47.51 |
| B1 – Attention | 0.2424 | 0.3412 | 1.7857 | 0.9935 | 51.78 |
| B2 – Conv1D+LSTM | 0.2389 | 0.3385 | 1.7758 | 0.9936 | 54.98 |
| B3 – Bidirectional | 0.2588 | 0.3617 | 1.8894 | 0.9927 | 48.75 |
| B4 – LSTM+GRU | 0.2474 | 0.3457 | 1.8307 | 0.9933 | 47.69 |

## Observabilidade

- **MLflow**: rastreamento de todos os experimentos (métricas, parâmetros, artefatos)
- **TensorBoard**: curvas de loss/accuracy durante treinamento
- **Reports**: gráficos e métricas persistidos em `reports/`

## Tecnologias

- Python, TensorFlow/Keras, MLflow, Optuna, PyWavelets
- yfinance, pandas, scikit-learn, ta (Technical Analysis)
- matplotlib, seaborn, plotly
- FastAPI, Docker