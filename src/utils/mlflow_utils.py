"""
Helpers de logging no MLflow.

Conforme PLANEJAMENTO.md - Seção 4.4.1:
- Setup do experimento
- Funções auxiliares para logging padronizado
"""

import logging
import os
from typing import Dict, Optional

import mlflow

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT = "lstm_financial_petr4"
DEFAULT_TRACKING_URI = "mlruns"


def setup_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_uri: Optional[str] = None,
) -> str:
    """
    Configura MLflow tracking.

    Returns
    -------
    str
        ID do experimento.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment: '{experiment_name}' (ID: {experiment.experiment_id})")
    return experiment.experiment_id


def log_backtest_and_classification(
    classification_metrics: Dict,
    backtest_metrics: Dict,
    prefix: str = "",
) -> None:
    """
    Loga métricas de classificação e backtest na run ativa.
    """
    for k, v in classification_metrics.items():
        if isinstance(v, (int, float)):
            key = f"{prefix}cls_{k}" if prefix else f"cls_{k}"
            mlflow.log_metric(key, v)

    for k, v in backtest_metrics.items():
        if isinstance(v, (int, float)):
            key = f"{prefix}bt_{k}" if prefix else f"bt_{k}"
            mlflow.log_metric(key, v)


def get_all_runs_comparison(experiment_name: str = DEFAULT_EXPERIMENT) -> "pd.DataFrame":
    """
    Busca todas as runs do experimento e retorna DataFrame comparativo.
    """
    import pandas as pd

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(f"Experimento '{experiment_name}' não encontrado.")
        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.auc_roc DESC"],
    )
    return runs
