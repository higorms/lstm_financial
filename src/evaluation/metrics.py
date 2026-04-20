"""
Métricas de Classificação.

Conforme PLANEJAMENTO.md - Seção 5.1:
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, Log Loss
- Matriz de Confusão, Classification Report
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calcula todas as métricas de classificação.

    Parameters
    ----------
    y_true : np.ndarray
        Labels reais (0 ou 1).
    y_pred_proba : np.ndarray
        Probabilidades preditas pelo modelo.
    threshold : float
        Limiar de decisão.

    Returns
    -------
    Dict[str, float]
        Dicionário com todas as métricas.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_pred_proba),
        "log_loss": log_loss(y_true, y_pred_proba),
    }

    logger.info("Métricas de classificação:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    return metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> str:
    """Retorna o classification report como string."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    return classification_report(
        y_true, y_pred, target_names=["Negativo (0)", "Positivo (1)"]
    )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plota e opcionalmente salva a matriz de confusão."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negativo (0)", "Positivo (1)"],
        yticklabels=["Negativo (0)", "Positivo (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix salva: {save_path}")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plota a curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC curve salva: {save_path}")

    return fig


def log_classification_to_mlflow(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    report_dir: str,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calcula métricas, gera gráficos e loga tudo no MLflow (run ativa).
    """
    metrics = compute_classification_metrics(y_true, y_pred_proba, threshold)

    # Log métricas
    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    # Classification report
    report = get_classification_report(y_true, y_pred_proba, threshold)
    report_path = f"{report_dir}/classification_report.txt"
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # Confusion matrix
    cm_path = f"{report_dir}/confusion_matrix.png"
    fig = plot_confusion_matrix(y_true, y_pred_proba, threshold, save_path=cm_path)
    mlflow.log_artifact(cm_path)
    plt.close(fig)

    # ROC curve
    roc_path = f"{report_dir}/roc_curve.png"
    fig = plot_roc_curve(y_true, y_pred_proba, save_path=roc_path)
    mlflow.log_artifact(roc_path)
    plt.close(fig)

    return metrics
