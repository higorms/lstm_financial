"""
Serialização — Salvamento de modelo, scaler e configurações.

Conforme PLANEJAMENTO.md - Seção 6:
- Modelo: .keras (SavedModel)
- Scaler: .pkl (joblib)
- Configuração de features: .json
- Métricas: .json
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def save_model(model: tf.keras.Model, filepath: str) -> None:
    """Salva modelo Keras em formato .keras."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    model.save(filepath)
    logger.info(f"Modelo salvo: {filepath}")


def load_model(filepath: str) -> tf.keras.Model:
    """Carrega modelo Keras."""
    model = tf.keras.models.load_model(filepath)
    logger.info(f"Modelo carregado: {filepath}")
    return model


def save_scaler(scaler: Any, filepath: str) -> None:
    """Salva scaler com joblib."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, filepath)
    logger.info(f"Scaler salvo: {filepath}")


def load_scaler(filepath: str) -> Any:
    """Carrega scaler com joblib."""
    scaler = joblib.load(filepath)
    logger.info(f"Scaler carregado: {filepath}")
    return scaler


def save_feature_config(
    feature_names: List[str],
    window_size: int,
    extra_params: Dict = None,
    filepath: str = "models/feature_config.json",
) -> None:
    """Salva configuração de features em JSON."""
    config = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "window_size": window_size,
    }
    if extra_params:
        config.update(extra_params)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Feature config salva: {filepath}")


def load_feature_config(filepath: str = "models/feature_config.json") -> Dict:
    """Carrega configuração de features."""
    with open(filepath, "r") as f:
        config = json.load(f)
    logger.info(f"Feature config carregada: {filepath}")
    return config


def save_metrics(metrics: Dict, filepath: str) -> None:
    """Salva métricas em JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Converter tipos numpy para nativos Python
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = float(v)
        else:
            clean[k] = v

    with open(filepath, "w") as f:
        json.dump(clean, f, indent=2)
    logger.info(f"Métricas salvas: {filepath}")
