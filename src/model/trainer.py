"""
Treinamento de modelos com integração MLflow + TensorBoard.

Conforme PLANEJAMENTO.md - Seções 4.3 e 4.4:
- Loss: binary_crossentropy
- Optimizer: Adam + ReduceLROnPlateau
- EarlyStopping, ModelCheckpoint
- Log de parâmetros, métricas e artefatos no MLflow
- Curvas de treinamento no TensorBoard
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.models import Model

logger = logging.getLogger(__name__)


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Calcula class_weight para lidar com desbalanceamento.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info(f"Class weights: {class_weight}")
    return class_weight


def get_callbacks(
    run_name: str,
    logs_dir: str = "logs",
    models_dir: str = "models",
    patience_es: int = 15,
    patience_lr: int = 5,
    lr_factor: float = 0.5,
) -> list:
    """
    Retorna lista de callbacks conforme Seção 4.3.
    """
    tb_log_dir = os.path.join(logs_dir, run_name)
    checkpoint_path = os.path.join(models_dir, f"{run_name}_best.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_factor,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        TensorBoard(
            log_dir=tb_log_dir,
            histogram_freq=1,
            write_graph=True,
        ),
    ]
    return callbacks


def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    run_name: str,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    epochs: int = 200,
    use_class_weight: bool = True,
    logs_dir: str = "logs",
    models_dir: str = "models",
    experiment_name: str = "lstm_financial_petr4",
    tags: Optional[Dict[str, str]] = None,
    log_to_mlflow: bool = True,
) -> Tuple[Model, dict]:
    """
    Treina o modelo com integração completa MLflow + TensorBoard.

    Parameters
    ----------
    model : Model
        Modelo Keras compilado ou não.
    X_train, y_train : np.ndarray
        Dados de treino.
    X_val, y_val : np.ndarray
        Dados de validação.
    run_name : str
        Nome da run (ex: 'etapa_A_baseline').
    learning_rate : float
        Taxa de aprendizado inicial.
    batch_size : int
        Tamanho do batch.
    epochs : int
        Máximo de épocas.
    use_class_weight : bool
        Se True, calcula e aplica class_weight.
    logs_dir, models_dir : str
        Diretórios de saída.
    experiment_name : str
        Nome do experimento no MLflow.
    tags : Dict[str, str], optional
        Tags adicionais para a run do MLflow.
    log_to_mlflow : bool
        Se True, registra no MLflow.

    Returns
    -------
    Tuple[Model, dict]
        (modelo treinado, histórico de métricas como dict)
    """
    # Compilar modelo
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Class weight
    class_weight = None
    if use_class_weight:
        class_weight = compute_class_weights(y_train)

    # Callbacks
    callbacks = get_callbacks(
        run_name=run_name,
        logs_dir=logs_dir,
        models_dir=models_dir,
    )

    # Garantir diretórios existem
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"TREINAMENTO: {run_name}")
    logger.info(f"{'='*60}")
    logger.info(f"  Arquitetura: {model.name}")
    logger.info(f"  Input shape: {X_train.shape[1:]}")
    logger.info(f"  LR: {learning_rate}, Batch: {batch_size}, Max epochs: {epochs}")
    logger.info(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    # Parâmetros a logar
    params = {
        "architecture": model.name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "window_size": X_train.shape[1],
        "n_features": X_train.shape[2],
        "use_class_weight": use_class_weight,
        "train_samples": X_train.shape[0],
        "val_samples": X_val.shape[0],
    }
    if tags:
        params.update({f"tag_{k}": v for k, v in tags.items()})

    # Treinamento
    if log_to_mlflow:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            # Log parâmetros
            mlflow.log_params(params)
            if tags:
                mlflow.set_tags(tags)

            # Treinar
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight,
                callbacks=callbacks,
                verbose=1,
            )

            # Log métricas finais
            best_epoch = np.argmin(history.history["val_loss"])
            mlflow.log_metric("best_epoch", best_epoch)
            mlflow.log_metric("train_loss", history.history["loss"][best_epoch])
            mlflow.log_metric("val_loss", history.history["val_loss"][best_epoch])
            mlflow.log_metric("train_accuracy", history.history["accuracy"][best_epoch])
            mlflow.log_metric("val_accuracy", history.history["val_accuracy"][best_epoch])
            mlflow.log_metric("total_epochs", len(history.history["loss"]))

            # Log modelo
            mlflow.keras.log_model(model, "model")

            logger.info(f"  MLflow run ID: {run.info.run_id}")
    else:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )

    best_epoch = np.argmin(history.history["val_loss"])
    logger.info(f"\n  Melhor época: {best_epoch}")
    logger.info(f"  Val Loss: {history.history['val_loss'][best_epoch]:.4f}")
    logger.info(f"  Val Accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")

    return model, history.history
