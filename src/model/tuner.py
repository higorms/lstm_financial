"""
Otimização de Hiperparâmetros com Optuna + MLflow.

Conforme PLANEJAMENTO.md - Seção 4.2 (Etapa C):
- Espaço de busca definido no planejamento
- Integração Optuna → MLflow (nested runs)
- Objetivo: maximizar AUC-ROC na validação
"""

import logging
from typing import Callable, Dict, Optional, Tuple

import mlflow
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.model.builder import build_model

logger = logging.getLogger(__name__)


def create_objective(
    architecture: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    experiment_name: str = "lstm_financial_petr4",
    parent_run_id: Optional[str] = None,
) -> Callable:
    """
    Cria a função objetivo do Optuna com o espaço de busca do planejamento.

    Parameters
    ----------
    architecture : str
        Nome da arquitetura vencedora da Etapa B.
    X_train, y_train, X_val, y_val : np.ndarray
        Dados (já em formato de sequências 3D).
    experiment_name : str
        Nome do experimento MLflow.
    parent_run_id : str, optional
        ID da run pai no MLflow (para nested runs).
    """

    def objective(trial: optuna.Trial) -> float:
        # Espaço de busca conforme planejamento
        window_size = trial.suggest_categorical("window_size", [30, 45, 60, 90])
        lstm1_units = trial.suggest_int("lstm1_units", 64, 256, step=32)
        lstm2_units = trial.suggest_int("lstm2_units", 32, 128, step=32)
        dense_units = trial.suggest_int("dense_units", 16, 64, step=16)
        dropout_lstm = trial.suggest_float("dropout_lstm", 0.1, 0.5)
        dropout_dense = trial.suggest_float("dropout_dense", 0.1, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])
        l2_reg = trial.suggest_float("l2_reg", 1e-5, 1e-2, log=True)

        # Ajustar dados se window_size diferente do original
        # Nota: os dados já vêm com um window_size fixo. Se o trial propõe
        # um window_size diferente, precisamos recortar ou reconstruir.
        # Para simplificar a otimização, usamos os dados como estão e
        # o window_size afeta apenas se for <= ao window original.
        current_window = X_train.shape[1]
        if window_size <= current_window:
            X_tr = X_train[:, -window_size:, :]
            X_v = X_val[:, -window_size:, :]
        else:
            X_tr = X_train
            X_v = X_val
            window_size = current_window

        input_shape = (X_tr.shape[1], X_tr.shape[2])

        # Construir modelo
        build_kwargs = {
            "dense_units": dense_units,
            "dropout_lstm": dropout_lstm,            "dropout_dense": dropout_dense,
            "use_batchnorm": use_batchnorm,
            "l2_reg": l2_reg,
        }

        # Adaptar nomes dos parâmetros conforme a arquitetura
        if architecture in ("baseline", "attention", "conv1d_lstm", "bidirectional"):
            build_kwargs["lstm1_units"] = lstm1_units
            build_kwargs["lstm2_units"] = lstm2_units
        elif architecture == "lstm_gru":
            build_kwargs["lstm_units"] = lstm1_units
            build_kwargs["gru_units"] = lstm2_units

        try:
            model = build_model(architecture, input_shape, **build_kwargs)
        except Exception as e:
            logger.warning(f"Trial {trial.number}: erro ao construir modelo: {e}")
            return 0.0

        # Compilar
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        # Callbacks leves (sem TensorBoard para velocidade)
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=10,
                restore_best_weights=True, verbose=0,
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=3, min_lr=1e-6, verbose=0,
            ),
        ]

        # Treinar
        history = model.fit(
            X_tr, y_train,
            validation_data=(X_v, y_val),
            epochs=100,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
        )

        # Avaliar
        y_pred_proba = model.predict(X_v, verbose=0).flatten()
        try:
            auc = roc_auc_score(y_val, y_pred_proba)
        except Exception:
            auc = 0.0

        # Log no MLflow como nested run
        try:
            with mlflow.start_run(
                run_name=f"trial_{trial.number:03d}",
                nested=True,
            ):
                mlflow.log_params(trial.params)
                mlflow.log_metric("auc_roc", auc)
                mlflow.log_metric(
                    "val_loss", min(history.history["val_loss"])
                )
                mlflow.log_metric(
                    "val_accuracy", max(history.history["val_accuracy"])
                )
                mlflow.set_tag("stage", "C")
                mlflow.set_tag("trial_number", str(trial.number))
        except Exception as e:
            logger.warning(f"MLflow logging failed for trial {trial.number}: {e}")

        # Limpar sessão para liberar memória
        tf.keras.backend.clear_session()

        logger.info(f"  Trial {trial.number}: AUC-ROC={auc:.4f}")
        return auc

    return objective


def run_optuna_optimization(
    architecture: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    experiment_name: str = "lstm_financial_petr4",
) -> Tuple[optuna.Study, Dict]:
    """
    Executa a otimização Optuna integrada ao MLflow.

    Returns
    -------
    Tuple[optuna.Study, Dict]
        (estudo Optuna, melhores hiperparâmetros)
    """
    logger.info("=" * 60)
    logger.info(f"OTIMIZAÇÃO DE HIPERPARÂMETROS (Etapa C)")
    logger.info(f"Arquitetura: {architecture}, Trials: {n_trials}")
    logger.info("=" * 60)

    from src.utils.mlflow_utils import DEFAULT_TRACKING_URI
    if not mlflow.get_tracking_uri() or "mlruns" in mlflow.get_tracking_uri():
        mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="etapa_C_optuna") as parent_run:
        mlflow.set_tag("stage", "C")
        mlflow.set_tag("architecture", architecture)
        mlflow.log_param("n_trials", n_trials)

        objective = create_objective(
            architecture=architecture,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            experiment_name=experiment_name,
            parent_run_id=parent_run.info.run_id,
        )

        study = optuna.create_study(
            direction="maximize",
            study_name=f"lstm_{architecture}_optimization",
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Log melhores resultados
        best = study.best_params
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric("best_auc_roc", study.best_value)

        logger.info(f"\nMelhor AUC-ROC: {study.best_value:.4f}")
        logger.info(f"Melhores parâmetros: {best}")

    return study, best
