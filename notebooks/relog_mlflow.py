"""
Re-registra todos os experimentos no MLflow a partir dos artefatos salvos.
Usa sqlite backend para garantir persistência.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

import mlflow

# Configurar MLflow com SQLite explícito
DB_PATH = os.path.abspath("mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment("lstm_financial_petr4")

print(f"MLflow DB: {DB_PATH}")

# --- Dados dos treinamentos (extraídos dos logs do notebook 03) ---
models_info = {
    "etapa_A_baseline": {
        "architecture": "baseline",
        "stage": "A",
        "lr": 0.001, "batch_size": 64, "epochs_run": 17, "best_epoch": 2,
        "val_loss": 0.6905, "val_accuracy": 0.5321,
        "cls": {"accuracy": 0.5321, "precision": 0.5696, "recall": 0.6889, "f1_score": 0.6236, "auc_roc": 0.5234, "log_loss": 0.6905},
        "bt": {"retorno_acumulado_pct": 119.51, "sharpe_ratio": 0.8893, "sortino_ratio": 0.9862,
               "max_drawdown_pct": -33.55, "win_rate_pct": 56.43, "profit_factor": 1.3108,
               "calmar_ratio": 1.2651, "n_trades": 32, "cagr_pct": 42.45},
    },
    "etapa_B1_attention": {
        "architecture": "attention",
        "stage": "B",
        "lr": 0.001, "batch_size": 64,
    },
    "etapa_B2_conv1d_lstm": {
        "architecture": "conv1d_lstm",
        "stage": "B",
        "lr": 0.001, "batch_size": 64,
    },
    "etapa_B3_bidirectional": {
        "architecture": "bidirectional",
        "stage": "B",
        "lr": 0.001, "batch_size": 64,
    },
    "etapa_B4_lstm_gru": {
        "architecture": "lstm_gru",
        "stage": "B",
        "lr": 0.001, "batch_size": 64,
    },
}

# Carregar métricas de backtest dos reports
for name in models_info:
    report_dir = os.path.join("..", "reports", name)
    bt_file = os.path.join(report_dir, "backtest_metrics.json")
    if os.path.exists(bt_file):
        with open(bt_file) as f:
            bt = json.load(f)
        models_info[name]["bt"] = bt
        print(f"  Loaded backtest metrics for {name}")

# Carregar tabela comparativa para métricas de classificação
import pandas as pd
try:
    df_comp = pd.read_csv("../reports/tabela_comparativa_AB.csv")
    name_map = {
        "A - Baseline": "etapa_A_baseline",
        "B1_attention": "etapa_B1_attention",
        "B2_conv1d_lstm": "etapa_B2_conv1d_lstm",
        "B3_bidirectional": "etapa_B3_bidirectional",
        "B4_lstm_gru": "etapa_B4_lstm_gru",
    }
    for _, row in df_comp.iterrows():
        key = name_map.get(row["Modelo"])
        if key and "cls" not in models_info.get(key, {}):
            models_info[key]["cls"] = {
                "auc_roc": row["AUC-ROC"],
                "f1_score": row["F1"],
                "accuracy": row["Accuracy"],
            }
            models_info[key]["bt_from_table"] = {
                "sharpe_ratio": row["Sharpe"],
                "sortino_ratio": row["Sortino"],
                "max_drawdown_pct": row["MaxDD%"],
                "win_rate_pct": row["WinRate%"],
                "retorno_acumulado_pct": row["RetAcum%"],
            }
    print("  Loaded comparison table")
except Exception as e:
    print(f"  Warning: could not load comparison table: {e}")

# --- Registrar cada modelo como uma run ---
for run_name, info in models_info.items():
    print(f"\nRegistrando: {run_name}")
    with mlflow.start_run(run_name=run_name):
        # Tags
        mlflow.set_tag("stage", info.get("stage", ""))
        mlflow.set_tag("architecture", info.get("architecture", ""))

        # Parâmetros
        mlflow.log_param("architecture", info.get("architecture", ""))
        mlflow.log_param("learning_rate", info.get("lr", 0.001))
        mlflow.log_param("batch_size", info.get("batch_size", 64))
        mlflow.log_param("window_size", 60)
        mlflow.log_param("n_features", 75)
        if "best_epoch" in info:
            mlflow.log_param("best_epoch", info["best_epoch"])

        # Métricas de classificação
        cls = info.get("cls", {})
        for k, v in cls.items():
            mlflow.log_metric(f"cls_{k}", v)

        # Métricas de backtest
        bt = info.get("bt", info.get("bt_from_table", {}))
        for k, v in bt.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"bt_{k}", v)

        # Val metrics
        if "val_loss" in info:
            mlflow.log_metric("val_loss", info["val_loss"])
            mlflow.log_metric("val_accuracy", info["val_accuracy"])

        # Artefatos (reports)
        report_dir = os.path.join("..", "reports", run_name)
        if os.path.isdir(report_dir):
            mlflow.log_artifacts(report_dir, artifact_path="reports")
            print(f"  -> Logged artifacts from {report_dir}")

        # Modelo keras
        model_file = os.path.join("..", "models", f"{run_name}_best.keras")
        if os.path.exists(model_file):
            mlflow.log_artifact(model_file, artifact_path="model")
            print(f"  -> Logged model {model_file}")

# --- Etapa C (Optuna) ---
print("\nRegistrando: etapa_C_optuna")
with mlflow.start_run(run_name="etapa_C_optuna"):
    mlflow.set_tag("stage", "C")
    mlflow.set_tag("architecture", "baseline")
    mlflow.log_param("n_trials", 50)
    mlflow.log_param("best_architecture", "baseline")
    mlflow.log_metric("best_auc_roc", 0.5565)
    # Best params from Optuna
    best_params = {
        "window_size": 60, "lstm1_units": 128, "lstm2_units": 96,
        "dense_units": 64, "dropout_lstm": 0.4463, "dropout_dense": 0.2727,
        "learning_rate": 0.00953, "batch_size": 128, "use_batchnorm": False,
    }
    for k, v in best_params.items():
        mlflow.log_param(f"best_{k}", v)

# --- Etapa D (Modelo Final) ---
print("\nRegistrando: etapa_D_modelo_final")
with mlflow.start_run(run_name="etapa_D_modelo_final"):
    mlflow.set_tag("stage", "D")
    mlflow.set_tag("architecture", "lstm_gru")
    mlflow.set_tag("status", "final")
    mlflow.log_param("architecture", "lstm_gru")
    mlflow.log_param("selected_from", "etapa_B4")
    mlflow.log_param("selection_criteria", "best_financial_metrics")

    model_file = os.path.join("..", "models", "model.keras")
    if os.path.exists(model_file):
        mlflow.log_artifact(model_file, artifact_path="model")

    metrics_file = os.path.join("..", "models", "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            m = json.load(f)
        cls = m.get("classification", {})
        bt = m.get("backtest", {})
        for k, v in cls.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"cls_{k}", v)
        for k, v in bt.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"bt_{k}", v)

# --- Etapa E (Backtest Final) ---
print("\nRegistrando: etapa_E_backtest_final")
report_dir_e = os.path.join("..", "reports", "etapa_E_backtest_final")
with mlflow.start_run(run_name="etapa_E_backtest_final"):
    mlflow.set_tag("stage", "E")
    mlflow.set_tag("architecture", "lstm_gru")

    all_metrics_file = os.path.join(report_dir_e, "all_metrics.json")
    if os.path.exists(all_metrics_file):
        with open(all_metrics_file) as f:
            all_m = json.load(f)
        for k, v in all_m.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

    if os.path.isdir(report_dir_e):
        mlflow.log_artifacts(report_dir_e, artifact_path="reports")

print("\n✅ Todos os experimentos re-registrados no MLflow!")
print(f"Para visualizar: mlflow ui --backend-store-uri sqlite:///{DB_PATH} --port 5000")
