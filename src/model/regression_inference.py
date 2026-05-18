"""
Regression inference helpers for the trained LSTM model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.serialization import load_model, load_scaler

logger = logging.getLogger(__name__)


@dataclass
class RegressionArtifacts:
    model: Any
    scaler_x: Any
    scaler_y: Any
    config: Dict[str, Any]
    feature_config: Dict[str, Any]
    df_features: pd.DataFrame
    close_prices: pd.Series
    window_size: int
    n_features: int
    feature_names: list[str]


def _normalize_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def load_regression_artifacts(base_dir: Path) -> RegressionArtifacts:
    """
    Load model, scalers, configs, and cached features for inference.
    """
    models_dir = base_dir / "models"
    data_dir = base_dir / "data" / "processed"

    model = load_model(str(models_dir / "model_regression.keras"))
    scaler_x = load_scaler(str(models_dir / "scaler_X_regression.pkl"))
    scaler_y = load_scaler(str(models_dir / "scaler_y_regression.pkl"))

    with open(models_dir / "regression_config.json", "r") as f:
        config = json.load(f)

    with open(models_dir / "feature_config.json", "r") as f:
        feature_config = json.load(f)

    df_features = pd.read_csv(data_dir / "features.csv", index_col=0, parse_dates=True)
    petr4_clean = pd.read_csv(data_dir / "petr4_clean.csv", index_col=0, parse_dates=True)

    common_idx = df_features.index.intersection(petr4_clean.index)
    df_features = df_features.loc[common_idx].sort_index()
    close_prices = petr4_clean.loc[common_idx, "Close"].sort_index()

    feature_names = feature_config.get("feature_names") or df_features.columns.tolist()
    missing = [name for name in feature_names if name not in df_features.columns]
    if missing:
        raise ValueError(f"Missing features in features.csv: {missing}")

    df_features = df_features[feature_names]

    window_size = int(config.get("window_size") or model.input_shape[1])
    n_features = int(config.get("n_features") or df_features.shape[1])

    if df_features.shape[1] != n_features:
        raise ValueError(
            "Feature count mismatch: "
            f"features.csv has {df_features.shape[1]}, config expects {n_features}"
        )

    logger.info(
        "Regression artifacts loaded: window_size=%s, n_features=%s, rows=%s",
        window_size,
        n_features,
        len(df_features),
    )

    return RegressionArtifacts(
        model=model,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        config=config,
        feature_config=feature_config,
        df_features=df_features,
        close_prices=close_prices,
        window_size=window_size,
        n_features=n_features,
        feature_names=feature_names,
    )


def predict_next_return(artifacts: RegressionArtifacts, date_value: Any) -> Dict[str, Any]:
    """
    Predict the next-day return for a given date (t -> t+1).

    Uses the same windowing logic as training: the model consumes the
    previous window_size rows ending at t-1.
    """
    ts = _normalize_timestamp(date_value)
    df_features = artifacts.df_features
    index = df_features.index

    if ts not in index:
        pos = int(index.searchsorted(ts, side="left"))
        if pos >= len(index):
            raise KeyError("Date not found in features dataset")
        ts = index[pos]
    else:
        pos = df_features.index.get_loc(ts)
    if not isinstance(pos, (int, np.integer)):
        raise ValueError("Date index is not unique")

    if pos < artifacts.window_size:
        raise ValueError(
            f"Not enough history for window_size={artifacts.window_size}"
        )

    if pos + 1 >= len(index):
        raise ValueError("No next date available for prediction")

    predicted_date = index[pos + 1]

    window = df_features.iloc[pos - artifacts.window_size : pos]
    x_scaled = artifacts.scaler_x.transform(window.values)
    x_seq = np.expand_dims(x_scaled, axis=0)

    y_scaled = artifacts.model.predict(x_seq, verbose=0).reshape(-1, 1)
    y_return = artifacts.scaler_y.inverse_transform(y_scaled).flatten()[0]

    close_t = float(artifacts.close_prices.loc[ts])
    pred_close = close_t * (1.0 + float(y_return))
    actual_close = float(artifacts.close_prices.loc[predicted_date])

    return {
        "date": ts.date(),
        "predicted_date": predicted_date.date(),
        "window_start": window.index[0].date(),
        "window_end": window.index[-1].date(),
        "close_t": close_t,
        "predicted_return": float(y_return),
        "predicted_close": float(pred_close),
        "actual_close": actual_close,
        "model_window_size": artifacts.window_size,
        "n_features": artifacts.n_features,
    }
