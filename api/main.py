"""
FastAPI app for regression predictions.
"""

from __future__ import annotations

import csv
import logging
import time
import sys
from contextlib import asynccontextmanager
from datetime import date as Date, datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.model.regression_inference import load_regression_artifacts, predict_next_return

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_artifacts = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _artifacts
    try:
        _artifacts = load_regression_artifacts(BASE_DIR)
    except Exception:
        logger.exception("Failed to load regression artifacts")
        _artifacts = None
    yield
    _artifacts = None


app = FastAPI(title="LSTM Financial Regression API", version="1.0.0", lifespan=lifespan)

INFERENCE_LOG_PATH = BASE_DIR / "logs" / "inference_regression.csv"
API_LOG_PATH = BASE_DIR / "logs" / "api_requests.csv"


class RegressionPrediction(BaseModel):
    predicted_date: Date
    predicted_return: float
    predicted_close: float


def _append_inference_metrics(result: dict) -> None:
    INFERENCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    predicted_close = float(result["predicted_close"])
    actual_close = float(result["actual_close"])
    error = predicted_close - actual_close
    abs_error = abs(error)
    squared_error = error ** 2
    rmse = squared_error ** 0.5
    mape = (abs_error / actual_close * 100.0) if actual_close != 0 else None

    row = {
        "inference_timestamp": datetime.now(timezone.utc).isoformat(),
        "predicted_date": result["predicted_date"],
        "predicted_close": predicted_close,
        "actual_close": actual_close,
        "error": error,
        "abs_error": abs_error,
        "squared_error": squared_error,
        "rmse": rmse,
        "mape": mape,
    }

    write_header = not INFERENCE_LOG_PATH.exists()
    with INFERENCE_LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _append_request_metrics(row: dict) -> None:
    API_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    write_header = not API_LOG_PATH.exists()
    with API_LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@app.middleware("http")
async def log_request_metrics(request: Request, call_next):
    started_at = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        logger.exception("Unhandled exception while processing request")
        raise
    finally:
        duration_ms = (time.perf_counter() - start_perf) * 1000.0
        try:
            _append_request_metrics(
                {
                    "request_timestamp": started_at.isoformat(),
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round(duration_ms, 3),
                }
            )
        except Exception:
            logger.exception("Failed to append request metrics")

@app.get("/health")
def health() -> dict:
    """Verifica se a API esta saudavel.

    Returns:
        dict: Status simples indicando que a API esta operante.

    Example:
        >>> GET /health
        {"status": "ok"}
    """
    return {"status": "ok"}


@app.get("/predict/regression", response_model=RegressionPrediction)
def predict_regression(date: Date) -> RegressionPrediction:
    """Gera previsao de retorno e fechamento para uma data alvo.

    Args:
        date (Date): Data para a qual a previsao deve ser calculada.

    Returns:
        RegressionPrediction: Predicao contendo data prevista, retorno e preco.

    Raises:
        HTTPException: 500 quando os artefatos nao estao carregados.
        HTTPException: 404 quando a data nao existe no dataset.
        HTTPException: 400 quando a data e invalida para a inferencia.

    Example:
        >>> GET /predict/regression?date=2024-01-15
        {
            "predicted_date": "2024-01-15",
            "predicted_return": 0.0123,
            "predicted_close": 131.45
        }
    """
    if _artifacts is None:
        raise HTTPException(status_code=500, detail="Artifacts not loaded")

    try:
        result = predict_next_return(_artifacts, date)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        _append_inference_metrics(result)
    except Exception:
        logger.exception("Failed to append inference metrics")

    return RegressionPrediction(
        predicted_date=result["predicted_date"],
        predicted_return=result["predicted_return"],
        predicted_close=result["predicted_close"],
    )
