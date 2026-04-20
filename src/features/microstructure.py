"""
Features de Microestrutura e Candlestick.

Conforme PLANEJAMENTO.md - Seção 2.4:
- body_ratio, upper_shadow, lower_shadow, gap, intraday_range
"""

import numpy as np
import pandas as pd


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de microestrutura e candlestick."""
    df = df.copy()

    high_low_range = df["High"] - df["Low"]
    # Evitar divisão por zero em dias com range zero
    safe_range = high_low_range.replace(0, np.nan)

    # Proporção do corpo do candle
    df["body_ratio"] = (df["Close"] - df["Open"]) / safe_range

    # Sombra superior
    df["upper_shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / safe_range

    # Sombra inferior
    df["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / safe_range

    # Gap de abertura
    df["gap"] = df["Open"] / df["Close"].shift(1) - 1

    # Amplitude intraday normalizada
    df["intraday_range"] = high_low_range / df["Close"]

    return df
