"""
Features de Retorno e Volatilidade.

Conforme PLANEJAMENTO.md - Seção 2.1:
- log_return, log_return acumulado (2, 5, 10, 21 dias)
- Volatilidade realizada (rolling 5, 10, 21, 63)
- Volatilidade Garman-Klass (OHLC)
- Volatilidade de Parkinson (High-Low)
- Razão vol curta / vol longa
"""

import numpy as np
import pandas as pd


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de log-retorno."""
    df = df.copy()

    # Log-retorno diário
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Log-retorno acumulado em N dias
    for n in [2, 5, 10, 21]:
        df[f"log_return_{n}d"] = df["log_return"].rolling(window=n).sum()

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de volatilidade."""
    df = df.copy()

    # Volatilidade realizada (desvio-padrão rolling dos log-retornos)
    for n in [5, 10, 21, 63]:
        df[f"realized_vol_{n}d"] = df["log_return"].rolling(window=n).std()

    # Volatilidade Garman-Klass (usa OHLC — mais eficiente que close-to-close)
    # GK = 0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])
    gk_daily = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    df["garman_klass_vol"] = gk_daily.rolling(window=21).mean().apply(np.sqrt)

    # Volatilidade de Parkinson (usa High-Low)
    # P = (1 / 4*ln(2)) * ln(H/L)^2
    parkinson_daily = (1 / (4 * np.log(2))) * log_hl**2
    df["parkinson_vol"] = parkinson_daily.rolling(window=21).mean().apply(np.sqrt)

    # Razão vol curta / vol longa — captura regime de volatilidade
    df["vol_ratio"] = df["realized_vol_5d"] / df["realized_vol_21d"]

    return df


def create_return_and_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de features de retorno e volatilidade."""
    df = add_return_features(df)
    df = add_volatility_features(df)
    return df
