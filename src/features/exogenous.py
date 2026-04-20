"""
Features Exógenas.

Conforme PLANEJAMENTO.md - Seção 2.5:
- Log-retornos do Brent, USD/BRL, Ibovespa (diário e 5d)
- Correlação rolling PETR4 vs Brent (5d)
- Beta rolling PETR4 vs Ibovespa (21d)

Conforme Seção 2.3 (Features Baseadas em Volume):
- volume_sma_ratio, volume_change, volume_price_trend, accumulation_distribution
"""

import numpy as np
import pandas as pd
import ta


def add_exogenous_features(
    df_main: pd.DataFrame,
    df_brent: pd.DataFrame,
    df_usdbrl: pd.DataFrame,
    df_ibov: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adiciona features exógenas ao DataFrame principal.

    Parameters
    ----------
    df_main : pd.DataFrame
        DataFrame do ativo principal (PETR4) — será modificado e retornado.
    df_brent, df_usdbrl, df_ibov : pd.DataFrame
        DataFrames dos ativos exógenos (já alinhados temporalmente).
    """
    df = df_main.copy()

    # --- Brent ---
    df["brent_return"] = np.log(df_brent["Close"] / df_brent["Close"].shift(1))
    df["brent_return_5d"] = df["brent_return"].rolling(window=5).sum()

    # --- USD/BRL ---
    df["usdbrl_return"] = np.log(df_usdbrl["Close"] / df_usdbrl["Close"].shift(1))
    df["usdbrl_return_5d"] = df["usdbrl_return"].rolling(window=5).sum()

    # --- Ibovespa ---
    df["ibov_return"] = np.log(df_ibov["Close"] / df_ibov["Close"].shift(1))
    df["ibov_return_5d"] = df["ibov_return"].rolling(window=5).sum()

    # --- Correlação rolling PETR4 vs Brent (5 dias) ---
    petr4_ret = np.log(df["Close"] / df["Close"].shift(1))
    df["correlation_brent_5d"] = petr4_ret.rolling(window=5).corr(df["brent_return"])

    # --- Beta rolling PETR4 vs Ibovespa (21 dias) ---
    cov = petr4_ret.rolling(window=21).cov(df["ibov_return"])
    var_ibov = df["ibov_return"].rolling(window=21).var()
    df["beta_ibov_21d"] = cov / var_ibov.replace(0, np.nan)

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona features baseadas em volume (PLANEJAMENTO.md - Seção 2.3).
    """
    df = df.copy()

    # Volume relativo: Volume / SMA(Volume, 20)
    vol_sma = df["Volume"].rolling(window=20).mean()
    df["volume_sma_ratio"] = df["Volume"] / vol_sma.replace(0, np.nan)

    # Variação percentual do volume
    df["volume_change"] = df["Volume"].pct_change()

    # Correlação rolling preço vs volume (21d)
    price_change = df["Close"].pct_change()
    df["volume_price_trend"] = price_change.rolling(window=21).corr(
        df["Volume"].pct_change()
    )

    # Accumulation/Distribution
    ad = ta.volume.AccDistIndexIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]
    )
    ad_raw = ad.acc_dist_index()    # Normalizar: variação percentual do AD (evita divisão por valores instáveis)
    df["accumulation_distribution"] = ad_raw.pct_change(periods=5).replace(
        [np.inf, -np.inf], np.nan
    )

    return df
