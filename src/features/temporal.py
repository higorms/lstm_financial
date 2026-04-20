"""
Features Temporais / Calendário.

Conforme PLANEJAMENTO.md - Seção 2.6:
- day_of_week (sin/cos encoding cíclico)
- month (sin/cos encoding cíclico)
- is_month_start, is_month_end, is_quarter_end
- days_since_year_start (normalizado)
"""

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features temporais e de calendário."""
    df = df.copy()
    idx = df.index

    # --- Dia da semana (sin/cos encoding cíclico) ---
    dow = idx.dayofweek  # 0=Monday, 4=Friday
    df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 5)  # 5 dias úteis
    df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 5)

    # --- Mês (sin/cos encoding cíclico) ---
    month = idx.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # --- Flags ---
    df["is_month_start"] = idx.is_month_start.astype(int)
    df["is_month_end"] = idx.is_month_end.astype(int)
    df["is_quarter_end"] = idx.is_quarter_end.astype(int)

    # --- Dias desde o início do ano (normalizado 0-1) ---
    day_of_year = idx.dayofyear
    days_in_year = pd.Series(idx.year, index=idx).apply(
        lambda y: 366 if pd.Timestamp(year=y, month=12, day=31).dayofyear == 366 else 365
    )
    df["days_since_year_start"] = day_of_year / days_in_year.values

    return df
