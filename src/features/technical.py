"""
Indicadores Técnicos Clássicos.

Conforme PLANEJAMENTO.md - Seção 2.2:
- RSI (7, 14), MACD, Bollinger Bands, SMA/EMA ratios, ATR, ADX,
  CCI, Williams %R, Stochastic, OBV, MFI, VWAP ratio, Ichimoku
"""

import numpy as np
import pandas as pd
import ta


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona todos os indicadores técnicos clássicos ao DataFrame."""
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ---------- RSI ----------
    df["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df["rsi_7"] = ta.momentum.RSIIndicator(close=close, window=7).rsi()

    # ---------- MACD ----------
    macd = ta.trend.MACD(close=close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # ---------- Bollinger Bands ----------
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_pct"] = bb.bollinger_pband()

    # ---------- SMA Ratios (Close / SMA_N) ----------
    for n in [5, 10, 20, 50, 200]:
        sma = close.rolling(window=n).mean()
        df[f"sma_ratio_{n}"] = close / sma

    # ---------- EMA Ratios (Close / EMA_N) ----------
    for n in [9, 21, 50]:
        ema = close.ewm(span=n, adjust=False).mean()
        df[f"ema_ratio_{n}"] = close / ema

    # ---------- ATR (normalizado) ----------
    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
    df["atr_14"] = atr.average_true_range() / close  # normalizado pelo preço

    # ---------- ADX ----------
    adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
    df["adx_14"] = adx.adx()

    # ---------- CCI ----------
    df["cci_20"] = ta.trend.CCIIndicator(
        high=high, low=low, close=close, window=20
    ).cci()

    # ---------- Williams %R ----------
    df["williams_r"] = ta.momentum.WilliamsRIndicator(
        high=high, low=low, close=close, lbp=14
    ).williams_r()

    # ---------- Stochastic ----------
    stoch = ta.momentum.StochasticOscillator(
        high=high, low=low, close=close, window=14, smooth_window=3
    )
    df["stochastic_k"] = stoch.stoch()
    df["stochastic_d"] = stoch.stoch_signal()

    # ---------- OBV (normalizado) ----------
    obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume)
    obv_raw = obv.on_balance_volume()
    # Normalizar OBV pela média rolling de 20 dias para evitar escala absoluta
    obv_sma = obv_raw.rolling(window=20).mean()
    df["obv_norm"] = obv_raw / obv_sma.replace(0, np.nan)

    # ---------- MFI ----------
    df["mfi_14"] = ta.volume.MFIIndicator(
        high=high, low=low, close=close, volume=volume, window=14
    ).money_flow_index()

    # ---------- VWAP ratio (aproximado — VWAP rolling diário) ----------
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).rolling(window=20).sum() / volume.rolling(
        window=20
    ).sum()
    df["vwap_ratio"] = close / vwap

    # ---------- Ichimoku ----------
    ichimoku = ta.trend.IchimokuIndicator(
        high=high, low=low, window1=9, window2=26, window3=52
    )
    tenkan = ichimoku.ichimoku_conversion_line()
    kijun = ichimoku.ichimoku_base_line()
    senkou_a = ichimoku.ichimoku_a()
    senkou_b = ichimoku.ichimoku_b()

    # Posições relativas ao Close (normalizadas)
    df["ichimoku_tenkan_ratio"] = close / tenkan.replace(0, np.nan)
    df["ichimoku_kijun_ratio"] = close / kijun.replace(0, np.nan)
    df["ichimoku_senkou_a_ratio"] = close / senkou_a.replace(0, np.nan)
    df["ichimoku_senkou_b_ratio"] = close / senkou_b.replace(0, np.nan)

    return df
