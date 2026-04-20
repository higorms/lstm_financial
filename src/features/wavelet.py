"""
Features via Decomposição Wavelet.

Conforme PLANEJAMENTO.md - Seção 2.7:
- Wavelet Daubechies db4, decomposição DWT multinível (4 níveis)
- Componentes: approx_4, detail_1-4, energy_ratio, denoised, denoised_return, trend_strength
- Aplicação ROLLING para evitar look-ahead bias (janela de 60 dias)
- MODWT complementar
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pywt

logger = logging.getLogger(__name__)

# Configurações conforme planejamento
WAVELET = "db4"
ROLLING_WINDOW = 60  # Janela para decomposição rolling
# Nível máximo de decomposição será calculado dinamicamente para evitar boundary effects
DECOMPOSITION_LEVEL = pywt.dwt_max_level(ROLLING_WINDOW, pywt.Wavelet(WAVELET).dec_len)  # tipicamente 3 para db4/60


def _dwt_decompose(signal: np.ndarray, wavelet: str = WAVELET,
                    level: int = DECOMPOSITION_LEVEL) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Aplica DWT multinível a um sinal 1D.

    Returns
    -------
    Tuple[np.ndarray, List[np.ndarray]]
        (coeficientes de aproximação, [detalhe_1, ..., detalhe_N])
        onde detalhe_1 é a mais alta frequência.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # coeffs[0] = approx, coeffs[1:] = details do nível mais profundo ao mais raso
    approx = coeffs[0]
    details = list(reversed(coeffs[1:]))  # details[0]=nível1 (alta freq), details[3]=nível4
    return approx, details


def _reconstruct_without_detail1(signal: np.ndarray, wavelet: str = WAVELET,
                                  level: int = DECOMPOSITION_LEVEL) -> np.ndarray:
    """
    Reconstrói o sinal removendo o nível de detalhe 1 (denoising).
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Zerar coeficientes de detalhe nível 1 (último na lista de pywt)
    coeffs[-1] = np.zeros_like(coeffs[-1])
    reconstructed = pywt.waverec(coeffs, wavelet)
    # Ajustar comprimento (waverec pode adicionar 1 amostra)
    return reconstructed[:len(signal)]


def _compute_energy(coeffs: np.ndarray) -> float:
    """Calcula a energia (soma dos quadrados) de coeficientes."""
    return np.sum(coeffs ** 2)


def _rolling_wavelet_features_at_t(window_data: np.ndarray) -> Dict[str, float]:
    """
    Calcula todas as features wavelet para uma única janela rolling.

    Parameters
    ----------
    window_data : np.ndarray
        Série de log-retornos da janela (tamanho = ROLLING_WINDOW).

    Returns
    -------
    Dict[str, float]
        Features wavelet para o instante t.
    """
    if len(window_data) < ROLLING_WINDOW or np.isnan(window_data).any():
        return {
            "wavelet_approx_4": np.nan,
            "wavelet_detail_1": np.nan,
            "wavelet_detail_2": np.nan,
            "wavelet_detail_3": np.nan,
            "wavelet_detail_4": np.nan,
            "wavelet_energy_ratio": np.nan,
            "wavelet_denoised": np.nan,
            "wavelet_denoised_return": np.nan,
            "wavelet_trend_strength": np.nan,
        }

    approx, details = _dwt_decompose(window_data)

    # Valor mais recente de cada componente (último coeficiente)
    features = {
        "wavelet_approx_4": approx[-1] if len(approx) > 0 else np.nan,
        "wavelet_detail_1": details[0][-1] if len(details) > 0 and len(details[0]) > 0 else np.nan,
        "wavelet_detail_2": details[1][-1] if len(details) > 1 and len(details[1]) > 0 else np.nan,
        "wavelet_detail_3": details[2][-1] if len(details) > 2 and len(details[2]) > 0 else np.nan,
        "wavelet_detail_4": details[3][-1] if len(details) > 3 and len(details[3]) > 0 else np.nan,
    }

    # Energy ratio: energia alta frequência (detail 1+2) / energia baixa (approx + detail 3+4)
    energy_high = _compute_energy(details[0]) + (_compute_energy(details[1]) if len(details) > 1 else 0)
    energy_low = _compute_energy(approx) + \
                 (_compute_energy(details[2]) if len(details) > 2 else 0) + \
                 (_compute_energy(details[3]) if len(details) > 3 else 0)
    features["wavelet_energy_ratio"] = energy_high / max(energy_low, 1e-10)

    # Série denoised (sem detail 1) — valor mais recente
    denoised = _reconstruct_without_detail1(window_data)
    features["wavelet_denoised"] = denoised[-1]

    # Retorno da série denoised (última variação)
    if len(denoised) >= 2:
        features["wavelet_denoised_return"] = denoised[-1] - denoised[-2]
    else:
        features["wavelet_denoised_return"] = np.nan

    # Trend strength: approx / std(original)
    std_orig = np.std(window_data)
    features["wavelet_trend_strength"] = approx[-1] / max(std_orig, 1e-10) if len(approx) > 0 else np.nan

    return features


def add_wavelet_features(df: pd.DataFrame, log_return_col: str = "log_return") -> pd.DataFrame:
    """
    Adiciona features wavelet ao DataFrame usando decomposição ROLLING.

    Para evitar look-ahead bias, a decomposição é aplicada em janelas rolling
    de tamanho ROLLING_WINDOW, usando apenas dados passados para calcular
    features no instante t.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame que já contém a coluna de log-retornos.
    log_return_col : str
        Nome da coluna de log-retornos.

    Returns
    -------
    pd.DataFrame
        DataFrame com features wavelet adicionadas.
    """
    df = df.copy()
    log_returns = df[log_return_col].values

    # Pré-alocar arrays para as features
    n = len(df)
    feature_names = [
        "wavelet_approx_4", "wavelet_detail_1", "wavelet_detail_2",
        "wavelet_detail_3", "wavelet_detail_4", "wavelet_energy_ratio",
        "wavelet_denoised", "wavelet_denoised_return", "wavelet_trend_strength",
    ]
    results = {name: np.full(n, np.nan) for name in feature_names}

    logger.info(f"Calculando features wavelet (rolling window={ROLLING_WINDOW})...")

    for i in range(ROLLING_WINDOW, n):
        window = log_returns[i - ROLLING_WINDOW:i]
        if np.isnan(window).any():
            continue
        feats = _rolling_wavelet_features_at_t(window)
        for name, value in feats.items():
            results[name][i] = value

    for name in feature_names:
        df[name] = results[name]

    logger.info(f"Features wavelet adicionadas: {len(feature_names)} colunas.")
    return df
