"""
Pipeline de Feature Engineering.

Orquestra todas as 7 famílias de features conforme PLANEJAMENTO.md - Seção 2.
Também lida com a preparação dos dados para o modelo (Seção 3):
- Divisão temporal (70/15/15)
- Normalização (scaler fitado apenas no treino)
- Construção de sequências 3D (samples, timesteps, features)
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.volatility import create_return_and_volatility_features
from src.features.technical import add_technical_features
from src.features.microstructure import add_microstructure_features
from src.features.exogenous import add_exogenous_features, add_volume_features
from src.features.temporal import add_temporal_features
from src.features.wavelet import add_wavelet_features

logger = logging.getLogger(__name__)


def build_feature_dataframe(
    data: Dict[str, pd.DataFrame],
    target: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constrói o DataFrame completo de features aplicando todas as 7 famílias.

    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Dados limpos e alinhados: {'petr4': df, 'brent': df, 'usdbrl': df, 'ibov': df}
    target : pd.Series
        Target binário.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (df_features, target_aligned) — features e target sem NaN.
    """
    df = data["petr4"].copy()

    logger.info("=" * 60)
    logger.info("INÍCIO DA FEATURE ENGINEERING")
    logger.info("=" * 60)

    # --- 2.1 Retorno e Volatilidade ---
    logger.info("\n[1/7] Features de retorno e volatilidade...")
    df = create_return_and_volatility_features(df)
    logger.info(f"  Colunas: {df.shape[1]}")

    # --- 2.2 Indicadores Técnicos ---
    logger.info("\n[2/7] Indicadores técnicos clássicos...")
    df = add_technical_features(df)
    logger.info(f"  Colunas: {df.shape[1]}")

    # --- 2.3 Features de Volume ---
    logger.info("\n[3/7] Features baseadas em volume...")
    df = add_volume_features(df)
    logger.info(f"  Colunas: {df.shape[1]}")

    # --- 2.4 Microestrutura e Candlestick ---
    logger.info("\n[4/7] Features de microestrutura e candlestick...")
    df = add_microstructure_features(df)
    logger.info(f"  Colunas: {df.shape[1]}")

    # --- 2.5 Features Exógenas ---
    logger.info("\n[5/7] Features exógenas (Brent, USD/BRL, Ibovespa)...")
    df = add_exogenous_features(df, data["brent"], data["usdbrl"], data["ibov"])
    logger.info(f"  Colunas: {df.shape[1]}")

    # --- 2.6 Features Temporais ---
    logger.info("\n[6/7] Features temporais / calendário...")
    df = add_temporal_features(df)
    logger.info(f"  Colunas: {df.shape[1]}")

    # --- 2.7 Features Wavelet ---
    logger.info("\n[7/7] Features via decomposição wavelet...")
    df = add_wavelet_features(df, log_return_col="log_return")
    logger.info(f"  Colunas: {df.shape[1]}")    # --- Remover colunas OHLCV originais (já extraímos tudo delas) ---
    cols_to_drop = ["Open", "High", "Low", "Close", "Volume"]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # --- Substituir infinitos por NaN ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- Diagnóstico de NaN por feature (para debug) ---
    nan_per_col = df.isna().sum()
    cols_with_nan = nan_per_col[nan_per_col > 0].sort_values(ascending=False)
    if len(cols_with_nan) > 0:
        logger.info(f"\n  Diagnóstico NaN ({len(cols_with_nan)} features com NaN):")
        for col, count in cols_with_nan.head(15).items():
            # Identificar onde estão os NaN (início ou fim da série)
            nan_idx = df[col].isna()
            first_nan = nan_idx.idxmax() if nan_idx.any() else None
            last_valid = df[col].last_valid_index()
            logger.info(f"    {col}: {count} NaN ({count/len(df)*100:.1f}%) "
                        f"| primeiro NaN: {first_nan}, último válido: {last_valid}")

    # --- Alinhar target e remover NaN ---
    target_aligned = target.loc[df.index]

    # Forward fill para features com poucos NaN esporádicos (ex: feriados desalinhados)
    # Depois drop das linhas que ainda tenham NaN (início da série por lookback de indicadores)
    df.ffill(inplace=True)

    valid_mask = df.notna().all(axis=1) & target_aligned.notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.info(f"\n  Registros removidos por NaN restantes: {n_dropped} "
                    f"(de {len(df)}, restam {valid_mask.sum()})")
    df = df.loc[valid_mask]
    target_aligned = target_aligned.loc[valid_mask].astype(int)

    logger.info("\n" + "=" * 60)
    logger.info("FEATURE ENGINEERING CONCLUÍDA")
    logger.info(f"Shape final: {df.shape}")
    logger.info(f"Features: {df.shape[1]}")
    logger.info(f"Registros válidos: {len(df)}")
    logger.info(f"Período: {df.index.min().date()} a {df.index.max().date()}")
    logger.info(f"Target — positivos: {target_aligned.sum()} ({target_aligned.mean()*100:.1f}%), "
                f"negativos: {(1-target_aligned).sum()}")
    logger.info("=" * 60)

    return df, target_aligned


def get_feature_names(df_features: pd.DataFrame) -> List[str]:
    """Retorna lista ordenada dos nomes das features."""
    return sorted(df_features.columns.tolist())


# =========================================================================
# Seção 3: Preparação dos Dados para o Modelo
# =========================================================================


def split_temporal(
    df: pd.DataFrame,
    target: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Divisão temporal dos dados (sem shuffle).
    Conforme Seção 3.1: 70% treino, 15% validação, 15% teste.

    Returns
    -------
    Dict com chaves 'train', 'val', 'test', cada uma contendo (df, target).
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": (df.iloc[:train_end], target.iloc[:train_end]),
        "val": (df.iloc[train_end:val_end], target.iloc[train_end:val_end]),
        "test": (df.iloc[val_end:], target.iloc[val_end:]),
    }

    for name, (d, t) in splits.items():
        logger.info(
            f"  {name}: {len(d)} registros "
            f"({d.index.min().date()} a {d.index.max().date()}) "
            f"| positivos: {t.mean()*100:.1f}%"
        )

    return splits


def normalize_features(
    splits: Dict[str, Tuple[pd.DataFrame, pd.Series]],
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], StandardScaler]:
    """
    Normaliza features com StandardScaler fitado apenas no treino.
    Conforme Seção 3.2.

    Returns
    -------
    Tuple contendo:
        - Dict com arrays normalizados {split_name: (X_array, y_array)}
        - scaler fitado (para salvar e reutilizar no deploy)
    """
    scaler = StandardScaler()

    train_df, train_target = splits["train"]
    scaler.fit(train_df)

    normalized = {}
    for name, (df, target) in splits.items():
        X = scaler.transform(df)
        y = target.values
        normalized[name] = (X, y)
        logger.info(f"  {name} normalizado: X={X.shape}, y={y.shape}")

    return normalized, scaler


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria janelas deslizantes 3D para a LSTM.
    Conforme Seção 3.3: (samples, timesteps, features).

    Parameters
    ----------
    X : np.ndarray
        Array 2D (n_samples, n_features) normalizado.
    y : np.ndarray
        Array 1D de targets.
    window_size : int
        Tamanho da janela temporal T.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X_seq, y_seq) onde X_seq.shape = (samples, window_size, features).
    """
    X_seq, y_seq = [], []
    for i in range(window_size, len(X)):
        X_seq.append(X[i - window_size:i])
        y_seq.append(y[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    logger.info(f"  Sequências criadas: X={X_seq.shape}, y={y_seq.shape}")
    return X_seq, y_seq


def prepare_model_data(
    df_features: pd.DataFrame,
    target: pd.Series,
    window_size: int = 60,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], StandardScaler, Dict]:
    """
    Pipeline completo de preparação: split → normalização → sequências 3D.

    Returns
    -------
    Tuple contendo:
        - sequences: Dict {split_name: (X_3d, y)} 
        - scaler: StandardScaler fitado no treino
        - metadata: Dict com informações auxiliares (datas, etc.)
    """
    logger.info("=" * 60)
    logger.info("PREPARAÇÃO DOS DADOS PARA O MODELO")
    logger.info("=" * 60)

    # 1. Divisão temporal
    logger.info("\n1. Divisão temporal...")
    splits = split_temporal(df_features, target, train_ratio, val_ratio)

    # Guardar datas para backtest
    metadata = {
        "dates": {
            name: df.index.tolist() for name, (df, _) in splits.items()
        },
        "window_size": window_size,
        "n_features": df_features.shape[1],
        "feature_names": df_features.columns.tolist(),
    }

    # 2. Normalização
    logger.info("\n2. Normalização (StandardScaler fitado no treino)...")
    normalized, scaler = normalize_features(splits)

    # 3. Sequências 3D
    logger.info(f"\n3. Construção de sequências (T={window_size})...")
    sequences = {}
    for name, (X, y) in normalized.items():
        X_seq, y_seq = create_sequences(X, y, window_size)
        sequences[name] = (X_seq, y_seq)

        # Ajustar datas (perdem-se as primeiras window_size)
        metadata["dates"][name] = metadata["dates"][name][window_size:]

    logger.info(f"\nResumo final:")
    for name, (X, y) in sequences.items():
        logger.info(f"  {name}: X={X.shape}, y={y.shape}, positivos={y.mean()*100:.1f}%")

    logger.info("=" * 60)
    return sequences, scaler, metadata
