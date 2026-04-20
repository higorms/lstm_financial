"""
Módulo de pré-processamento de dados financeiros.

Conforme PLANEJAMENTO.md - Seções 1.2 e 1.3:
- Remoção de dias sem negociação / valores nulos
- Tratamento de stock splits via Adj Close
- Alinhamento temporal entre séries (inner join)
- Verificação de outliers (Z-score e IQR) — sinalizar mas não remover
- Construção do target binário
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def adjust_ohlc_for_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajusta preços OHLC usando o fator de ajuste derivado de Adj Close / Close.
    Isso corrige stock splits e proventos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas OHLCV + Adj Close.

    Returns
    -------
    pd.DataFrame
        DataFrame com preços OHLC ajustados.
    """
    df = df.copy()
    if "Adj Close" not in df.columns:
        logger.warning("Coluna 'Adj Close' não encontrada. Retornando sem ajuste.")
        return df

    adj_factor = df["Adj Close"] / df["Close"]
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col] * adj_factor

    # Após ajuste, Close == Adj Close; podemos remover Adj Close
    df.drop(columns=["Adj Close"], inplace=True, errors="ignore")

    logger.info("Preços OHLC ajustados por splits/proventos via Adj Close.")
    return df


def clean_single_asset(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    """
    Limpeza de um único ativo: remove nulos, linhas com volume zero.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame OHLCV de um ativo.
    asset_name : str
        Nome do ativo (para logs).

    Returns
    -------
    pd.DataFrame
        DataFrame limpo.
    """
    initial_len = len(df)

    # Remover linhas completamente nulas
    df = df.dropna(how="all")

    # Remover linhas com valores nulos em colunas essenciais
    essential_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df.dropna(subset=essential_cols)

    # Remover dias com volume zero (sem negociação) — exceto para câmbio/índices
    if "Volume" in df.columns and asset_name not in ("usdbrl",):
        zero_vol = (df["Volume"] == 0).sum()
        if zero_vol > 0:
            logger.info(f"  {asset_name}: removendo {zero_vol} dias com volume zero")
            df = df[df["Volume"] > 0]

    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"  {asset_name}: {removed} registros removidos na limpeza")

    return df


def align_datasets(
    data: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Alinha temporalmente todas as séries via inner join nas datas.

    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Dicionário {nome_ativo: DataFrame}.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dicionários com DataFrames alinhados nas mesmas datas.
    """
    # Encontrar datas comuns a todos os ativos
    common_dates = None
    for name, df in data.items():
        dates = set(df.index)
        if common_dates is None:
            common_dates = dates
        else:
            common_dates = common_dates.intersection(dates)

    common_dates = sorted(common_dates)
    logger.info(f"Datas comuns após alinhamento: {len(common_dates)} "
                f"(de {common_dates[0].date()} a {common_dates[-1].date()})")

    aligned = {}
    for name, df in data.items():
        before = len(df)
        aligned[name] = df.loc[df.index.isin(common_dates)].sort_index()
        after = len(aligned[name])
        if before != after:
            logger.info(f"  {name}: {before} → {after} registros após alinhamento")

    return aligned


def detect_outliers(
    df: pd.DataFrame,
    columns: list = None,
    z_threshold: float = 4.0,
    iqr_factor: float = 3.0,
) -> pd.DataFrame:
    """
    Detecta outliers via Z-score e IQR. Sinaliza mas NÃO remove.
    Conforme planejamento: em séries financeiras, outliers carregam informação.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com dados numéricos.
    columns : list, optional
        Colunas para verificar. Se None, usa colunas numéricas.
    z_threshold : float
        Limiar de Z-score para considerar outlier.
    iqr_factor : float
        Fator de IQR para considerar outlier.

    Returns
    -------
    pd.DataFrame
        DataFrame booleano indicando outliers (True = outlier).
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_flags = pd.DataFrame(False, index=df.index, columns=columns)

    for col in columns:
        series = df[col].dropna()

        # Z-score
        z_scores = np.abs((series - series.mean()) / series.std())
        z_outliers = z_scores > z_threshold

        # IQR
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        iqr_outliers = (series < q1 - iqr_factor * iqr) | (series > q3 + iqr_factor * iqr)

        combined = z_outliers | iqr_outliers
        outlier_flags.loc[combined.index, col] = combined

        n_outliers = combined.sum()
        if n_outliers > 0:
            logger.info(
                f"  Outliers em '{col}': {n_outliers} "
                f"({n_outliers/len(series)*100:.2f}%)"
            )

    total = outlier_flags.any(axis=1).sum()
    logger.info(f"Total de datas com pelo menos 1 outlier: {total}")
    return outlier_flags


def build_target(df: pd.DataFrame, close_col: str = "Close") -> pd.Series:
    """
    Constrói o target binário conforme planejamento:
    target = 1 se (Close[t+1] - Close[t]) / Close[t] >= 0
    target = 0 caso contrário

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com coluna de preço de fechamento.
    close_col : str
        Nome da coluna de fechamento.

    Returns
    -------
    pd.Series
        Série com target binário (0 ou 1). Último registro será NaN.
    """
    returns = df[close_col].pct_change(periods=1).shift(-1)
    target = (returns >= 0).astype(int)
    # O último registro não tem retorno futuro
    target.iloc[-1] = np.nan
    logger.info(
        f"Target construído: {int(target.sum())} positivos ({target.mean()*100:.1f}%), "
        f"{int((1 - target).sum())} negativos, {target.isna().sum()} NaN"
    )
    return target


def preprocess_all(
    data: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, pd.DataFrame], pd.Series, pd.DataFrame]:
    """
    Pipeline completo de pré-processamento.

    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Dados brutos {nome_ativo: DataFrame}.

    Returns
    -------
    Tuple contendo:
        - data_clean: Dict[str, pd.DataFrame] — dados limpos e alinhados
        - target: pd.Series — target binário
        - outlier_flags: pd.DataFrame — flags de outliers do ativo principal
    """
    logger.info("=" * 60)
    logger.info("INÍCIO DO PRÉ-PROCESSAMENTO")
    logger.info("=" * 60)

    # 1. Ajustar splits no ativo principal
    logger.info("\n1. Ajustando OHLC por splits/proventos...")
    data_clean = {}
    for name, df in data.items():
        if "Adj Close" in df.columns:
            data_clean[name] = adjust_ohlc_for_splits(df)
        else:
            data_clean[name] = df.copy()

    # 2. Limpeza individual
    logger.info("\n2. Limpeza individual dos ativos...")
    for name in data_clean:
        data_clean[name] = clean_single_asset(data_clean[name], name)

    # 3. Alinhamento temporal
    logger.info("\n3. Alinhamento temporal (inner join)...")
    data_clean = align_datasets(data_clean)

    # 4. Detecção de outliers no ativo principal
    logger.info("\n4. Detecção de outliers (PETR4)...")
    outlier_flags = detect_outliers(
        data_clean["petr4"],
        columns=["Close", "Volume"],
    )

    # 5. Construção do target
    logger.info("\n5. Construção do target binário...")
    target = build_target(data_clean["petr4"])

    logger.info("\n" + "=" * 60)
    logger.info("PRÉ-PROCESSAMENTO CONCLUÍDO")
    logger.info(f"Registros finais: {len(data_clean['petr4'])}")
    logger.info(f"Período: {data_clean['petr4'].index.min().date()} a "
                f"{data_clean['petr4'].index.max().date()}")
    logger.info("=" * 60)

    return data_clean, target, outlier_flags
