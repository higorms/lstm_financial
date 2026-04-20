"""
Módulo de coleta de dados financeiros via yfinance.

Conforme PLANEJAMENTO.md - Seção 1.1:
- Download do ativo PETR4.SA (period="max", granularidade 1d)
- Download de séries auxiliares: Brent (BZ=F), USD/BRL (USDBRL=X), Ibovespa (^BVSP)
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Configuração dos ativos conforme planejamento
ASSETS_CONFIG = {
    "petr4": {
        "symbol": "PETR4.SA",
        "description": "Petrobras PN - Ativo principal",
    },
    "brent": {
        "symbol": "BZ=F",
        "description": "Petróleo Brent - Feature exógena (principal driver)",
    },
    "usdbrl": {
        "symbol": "USDBRL=X",
        "description": "Dólar/Real - Feature exógena (exposição cambial)",
    },
    "ibov": {
        "symbol": "^BVSP",
        "description": "Ibovespa - Feature exógena (contexto de mercado)",
    },
}


def download_asset(
    symbol: str,
    period: str = "max",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Faz download de um ativo via yfinance.

    Parameters
    ----------
    symbol : str
        Ticker do ativo (ex: 'PETR4.SA').
    period : str
        Período de dados (default: 'max' para maior histórico possível).
    interval : str
        Granularidade (default: '1d').

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas OHLCV + Adj Close, indexado por Date.
    """
    logger.info(f"Baixando dados de {symbol} (period={period}, interval={interval})")
    df = yf.download(symbol, period=period, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"Nenhum dado retornado para {symbol}")

    # Garantir que o índice é DatetimeIndex e remover timezone se presente
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Flatten MultiIndex columns se existir (yfinance >= 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    logger.info(
        f"  -> {symbol}: {len(df)} registros, "
        f"de {df.index.min().date()} a {df.index.max().date()}"
    )
    return df


def download_all_assets(
    period: str = "max",
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """
    Faz download de todos os ativos definidos em ASSETS_CONFIG.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dicionário {nome_ativo: DataFrame}.
    """
    data = {}
    for name, config in ASSETS_CONFIG.items():
        try:
            df = download_asset(config["symbol"], period=period, interval=interval)
            data[name] = df
            logger.info(f"  ✓ {config['description']}")
        except Exception as e:
            logger.error(f"  ✗ Erro ao baixar {config['symbol']}: {e}")
            raise
    return data


def save_raw_data(
    data: Dict[str, pd.DataFrame],
    output_dir: str = "data/raw",
) -> None:
    """
    Salva os DataFrames brutos em CSV na pasta raw.

    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Dicionário {nome_ativo: DataFrame}.
    output_dir : str
        Diretório de saída.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, df in data.items():
        filepath = output_path / f"{name}_raw.csv"
        df.to_csv(filepath)
        logger.info(f"Salvo: {filepath} ({len(df)} registros)")


def load_raw_data(
    input_dir: str = "data/raw",
) -> Dict[str, pd.DataFrame]:
    """
    Carrega os DataFrames brutos de CSV.

    Parameters
    ----------
    input_dir : str
        Diretório de entrada.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dicionário {nome_ativo: DataFrame}.
    """
    input_path = Path(input_dir)
    data = {}
    for name in ASSETS_CONFIG.keys():
        filepath = input_path / f"{name}_raw.csv"
        if filepath.exists():
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            data[name] = df
            logger.info(f"Carregado: {filepath} ({len(df)} registros)")
        else:
            logger.warning(f"Arquivo não encontrado: {filepath}")
    return data
