"""
Data Loader Module - Fetch and preprocess stock data using yfinance
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta


def fetchHourlyData(ticker: str, days: int = 730) -> pd.DataFrame:
    """
    Fetch hourly stock data from yfinance.

    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        days: Number of days of historical data

    Returns:
        DataFrame with OHLCV data and engineered features
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=f"{days}d", interval="1h")

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    df.reset_index(inplace=True)
    df.rename(columns={"Datetime": "datetime"}, inplace=True)
    df.set_index("datetime", inplace=True)

    df = df[['open', 'high', 'low', 'close', 'volume']]

    df = calculate_features(df)

    return df


def calculateFeatures(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate engineered features for HMM model."""
    df = df.copy()

    df['return'] = df['close'].pct_change()

    df['range'] = (df['high'] - df['low']) / df['close']

    df['volumeVolatility'] = (
        df['volume'].rolling(window=20).std() /
        df['volume'].rolling(window=20).mean()
    )

    df.dropna(inplace=True)

    return df


def resampleToDaily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample hourly data to daily for buy & hold benchmark."""
    daily = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    daily.dropna(inplace=True)
    return daily
