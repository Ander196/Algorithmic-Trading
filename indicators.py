"""
Technical Indicators Module - Calculate all strategy indicators
"""
import pandas as pd
import numpy as np


def calculateRsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculateMomentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """Calculate momentum as percentage change."""
    momentum = (prices - prices.shift(period)) / prices.shift(period) * 100
    return momentum


def calculateAtr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    prev_close = df['close'].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def calculateVolatility(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate volatility as ATR/Close percentage."""
    atr = calculateAtr(df, period)
    volatility = (atr / df['close']) * 100
    return volatility


def calculateAdx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX indicator."""
    high = df['high']
    low = df['low']
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = high - prev_high
    minus_dm = prev_low - low

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = calculateAtr(df, period)

    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window=period, min_periods=period).mean()

    return adx


def calculateEma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    ema = prices.ewm(span=period, adjust=False).mean()
    return ema


def calculateMacd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD, signal line, and histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculateVolumeSma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Calculate simple moving average of volume."""
    return volume.rolling(window=period, min_periods=period).mean()


def calculateAllIndicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators needed for strategy."""
    df = df.copy()

    df['rsi'] = calculateRsi(df['close'], 14)
    df['momentum'] = calculateMomentum(df['close'], 10)
    df['volatility'] = calculateVolatility(df, 14)
    df['volumeSma20'] = calculateVolumeSma(df['volume'], 20)
    df['adx'] = calculateAdx(df, 14)
    df['ema50'] = calculateEma(df['close'], 50)
    df['ema200'] = calculateEma(df['close'], 200)
    df['macd'], df['macdSignal'], df['macdHist'] = calculateMacd(df['close'])

    df['volumeAboveSma'] = df['volume'] > df['volumeSma20']
    df['priceAboveEma50'] = df['close'] > df['ema50']
    df['priceAboveEma200'] = df['close'] > df['ema200']
    df['macdAboveSignal'] = df['macd'] > df['macdSignal']

    return df
