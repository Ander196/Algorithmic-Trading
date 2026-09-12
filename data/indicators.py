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


# =============================================================================
# HMM Observable Features - Volatility Classifier Inputs
# =============================================================================

def computeLogReturns(prices: pd.Series, period: int) -> pd.Series:
    """Compute log returns over specified period."""
    return np.log(prices / prices.shift(period))


def computeRealizedVolatility(returns: pd.Series, period: int = 20) -> pd.Series:
    """Compute realized volatility as rolling standard deviation of returns."""
    return returns.rolling(window=period, min_periods=period).std()


def computeVolRatio(vol_5: pd.Series, vol_20: pd.Series) -> pd.Series:
    """Compute volatility ratio (5-period / 20-period)."""
    return vol_5 / vol_20


def computeVolumeZScore(volume: pd.Series, lookback: int = 50) -> pd.Series:
    """Compute normalized volume as z-score vs rolling mean."""
    rolling_mean = volume.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = volume.rolling(window=lookback, min_periods=lookback).std()
    return (volume - rolling_mean) / rolling_std


def computeVolumeTrend(volume: pd.Series, period: int = 10) -> pd.Series:
    """Compute volume trend as slope of rolling SMA."""
    volume_sma = volume.rolling(window=period, min_periods=period).mean()
    return volume_sma.pct_change(period)


def computeSmaSlope(prices: pd.Series, period: int = 50) -> pd.Series:
    """Compute slope of SMA as percentage change over period."""
    sma = prices.rolling(window=period, min_periods=period).mean()
    return sma.pct_change(period)


def computeRSIZScore(rsi: pd.Series, lookback: int = 50) -> pd.Series:
    """Compute RSI z-score for mean reversion signal."""
    rsi_mean = rsi.rolling(window=lookback, min_periods=lookback // 2).mean()
    rsi_std = rsi.rolling(window=lookback, min_periods=lookback // 2).std()
    return (rsi - rsi_mean) / rsi_std


def computeDistanceFromSma200(prices: pd.Series, sma_period: int = 200) -> pd.Series:
    """Compute distance from 200 SMA as percentage of price."""
    sma = prices.rolling(window=sma_period, min_periods=sma_period).mean()
    return ((prices - sma) / sma) * 100


def computeRateOfChange(prices: pd.Series, period: int) -> pd.Series:
    """Compute rate of change (ROC) as percentage."""
    return ((prices - prices.shift(period)) / prices.shift(period)) * 100


def computeNormalizedAtr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute normalized ATR (ATR / close) for range analysis."""
    atr = calculateAtr(df, period)
    return atr / df['close']


def computeRollingZScore(series: pd.Series, lookback: int = 252) -> pd.Series:
    """Standardize any feature with rolling z-score (252-period lookback)."""
    rolling_mean = series.rolling(window=lookback, min_periods=lookback // 2).mean()
    rolling_std = series.rolling(window=lookback, min_periods=lookback // 2).std()
    return (series - rolling_mean) / rolling_std


def computeAllHMMFeatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all observable features for HMM volatility classifier.

    Features:
    - Returns: log returns over 1, 5, 20 periods
    - Volatility: realized vol (20-period rolling std), vol ratio (5/20)
    - Volume: normalized volume (z-score vs 50-period), volume trend (10-period SMA slope)
    - Trend: ADX (14-period), slope of 50-period SMA
    - Mean reversion: RSI(14) z-score, distance from 200 SMA as % of price
    - Momentum: ROC 10 and 20 period
    - Range: normalized ATR (14-period)

    All features standardized with rolling z-scores (252-period lookback).
    """
    features = pd.DataFrame(index=df.index)

    # Returns: log returns over 1, 5, 20 periods
    log_returns = np.log(df['close'] / df['close'].shift(1))
    features['return_1'] = computeRollingZScore(log_returns, 252)
    features['return_5'] = computeRollingZScore(computeLogReturns(df['close'], 5), 252)
    features['return_20'] = computeRollingZScore(computeLogReturns(df['close'], 20), 252)

    # Volatility: realized vol (20-period), vol ratio (5/20)
    vol_5 = computeRealizedVolatility(log_returns, 5)
    vol_20 = computeRealizedVolatility(log_returns, 20)
    features['realized_vol_20'] = computeRollingZScore(vol_20.fillna(0), 252)
    features['vol_ratio'] = computeRollingZScore(computeVolRatio(vol_5, vol_20).fillna(1), 252)

    # Volume: normalized volume (z-score), volume trend
    features['volume_zscore'] = computeRollingZScore(computeVolumeZScore(df['volume'], 50), 252)
    features['volume_trend'] = computeRollingZScore(computeVolumeTrend(df['volume'], 10), 252)

    # Trend: ADX, SMA slope
    features['adx'] = computeRollingZScore(calculateAdx(df, 14), 252)
    features['sma_slope_50'] = computeRollingZScore(computeSmaSlope(df['close'], 50), 252)

    # Mean reversion: RSI z-score, distance from 200 SMA
    rsi = calculateRsi(df['close'], 14)
    features['rsi_zscore'] = computeRollingZScore(computeRSIZScore(rsi, 50).fillna(0), 252)
    features['distance_sma200'] = computeRollingZScore(computeDistanceFromSma200(df['close'], 200), 252)

    # Momentum: ROC 10 and 20
    features['roc_10'] = computeRollingZScore(computeRateOfChange(df['close'], 10), 252)
    features['roc_20'] = computeRollingZScore(computeRateOfChange(df['close'], 20), 252)

    # Range: normalized ATR
    features['norm_atr'] = computeRollingZScore(computeNormalizedAtr(df, 14), 252)

    return features


def prepareFeaturesForHMM(df: pd.DataFrame, min_periods: int = 504) -> pd.DataFrame:
    """
    Prepare features for HMM training.

    Args:
        df: DataFrame with OHLCV data
        min_periods: Minimum required periods (default 504 = 2 years trading days)

    Returns:
        DataFrame with all features, NaN rows dropped
    """
    features = computeAllHMMFeatures(df)

    # Drop rows with insufficient lookback data
    features = features.dropna()

    if len(features) < min_periods:
        raise ValueError(
            f"Insufficient data: {len(features)} periods, need at least {min_periods}. "
            "Ensure at least 2 years of daily data."
        )

    return features