"""
Supabase client module for database operations.

Provides connection to Supabase and helper functions for storing
HMM results and other trading data.
"""
import os
from datetime import datetime
from typing import Optional

import numpy as np
from supabase import Client, create_client

_env_url = os.getenv("SUPABASE_URL")
_env_key = os.getenv("SUPABASE_ANON_KEY")

_client: Optional[Client] = None


def getClient() -> Client:
    """
    Get or create Supabase client singleton.

    Returns:
        Supabase Client instance
    """
    global _client
    if _client is None:
        if not _env_url or not _env_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")
        _client = create_client(_env_url, _env_key)
    return _client


def storeHMMResult(
    ticker: str,
    result_date: datetime | str,
    regime_label: str,
    state_id: int,
    probability: float,
    state_probabilities: list[float],
    volatility_level: Optional[str] = None,
    position_multiplier: Optional[float] = None,
    is_confirmed: bool = False,
    is_flickering: bool = False,
    stability_bars: Optional[int] = None,
    bic_score: Optional[float] = None,
    n_regimes: Optional[int] = None,
    training_date: Optional[datetime] = None,
) -> dict:
    """
    Store HMM model results in the database.

    Args:
        ticker: Stock ticker symbol
        result_date: Date of the result
        regime_label: Detected regime label (BEAR, BULL, etc.)
        state_id: State ID from HMM
        probability: Probability of the current state
        state_probabilities: Array of probabilities for all states
        volatility_level: CALM, MODERATE, or TURBULENT
        position_multiplier: Position sizing multiplier
        is_confirmed: Whether regime is confirmed
        is_flickering: Whether regime is flickering
        stability_bars: Consecutive bars in current regime
        bic_score: BIC score of the model
        n_regimes: Number of regimes in the model
        training_date: When model was trained

    Returns:
        The inserted record
    """
    client = getClient()

    if isinstance(result_date, datetime):
        result_date = result_date.date()

    # Normalize state probabilities if they're unnormalized (emission probs can be huge)
    if state_probabilities is not None and len(state_probabilities) > 0:
        probs = np.array(state_probabilities, dtype=float)
        total = probs.sum()
        if total > 0:
            normalized_probs = probs / total
        else:
            normalized_probs = np.ones(len(probs)) / len(probs)
    else:
        normalized_probs = np.array([])

    data = {
        "ticker": ticker.upper(),
        "result_date": result_date.isoformat() if hasattr(result_date, 'isoformat') else str(result_date),
        "regime_label": regime_label,
        "state_id": state_id,
        "probability": float(probability),
        "state_probabilities": normalized_probs.tolist(),
        "volatility_level": volatility_level,
        "position_multiplier": float(position_multiplier) if position_multiplier else None,
        "is_confirmed": is_confirmed,
        "is_flickering": is_flickering,
        "stability_bars": stability_bars,
        # Set BIC to null if too large for database precision
        "bic_score": float(bic_score) if bic_score and abs(bic_score) < 1e8 else None,
        "n_regimes": n_regimes,
        "training_date": training_date.isoformat() if training_date else None,
    }

    response = client.table("hmm_results").insert(data).execute()
    return response.data[0] if response.data else {}


def getHMMResults(ticker: Optional[str] = None, limit: int = 100) -> list[dict]:
    """
    Retrieve HMM results from the database.

    Args:
        ticker: Optional ticker filter
        limit: Maximum number of results

    Returns:
        List of HMM result records
    """
    client = getClient()
    query = client.table("hmm_results").select("*")

    if ticker:
        query = query.eq("ticker", ticker.upper())

    query = query.order("result_date", desc=True).limit(limit)
    response = query.execute()
    return response.data or []


def storeStockPrices(
    ticker: str,
    price_date: str,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: Optional[int] = None,
    adj_close: Optional[float] = None,
) -> dict:
    """
    Store OHLCV stock price data.

    Args:
        ticker: Stock ticker symbol
        price_date: Date of the price
        open_price: Open price
        high: High price
        low: Low price
        close: Close price
        volume: Trading volume
        adj_close: Adjusted close price

    Returns:
        The inserted record
    """
    client = getClient()

    data = {
        "ticker": ticker.upper(),
        "price_date": price_date,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "adj_close": adj_close,
    }

    response = client.table("stock_prices").insert(data).execute()
    return response.data[0] if response.data else {}


def storeSignal(
    ticker: str,
    signal_date: str,
    score: float,
    action: str = "buy",
    indicators: Optional[dict] = None,
) -> dict:
    """
    Store trading signal.

    Args:
        ticker: Stock ticker symbol
        signal_date: Date of the signal
        score: Signal confidence score (0-1)
        action: buy, sell, or watch
        indicators: Additional indicator values as JSON

    Returns:
        The inserted record
    """
    client = getClient()

    data = {
        "ticker": ticker.upper(),
        "signal_date": signal_date,
        "score": score,
        "action": action,
        "indicators": indicators,
    }

    response = client.table("signals").insert(data).execute()
    return response.data[0] if response.data else {}