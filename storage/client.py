"""Supabase persistence helpers for models and runtime trading data."""
import os
from datetime import datetime
from typing import Optional

import numpy as np
from supabase import Client, create_client

_env_url = os.getenv("SUPABASE_URL")
_env_key = os.getenv("SUPABASE_ANON_KEY")
_client: Optional[Client] = None


def getClient() -> Client:
    global _client
    if _client is None:
        if not _env_url or not _env_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")
        _client = create_client(_env_url, _env_key)
    return _client


def storeStockRiskModel(model) -> dict:
    data = model.to_dict()
    response = getClient().table("stock_risk_models").upsert(data, on_conflict="ticker,model_version").execute()
    return response.data[0] if response.data else {}


def storeAllocationResult(signal) -> dict:
    data = {
        "result_timestamp": signal.timestamp.isoformat(),
        "ticker": signal.ticker,
        "market_regime": signal.regime_label,
        "market_regime_probability": signal.market_regime.probability,
        "base_multiplier": signal.base_multiplier,
        "relative_vol": signal.relative_vol,
        "beta": signal.beta,
        "vol_scalar": signal.vol_scalar,
        "final_multiplier": signal.final_multiplier,
        "is_regime_confirmed": signal.is_regime_confirmed,
        "is_flickering": signal.is_flickering,
        "reasoning": signal.reasoning,
    }
    response = getClient().table("allocation_results").insert(data).execute()
    return response.data[0] if response.data else {}


def storeHMMResult(*args, **kwargs) -> dict:
    """Legacy compatibility for the old hmm_results table."""
    ticker = kwargs.get("ticker", args[0] if args else None)
    result_date = kwargs.get("result_date", args[1] if len(args) > 1 else None)
    state_probabilities = kwargs.get("state_probabilities", args[5] if len(args) > 5 else [])
    probs = np.asarray(state_probabilities, dtype=float)
    total = float(probs.sum()) if probs.size else 0.0
    if total > 0:
        probs = probs / total
    data = {
        "ticker": str(ticker).upper(),
        "result_date": result_date.date().isoformat() if isinstance(result_date, datetime) else str(result_date),
        "regime_label": kwargs.get("regime_label", args[2] if len(args) > 2 else None),
        "state_id": kwargs.get("state_id", args[3] if len(args) > 3 else None),
        "probability": float(kwargs.get("probability", args[4] if len(args) > 4 else 0.0)),
        "state_probabilities": probs.tolist(),
        "volatility_level": kwargs.get("volatility_level"),
        "position_multiplier": kwargs.get("position_multiplier"),
        "is_confirmed": kwargs.get("is_confirmed", False),
        "is_flickering": kwargs.get("is_flickering", False),
        "stability_bars": kwargs.get("stability_bars"),
        "bic_score": kwargs.get("bic_score"),
        "n_regimes": kwargs.get("n_regimes"),
        "training_date": kwargs.get("training_date").isoformat() if kwargs.get("training_date") else None,
    }
    response = getClient().table("hmm_results").insert(data).execute()
    return response.data[0] if response.data else {}


def getHMMResults(ticker: Optional[str] = None, limit: int = 100) -> list[dict]:
    query = getClient().table("hmm_results").select("*")
    if ticker:
        query = query.eq("ticker", ticker.upper())
    return query.order("result_date", desc=True).limit(limit).execute().data or []


def storeStockPrices(ticker: str, price_date: str, open_price: float, high: float, low: float, close: float, volume: Optional[int] = None, adj_close: Optional[float] = None) -> dict:
    data = {"ticker": ticker.upper(), "price_date": price_date, "open": open_price, "high": high, "low": low, "close": close, "volume": volume, "adj_close": adj_close}
    response = getClient().table("stock_prices").insert(data).execute()
    return response.data[0] if response.data else {}


def storeSignal(ticker: str, signal_date: str, score: float, action: str = "buy", indicators: Optional[dict] = None) -> dict:
    data = {"ticker": ticker.upper(), "signal_date": signal_date, "score": score, "action": action, "indicators": indicators}
    response = getClient().table("signals").insert(data).execute()
    return response.data[0] if response.data else {}
