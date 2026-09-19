#!/usr/bin/env python
"""Daily Layer 1 inference + Layer 2 allocation calculation.

This job deliberately stops after AllocationSignal persistence. Strategy,
selection, risk-management and broker execution are downstream concerns.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd

from core.hmm_market import MarketRegimeClassifier
from core.stock_risk import AllocationEngine
from core.two_layer_engine import TwoLayerRegimeEngine
from data.data_loader import getActiveTickers
from jobs.common import fetch_price_history, get_supabase_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_active_market_model() -> tuple[MarketRegimeClassifier, str]:
    path = os.getenv("MARKET_MODEL_PATH", "models/market/current.json")
    model = MarketRegimeClassifier().load_model(path)
    version = getattr(model, "model_version", None)
    if not version:
        raise RuntimeError("Active market model does not contain model_version")
    return model, version


def _warm_market_model(model: MarketRegimeClassifier, history: pd.DataFrame) -> object:
    features = model.compute_features(history.set_index("price_date"))
    if features.empty:
        raise RuntimeError("Market history is insufficient to compute Layer 1 features")
    for _, row in features.iterrows():
        model.step(row)
    state = model.get_current_state()
    if state is None:
        raise RuntimeError("Layer 1 produced no market state")
    return state


def _calculate_stock(
    engine: TwoLayerRegimeEngine,
    market_state,
    market_model_version: str,
    ticker: str,
    market_history: pd.DataFrame,
    stock_history: pd.DataFrame,
):
    joined = market_history[["price_date", "close"]].rename(columns={"close": "market_close"}).merge(
        stock_history[["price_date", "close"]].rename(columns={"close": "stock_close"}),
        on="price_date",
        how="inner",
    ).sort_values("price_date")

    if len(joined) < engine.stock_config.beta_window + 1:
        raise RuntimeError(
            f"{ticker}: only {len(joined)} common sessions; "
            f"need {engine.stock_config.beta_window + 1}"
        )

    model = engine.get_or_create_stock_model(ticker)
    model.market_model_version = market_model_version

    # Rebuild deterministic rolling state from historical observations and
    # then calculate today's profile with exactly one final step.
    model.reset()
    for _, row in joined.iloc[:-1].iterrows():
        model.step(float(row["stock_close"]), float(row["market_close"]), row["price_date"].to_pydatetime())
    profile = model.step(
        float(joined.iloc[-1]["stock_close"]),
        float(joined.iloc[-1]["market_close"]),
        joined.iloc[-1]["price_date"].to_pydatetime(),
    )
    if profile is None:
        raise RuntimeError(f"{ticker}: Layer 2 did not warm up")

    signal = AllocationEngine.calculate(market_state, model, profile)
    engine.persist_stock_model(model)
    engine.persist_allocation(signal, market_model_version, model.model_version)
    return signal


def main() -> None:
    client = get_supabase_client()
    market_ticker = os.getenv("MARKET_TICKER", "SPY").upper()

    model, market_model_version = _load_active_market_model()
    market_history = fetch_price_history(client, market_ticker, limit=350)
    market_state = _warm_market_model(model, market_history)

    # Persist the daily Layer 1 state, but do not invoke strategy or execution.
    client.table("market_regime_results").upsert(
        {
            "result_date": market_state.timestamp.date().isoformat(),
            "market_ticker": market_ticker,
            "market_model_version": market_model_version,
            "state_id": market_state.state_id,
            "regime_label": market_state.label,
            "probability": market_state.probability,
            "state_probabilities": market_state.state_probabilities,
            "base_multiplier": market_state.base_multiplier,
            "is_confirmed": market_state.is_confirmed,
            "consecutive_bars": market_state.consecutive_bars,
            "is_flickering": market_state.is_flickering,
        },
        on_conflict="market_ticker,result_date,market_model_version",
    ).execute()

    tickers = [t.upper() for t in getActiveTickers(client) if t.upper() != market_ticker]
    engine = TwoLayerRegimeEngine(model)

    successes = 0
    failures: list[str] = []
    for ticker in tickers:
        try:
            stock_history = fetch_price_history(client, ticker, limit=100)
            _calculate_stock(
                engine,
                market_state,
                market_model_version,
                ticker,
                market_history,
                stock_history,
            )
            successes += 1
        except Exception as exc:
            logger.exception("%s: allocation failed", ticker)
            failures.append(f"{ticker}: {exc}")

    logger.info(
        "Daily allocation complete: market=%s model=%s successes=%d failures=%d",
        market_state.label,
        market_model_version,
        successes,
        len(failures),
    )
    if failures:
        for failure in failures:
            logger.error(failure)
        # Partial ticker failures are visible in logs but do not invalidate
        # successfully calculated allocations.
    if successes == 0 and tickers:
        sys.exit(1)


if __name__ == "__main__":
    main()
