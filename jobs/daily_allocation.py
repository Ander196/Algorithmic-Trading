#!/usr/bin/env python
"""Daily Layer 1 inference + Layer 2 allocation calculation.

This job deliberately stops after AllocationSignal persistence. It does not
select securities to trade and does not invoke Strategy/RiskManager/execution.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from core.hmm_market import MarketRegimeClassifier
from core.stock_risk import AllocationEngine, StockRiskConfig, StockRiskModel
from data.data_loader import getActiveTickers
from jobs.common import fetch_price_history, get_supabase_client
from storage.client import storeAllocationResult, storeStockRiskModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STOCK_HISTORY_LOOKBACK = 160


def _market_model_version(path: Path, payload: dict) -> str:
    version = payload.get("metadata", {}).get("model_version")
    if version:
        return version
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


def _load_active_market_model() -> tuple[MarketRegimeClassifier, str]:
    path = Path(os.getenv("MARKET_MODEL_PATH", "models/market/current.json"))
    if not path.exists():
        raise RuntimeError(f"Active Layer 1 model not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    model = MarketRegimeClassifier().load_model(str(path))
    return model, _market_model_version(path, payload)


def _warm_market_model(model: MarketRegimeClassifier, history: pd.DataFrame):
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
    market_state,
    market_model_version: str,
    ticker: str,
    market_history: pd.DataFrame,
    stock_history: pd.DataFrame,
    model_dir: Path,
):
    joined = market_history[["price_date", "close"]].rename(
        columns={"close": "market_close"}
    ).merge(
        stock_history[["price_date", "close"]].rename(columns={"close": "stock_close"}),
        on="price_date",
        how="inner",
    ).sort_values("price_date")

    config = StockRiskConfig()
    minimum = config.beta_window + 1
    if len(joined) < minimum:
        raise RuntimeError(f"only {len(joined)} common sessions; need {minimum}")

    model_path = model_dir / ticker / "risk_model.json"
    if model_path.exists():
        model = StockRiskModel.load_model(str(model_path))
        model.market_model_version = market_model_version
    else:
        model = StockRiskModel(ticker, config, market_model_version)

    model.reset()
    for _, row in joined.iloc[:-1].iterrows():
        model.step(
            float(row["stock_close"]),
            float(row["market_close"]),
            row["price_date"].to_pydatetime(),
        )

    last = joined.iloc[-1]
    profile = model.step(
        float(last["stock_close"]),
        float(last["market_close"]),
        last["price_date"].to_pydatetime(),
    )
    if profile is None:
        raise RuntimeError("Layer 2 did not warm up")

    signal = AllocationEngine.calculate(market_state, model, profile)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    storeStockRiskModel(model)
    storeAllocationResult(
        signal,
        market_model_version=market_model_version,
        stock_model_version=model.model_version,
    )
    return signal


def main() -> None:
    client = get_supabase_client()
    market_ticker = os.getenv("MARKET_TICKER", "SPY").upper()
    model_dir = Path(os.getenv("STOCK_MODEL_DIR", "models/stocks"))

    market_model, market_model_version = _load_active_market_model()
    market_history = fetch_price_history(
        client,
        market_ticker,
        limit=int(os.getenv("MARKET_ALLOCATION_HISTORY_BARS", "350")),
    )
    market_state = _warm_market_model(market_model, market_history)

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
    successes = 0
    failures: list[str] = []

    for ticker in tickers:
        try:
            stock_history = fetch_price_history(
                client,
                ticker,
                limit=int(os.getenv("STOCK_ALLOCATION_HISTORY_BARS", str(STOCK_HISTORY_LOOKBACK))),
            )
            _calculate_stock(
                market_state,
                market_model_version,
                ticker,
                market_history,
                stock_history,
                model_dir,
            )
            successes += 1
        except Exception as exc:
            logger.exception("%s: allocation failed", ticker)
            failures.append(f"{ticker}: {exc}")

    logger.info(
        "Daily allocation complete: market=%s model=%s successes=%d failures=%d",
        market_state.label, market_model_version, successes, len(failures),
    )
    for failure in failures:
        logger.error(failure)

    if successes == 0 and tickers:
        sys.exit(1)


if __name__ == "__main__":
    main()
