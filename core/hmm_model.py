"""Compatibility facade for the unified regime architecture.

The duplicate HMM implementation that previously lived here has been retired.
The canonical Layer 1 implementation is ``core.hmm_market.MarketRegimeClassifier``.
Layer 2 is ``core.stock_risk.StockRiskModel`` and is deliberately not an HMM.

New code should import from ``core.hmm_market`` and ``core.stock_risk`` directly.
This facade keeps the most common legacy imports working during migration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from core.hmm_market import MarketRegimeClassifier
from core.stock_risk import AllocationEngine, StockRiskConfig, StockRiskModel
from core.types import MarketRegimeState

HMMVolatilityClassifier = MarketRegimeClassifier
RegimeState = MarketRegimeState


def train_hmm(
    df: pd.DataFrame,
    model_path: str | None = None,
    market_ticker: str = "UNKNOWN",
    **kwargs,
) -> MarketRegimeClassifier:
    """Legacy helper delegating to the canonical Layer 1 classifier."""
    supported = {
        key: kwargs[key]
        for key in ("n_init", "min_confidence", "stability_bars", "flicker_window", "flicker_threshold")
        if key in kwargs
    }
    model = MarketRegimeClassifier(**supported).fit(df, market_ticker=market_ticker)
    if model_path:
        model.save_model(model_path)
    return model


def run_train_only(
    df: pd.DataFrame,
    model_path: str | None = None,
    market_ticker: str = "UNKNOWN",
    **kwargs,
) -> MarketRegimeClassifier:
    """Legacy train-only entry point delegating to Layer 1."""
    return train_hmm(df, model_path=model_path, market_ticker=market_ticker, **kwargs)


__all__ = [
    "AllocationEngine",
    "HMMVolatilityClassifier",
    "MarketRegimeClassifier",
    "MarketRegimeState",
    "RegimeState",
    "StockRiskConfig",
    "StockRiskModel",
    "train_hmm",
    "run_train_only",
]
