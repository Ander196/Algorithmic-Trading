from datetime import datetime, timezone

import pandas as pd

from core.alpha_selector import AlphaSelectionConfig, AlphaSelector
from core.types import AllocationSignal, MarketRegimeState


def _allocation(ticker: str, multiplier: float) -> AllocationSignal:
    regime = MarketRegimeState(
        label="CALM",
        state_id=0,
        volatility_bucket="CALM",
        base_multiplier=1.0,
        probability=0.9,
        state_probabilities=[0.9, 0.1],
        timestamp=datetime.now(timezone.utc),
        is_confirmed=True,
        consecutive_bars=5,
        is_flickering=False,
    )
    return AllocationSignal(
        ticker=ticker,
        final_multiplier=multiplier,
        market_regime=regime,
        base_multiplier=1.0,
        vol_scalar=multiplier,
        regime_label="CALM",
        beta=1.0,
        relative_vol=1.0,
        is_regime_confirmed=True,
        is_flickering=False,
        timestamp=datetime.now(timezone.utc),
        reasoning="test",
    )


def test_selector_filters_and_ranks_candidates() -> None:
    predictions = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "alpha_score": [3.0, 2.0, 4.0, 1.0],
            "probability_positive": [0.80, 0.70, 0.40, 0.90],
        }
    )
    allocations = {
        "AAA": _allocation("AAA", 1.0),
        "BBB": _allocation("BBB", 0.0),
        "CCC": _allocation("CCC", 1.0),
        "DDD": _allocation("DDD", 1.0),
    }

    result = AlphaSelector(
        AlphaSelectionConfig(top_n=2, min_probability_positive=0.50)
    ).rank(predictions, allocations)

    assert result["ticker"].tolist() == ["AAA", "DDD"]
    assert result["alpha_rank"].tolist() == [1, 2]


def test_selector_is_deterministic_on_ties() -> None:
    predictions = pd.DataFrame(
        {
            "ticker": ["BBB", "AAA"],
            "alpha_score": [1.0, 1.0],
            "probability_positive": [0.7, 0.7],
        }
    )
    allocations = {
        "AAA": _allocation("AAA", 1.0),
        "BBB": _allocation("BBB", 1.0),
    }

    result = AlphaSelector().rank(predictions, allocations)
    assert result["ticker"].tolist() == ["AAA", "BBB"]
