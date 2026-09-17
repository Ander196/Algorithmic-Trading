"""
core/types.py - Shared dataclasses for the two-layer HMM regime detection system.

This module contains only data structures - no business logic.
These types are used across the hmm_market, stock_adjuster, and hmm_model modules.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class RegimeInfo:
    """
    Metadata for each market regime state.

    Contains static information about what each regime means for trading,
    including expected volatility and recommended position sizing.
    """
    regime_id: int
    regime_name: str  # CALM, MODERATE, TURBULENT
    volatility_rank: int  # 0 = lowest volatility, higher = more volatile
    expected_return: float  # Expected return modifier (1.0 = normal)
    expected_volatility: float  # Annualized volatility expectation


@dataclass
class MarketRegimeState:
    """
    Current state of market-wide regime detection (Layer 1).

    This represents the macro-level market environment detected by the
    MarketRegimeClassifier. The base_multiplier is used to size positions
    based on overall market volatility.

    Attributes:
        label: Regime name (CALM, MODERATE, TURBULENT)
        state_id: HMM state index (0-based)
        volatility_bucket: Volatility classification
        base_multiplier: Position size multiplier from regime (1.0, 0.75, 0.5)
        probability: Probability of current state
        state_probabilities: Full probability distribution over all states
        timestamp: When this state was computed
        is_confirmed: Whether regime has persisted long enough to act
        consecutive_bars: Bars spent in current regime (for stability check)
        is_flickering: Whether regime is changing too frequently
    """
    label: str
    state_id: int
    volatility_bucket: str  # CALM, MODERATE, TURBULENT
    base_multiplier: float
    probability: float
    state_probabilities: list[float]
    timestamp: datetime
    is_confirmed: bool
    consecutive_bars: int
    is_flickering: bool


@dataclass
class StockVolatilityProfile:
    """
    Per-stock volatility characteristics (Layer 2 micro adjustments).

    Tracks how a specific stock's realized volatility compares to the market,
    and its beta relationship. Used to adjust the base multiplier up or down.

    Attributes:
        ticker: Stock symbol
        realised_vol: Annualized realized volatility of the stock
        market_vol: Annualized realized volatility of the market index
        relative_vol: Stock vol / market vol (ratio)
        beta: Rolling beta of stock vs market
        vol_scalar: Final volatility adjustment factor
        timestamp: When this profile was computed
    """
    ticker: str
    realised_vol: float
    market_vol: float
    relative_vol: float
    beta: float
    vol_scalar: float
    timestamp: datetime


@dataclass
class AllocationSignal:
    """
    Final allocation signal combining both layers.

    This is the output that the strategy layer uses to size positions.
    The final_multiplier is applied to the base position size to get the
    actual dollar allocation for this ticker.

    Attributes:
        ticker: Stock symbol
        final_multiplier: Combined multiplier (base × vol_scalar)
        market_regime: Current MarketRegimeState
        base_multiplier: Multiplier from Layer 1 only
        vol_scalar: Multiplier from Layer 2 only
        regime_label: Current regime name
        beta: Stock's rolling beta
        relative_vol: Stock's relative volatility
        is_regime_confirmed: Whether macro regime is stable
        is_flickering: Whether macro regime is flickering
        timestamp: When this signal was generated
        reasoning: Human-readable explanation of the signal
    """
    ticker: str
    final_multiplier: float
    market_regime: MarketRegimeState
    base_multiplier: float
    vol_scalar: float
    regime_label: str
    beta: float
    relative_vol: float
    is_regime_confirmed: bool
    is_flickering: bool
    timestamp: datetime
    reasoning: str