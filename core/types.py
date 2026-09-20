"""Canonical data contracts for the two-layer regime/allocation system.

This module contains data structures only; business logic belongs in the
market regime, stock risk, allocation and risk modules.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


@dataclass(frozen=True)
class RegimeInfo:
    regime_id: int
    regime_name: str
    volatility_rank: int
    expected_return: float
    expected_volatility: float


@dataclass(frozen=True)
class MarketRegimeState:
    label: str
    state_id: int
    volatility_bucket: str
    base_multiplier: float
    probability: float
    state_probabilities: list[float]
    timestamp: datetime
    is_confirmed: bool
    consecutive_bars: int
    is_flickering: bool


@dataclass(frozen=True)
class StockVolatilityProfile:
    ticker: str
    realised_vol: float
    market_vol: float
    relative_vol: float
    beta: float
    vol_scalar: float
    timestamp: datetime


@dataclass(frozen=True)
class AllocationSignal:
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


class TradeDirection(Enum):
    """Direction produced by a strategy and consumed by the risk layer."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class RiskDecisionStatus(Enum):
    """Outcome of risk validation."""

    APPROVED = "APPROVED"
    MODIFIED = "MODIFIED"
    REJECTED = "REJECTED"


class CircuitBreakerLevel(Enum):
    """Risk-manager circuit-breaker state exposed by the contract."""

    NONE = "NONE"
    REDUCE_SIZE = "REDUCE_SIZE"
    HALT_DAY = "HALT_DAY"
    HALT_WEEK = "HALT_WEEK"
    HALT_ALL = "HALT_ALL"


@dataclass(frozen=True)
class StrategySignal:
    """Strategy intent before portfolio-level risk sizing.

    Strategy decides what to trade and where the invalidation level is.
    It does not decide portfolio allocation, leverage, or final quantity.
    Those concerns belong to RiskManager.
    """

    ticker: str
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    confidence: float
    strategy_name: str
    timestamp: datetime
    reasoning: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskDecision:
    """Final risk-layer decision for a StrategySignal.

    The decision contains the executable sizing result but does not place an
    order. Execution remains a separate downstream concern.
    """

    status: RiskDecisionStatus
    strategy_signal: Optional[StrategySignal]
    quantity: Optional[float]
    notional: Optional[float]
    risk_amount: Optional[float]
    risk_pct: Optional[float]
    allocation_cap_pct: Optional[float]
    allocation_cap_notional: Optional[float]
    rejection_reason: Optional[str]
    modifications: tuple[str, ...]
    circuit_breaker_level: CircuitBreakerLevel
    risk_metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class StockRiskModelMetadata:
    ticker: str
    model_version: str
    schema_version: int
    market_model_version: Optional[str]
    created_at: datetime
    training_start: Optional[datetime]
    training_end: Optional[datetime]
