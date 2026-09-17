"""Canonical data contracts for the two-layer regime/allocation system.

This module contains data structures only; business logic belongs in the
market regime, stock risk, allocation and risk modules.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


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


@dataclass(frozen=True)
class StockRiskModelMetadata:
    ticker: str
    model_version: str
    schema_version: int
    market_model_version: Optional[str]
    created_at: datetime
    training_start: Optional[datetime]
    training_end: Optional[datetime]
