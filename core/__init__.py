"""Canonical two-layer market-regime and stock-risk package."""
from core.hmm_market import MarketRegimeClassifier
from core.stock_risk import AllocationEngine, StockRiskConfig, StockRiskModel
from core.types import AllocationSignal, MarketRegimeState, RegimeInfo, StockVolatilityProfile

__all__ = [
    "AllocationEngine",
    "AllocationSignal",
    "MarketRegimeClassifier",
    "MarketRegimeState",
    "RegimeInfo",
    "StockRiskConfig",
    "StockRiskModel",
    "StockVolatilityProfile",
]

__version__ = "2.0.0"
