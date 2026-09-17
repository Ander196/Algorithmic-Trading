"""Canonical two-layer market-regime and stock-risk package."""
from core.hmm_market import MarketRegimeClassifier
from core.stock_risk import AllocationEngine, StockRiskConfig, StockRiskModel
from core.two_layer_engine import TwoLayerRegimeEngine
from core.types import AllocationSignal, MarketRegimeState, RegimeInfo, StockVolatilityProfile

__all__ = [
    "AllocationEngine",
    "AllocationSignal",
    "MarketRegimeClassifier",
    "MarketRegimeState",
    "RegimeInfo",
    "StockRiskConfig",
    "StockVolatilityProfile",
    "StockRiskModel",
    "TwoLayerRegimeEngine",
]

__version__ = "2.0.0"
