"""
hmm_regime - Two-Layer HMM Volatility Regime Detection Package

This package provides a professional market regime detection system using
a two-layer Hidden Markov Model architecture:

- Layer 1 (Macro): Market-wide volatility regime detection using Gaussian HMM
- Layer 2 (Micro): Per-stock volatility/beta adjustments

Exports:
    TwoLayerRegimeEngine: Main orchestration class
    MarketRegimeClassifier: Layer 1 - Market regime classifier
    StockVolatilityAdjuster: Layer 2 - Per-stock adjuster
    RegimeInfo: Regime metadata dataclass
    MarketRegimeState: Market regime state dataclass
    StockVolatilityProfile: Stock volatility profile dataclass
    AllocationSignal: Final allocation signal dataclass
"""
from core.hmm_market import MarketRegimeClassifier
from core.hmm_model import TwoLayerRegimeEngine
from core.stock_adjuster import StockVolatilityAdjuster
from core.types import (
    AllocationSignal,
    MarketRegimeState,
    RegimeInfo,
    StockVolatilityProfile,
)

__all__ = [
    "TwoLayerRegimeEngine",
    "MarketRegimeClassifier",
    "StockVolatilityAdjuster",
    "RegimeInfo",
    "MarketRegimeState",
    "StockVolatilityProfile",
    "AllocationSignal",
]

__version__ = "1.0.0"