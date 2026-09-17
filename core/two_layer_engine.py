"""Canonical orchestration for Layer 1 market regime + Layer 2 stock risk."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from core.hmm_market import MarketRegimeClassifier
from core.stock_risk import AllocationEngine, StockRiskConfig, StockRiskModel
from core.types import AllocationSignal, MarketRegimeState
from storage.client import storeAllocationResult, storeStockRiskModel


class TwoLayerRegimeEngine:
    """Run one market HMM and one risk model per stock.

    The engine never trains a separate HMM per stock.  A stock model is a
    deterministic, versioned risk model whose output is conditioned on the
    current Layer 1 market state.
    """

    def __init__(
        self,
        market_model: MarketRegimeClassifier,
        model_dir: str | Path = "models/stocks",
        stock_config: StockRiskConfig | None = None,
    ) -> None:
        self.market_model = market_model
        self.model_dir = Path(model_dir)
        self.stock_config = stock_config or StockRiskConfig()
        self.stock_models: dict[str, StockRiskModel] = {}

    def get_or_create_stock_model(self, ticker: str) -> StockRiskModel:
        ticker = ticker.upper()
        if ticker not in self.stock_models:
            model_path = self.model_dir / ticker / "risk_model.json"
            if model_path.exists():
                model = StockRiskModel.load_model(model_path)
                # The runtime market model is authoritative for this session.
                model.market_model_version = self.market_model_version
            else:
                model = StockRiskModel(ticker, self.stock_config, self.market_model_version)
                model.save_model(model_path)
                storeStockRiskModel(model)
            self.stock_models[ticker] = model
        return self.stock_models[ticker]

    @property
    def market_model_version(self) -> str:
        """Stable identifier derived from the Layer 1 training artifact."""
        training = self.market_model.training_date.isoformat() if self.market_model.training_date else "unfitted"
        ticker = self.market_model.market_ticker or "UNKNOWN"
        return f"{ticker}:{training}:{self.market_model.n_states}:{self.market_model.bic_score:.8f}"

    def step_market(self, feature_bar) -> MarketRegimeState:
        return self.market_model.step(feature_bar)

    def step_stock(
        self,
        market_state: MarketRegimeState,
        ticker: str,
        stock_close: float,
        market_close: float,
        timestamp=None,
        persist: bool = True,
    ) -> AllocationSignal | None:
        model = self.get_or_create_stock_model(ticker)
        profile = model.step(stock_close, market_close, timestamp)
        if profile is None:
            return None
        signal = AllocationEngine.calculate(market_state, model, profile)
        if persist:
            storeAllocationResult(signal)
        return signal

    def step_universe(
        self,
        market_feature_bar,
        stock_bars: Iterable[tuple[str, float]],
        market_close: float,
        timestamp=None,
        persist: bool = True,
    ) -> list[AllocationSignal]:
        market_state = self.step_market(market_feature_bar)
        results: list[AllocationSignal] = []
        for ticker, stock_close in stock_bars:
            signal = self.step_stock(market_state, ticker, stock_close, market_close, timestamp, persist)
            if signal is not None:
                results.append(signal)
        return results
