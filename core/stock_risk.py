"""Layer 2: per-stock risk model conditioned on the market regime.

Layer 1 is the single market-wide HMM in :mod:`core.hmm_market`.
Layer 2 is deliberately NOT another HMM.  It converts stock-specific
volatility and beta characteristics into a bounded scalar and combines it
with the Layer 1 base multiplier in ``AllocationEngine``.

The model is deterministic and versioned.  Its JSON artifact contains the
parameters required to reproduce the calculation; runtime observations are
stored separately as allocation results.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from core.types import MarketRegimeState, StockVolatilityProfile

MODEL_TYPE = "StockRiskModel"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StockRiskConfig:
    vol_window: int = 21
    beta_window: int = 63
    min_scalar: float = 0.25
    max_scalar: float = 1.0
    beta_penalty_strength: float = 0.30
    turbulent_beta_penalty: float = 1.25


class StockRiskModel:
    """Calculate a per-stock risk scalar using market context.

    ``step`` consumes synchronized stock/index closes.  Once warmed up it
    returns a ``StockVolatilityProfile``.  ``calculate_scalar`` applies a
    small additional regime-sensitive penalty to the profile.  The Layer 1
    multiplier is never hidden inside the stock model; composition happens in
    ``AllocationEngine`` so the two layers remain independently auditable.
    """

    def __init__(self, ticker: str, config: StockRiskConfig | None = None):
        self.ticker = ticker.upper()
        self.config = config or StockRiskConfig()
        if self.config.vol_window < 2 or self.config.beta_window < 10:
            raise ValueError("Invalid volatility/beta windows")
        self.stock_prices: list[float] = []
        self.market_prices: list[float] = []
        self.timestamps: list[datetime] = []
        self.current_profile: Optional[StockVolatilityProfile] = None
        self.created_at = datetime.now(timezone.utc)

    def step(
        self,
        stock_close: float,
        market_close: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[StockVolatilityProfile]:
        if stock_close <= 0 or market_close <= 0:
            raise ValueError("Prices must be positive")
        timestamp = timestamp or datetime.now(timezone.utc)
        self.stock_prices.append(float(stock_close))
        self.market_prices.append(float(market_close))
        self.timestamps.append(timestamp)

        max_len = self.config.beta_window + 1
        if len(self.stock_prices) > max_len:
            self.stock_prices = self.stock_prices[-max_len:]
            self.market_prices = self.market_prices[-max_len:]
            self.timestamps = self.timestamps[-max_len:]

        if len(self.stock_prices) < self.config.beta_window + 1:
            return None

        stock_returns = np.diff(np.log(np.asarray(self.stock_prices)))
        market_returns = np.diff(np.log(np.asarray(self.market_prices)))
        stock_vol = self._realized_volatility(stock_returns)
        market_vol = self._realized_volatility(market_returns)
        relative_vol = stock_vol / market_vol if market_vol > 0 else 1.0
        beta = self._rolling_beta(stock_returns, market_returns)
        beta = float(np.clip(beta, 0.0, 5.0))
        scalar = self._base_scalar(relative_vol, beta)
        self.current_profile = StockVolatilityProfile(
            ticker=self.ticker,
            realised_vol=float(stock_vol),
            market_vol=float(market_vol),
            relative_vol=float(relative_vol),
            beta=beta,
            vol_scalar=float(scalar),
            timestamp=timestamp,
        )
        return self.current_profile

    def calculate_scalar(
        self,
        profile: StockVolatilityProfile | None = None,
        market_regime: MarketRegimeState | None = None,
    ) -> float:
        profile = profile or self.current_profile
        if profile is None:
            raise RuntimeError("Stock risk model is not warmed up")
        scalar = self._base_scalar(profile.relative_vol, profile.beta)
        if market_regime is not None and market_regime.volatility_bucket == "TURBULENT":
            if profile.beta > 1.0:
                extra = 1.0 / (1.0 + (profile.beta - 1.0) * self.config.beta_penalty_strength * self.config.turbulent_beta_penalty)
                scalar *= extra
        return float(np.clip(scalar, self.config.min_scalar, self.config.max_scalar))

    def _realized_volatility(self, returns: np.ndarray) -> float:
        recent = returns[-self.config.vol_window:]
        if len(recent) < 2:
            return 0.0
        return float(np.std(recent, ddof=1) * np.sqrt(252))

    def _rolling_beta(self, stock_returns: np.ndarray, market_returns: np.ndarray) -> float:
        recent_stock = stock_returns[-self.config.beta_window:]
        recent_market = market_returns[-self.config.beta_window:]
        if len(recent_stock) < 10:
            return 1.0
        variance = float(np.var(recent_market, ddof=1))
        if variance <= 0:
            return 1.0
        covariance = float(np.cov(recent_stock, recent_market, ddof=1)[0, 1])
        beta = covariance / variance
        return beta if np.isfinite(beta) else 1.0

    def _base_scalar(self, relative_vol: float, beta: float) -> float:
        vol_penalty = 1.0 / max(relative_vol, 1.0)
        beta_penalty = 1.0 / (1.0 + (beta - 1.0) * self.config.beta_penalty_strength) if beta > 1.0 else 1.0
        return float(np.clip(vol_penalty * beta_penalty, self.config.min_scalar, self.config.max_scalar))

    def get_profile(self) -> Optional[StockVolatilityProfile]:
        return self.current_profile

    def is_warmed_up(self) -> bool:
        return len(self.stock_prices) >= self.config.beta_window + 1

    def reset(self) -> None:
        self.stock_prices.clear()
        self.market_prices.clear()
        self.timestamps.clear()
        self.current_profile = None

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_type": MODEL_TYPE,
            "ticker": self.ticker,
            "created_at": self.created_at.isoformat(),
            "config": asdict(self.config),
        }

    def save_model(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_model(cls, path: str | Path) -> "StockRiskModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_type") != MODEL_TYPE:
            raise ValueError(f"Unsupported model_type: {payload.get('model_type')}")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {payload.get('schema_version')}")
        model = cls(payload["ticker"], StockRiskConfig(**payload["config"]))
        if payload.get("created_at"):
            model.created_at = datetime.fromisoformat(payload["created_at"])
        return model


class AllocationEngine:
    """Compose Layer 1 and Layer 2 into an auditable final multiplier."""

    @staticmethod
    def calculate(
        market_regime: MarketRegimeState,
        stock_model: StockRiskModel,
        profile: StockVolatilityProfile | None = None,
    ) -> float:
        stock_scalar = stock_model.calculate_scalar(profile, market_regime)
        return round(float(np.clip(market_regime.base_multiplier * stock_scalar, 0.0, 1.0)), 4)
