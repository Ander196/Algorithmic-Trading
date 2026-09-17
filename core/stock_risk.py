"""Layer 2: per-stock risk model conditioned on the Layer 1 regime."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from core.types import AllocationSignal, MarketRegimeState, StockVolatilityProfile

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
    """Per-stock volatility/beta model; deliberately not another HMM."""

    def __init__(self, ticker: str, config: StockRiskConfig | None = None, market_model_version: str | None = None):
        self.ticker = ticker.upper()
        self.config = config or StockRiskConfig()
        if self.config.vol_window < 2 or self.config.beta_window < 10:
            raise ValueError("Invalid volatility/beta windows")
        self.market_model_version = market_model_version
        self.stock_prices: list[float] = []
        self.market_prices: list[float] = []
        self.timestamps: list[datetime] = []
        self.current_profile: Optional[StockVolatilityProfile] = None
        self.created_at = datetime.now(timezone.utc)

    @property
    def model_version(self) -> str:
        canonical = json.dumps(self.to_dict(include_version=False), sort_keys=True).encode()
        return hashlib.sha256(canonical).hexdigest()[:16]

    def step(self, stock_close: float, market_close: float, timestamp: Optional[datetime] = None) -> Optional[StockVolatilityProfile]:
        if stock_close <= 0 or market_close <= 0:
            raise ValueError("Prices must be positive")
        timestamp = timestamp or datetime.now(timezone.utc)
        self.stock_prices.append(float(stock_close))
        self.market_prices.append(float(market_close))
        self.timestamps.append(timestamp)
        max_len = self.config.beta_window + 1
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
        beta = float(np.clip(self._rolling_beta(stock_returns, market_returns), 0.0, 5.0))
        self.current_profile = StockVolatilityProfile(
            ticker=self.ticker,
            realised_vol=float(stock_vol),
            market_vol=float(market_vol),
            relative_vol=float(relative_vol),
            beta=beta,
            vol_scalar=self._base_scalar(relative_vol, beta),
            timestamp=timestamp,
        )
        return self.current_profile

    def calculate_scalar(self, profile: StockVolatilityProfile | None = None, market_regime: MarketRegimeState | None = None) -> float:
        profile = profile or self.current_profile
        if profile is None:
            raise RuntimeError("Stock risk model is not warmed up")
        scalar = self._base_scalar(profile.relative_vol, profile.beta)
        if market_regime and market_regime.volatility_bucket == "TURBULENT" and profile.beta > 1.0:
            scalar *= 1.0 / (1.0 + (profile.beta - 1.0) * self.config.beta_penalty_strength * self.config.turbulent_beta_penalty)
        return float(np.clip(scalar, self.config.min_scalar, self.config.max_scalar))

    def _realized_volatility(self, returns: np.ndarray) -> float:
        recent = returns[-self.config.vol_window:]
        return float(np.std(recent, ddof=1) * np.sqrt(252)) if len(recent) >= 2 else 0.0

    def _rolling_beta(self, stock_returns: np.ndarray, market_returns: np.ndarray) -> float:
        stock_rets = stock_returns[-self.config.beta_window:]
        market_rets = market_returns[-self.config.beta_window:]
        if len(stock_rets) < 10:
            return 1.0
        variance = float(np.var(market_rets, ddof=1))
        if variance <= 0:
            return 1.0
        beta = float(np.cov(stock_rets, market_rets, ddof=1)[0, 1] / variance)
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
        self.stock_prices.clear(); self.market_prices.clear(); self.timestamps.clear(); self.current_profile = None

    def to_dict(self, include_version: bool = True) -> dict:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "model_type": MODEL_TYPE,
            "ticker": self.ticker,
            "market_model_version": self.market_model_version,
            "created_at": self.created_at.isoformat(),
            "config": asdict(self.config),
        }
        if include_version:
            payload["model_version"] = self.model_version
        return payload

    def save_model(self, path: str | Path) -> None:
        destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_model(cls, path: str | Path) -> "StockRiskModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_type") != MODEL_TYPE or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported StockRiskModel JSON schema")
        model = cls(payload["ticker"], StockRiskConfig(**payload["config"]), payload.get("market_model_version"))
        if payload.get("created_at"):
            model.created_at = datetime.fromisoformat(payload["created_at"])
        expected = payload.get("model_version")
        if expected and expected != model.model_version:
            raise ValueError("StockRiskModel JSON model_version does not match its configuration")
        return model


class AllocationEngine:
    """Pure composition of Layer 1 and Layer 2."""

    @staticmethod
    def calculate(market_regime: MarketRegimeState, stock_model: StockRiskModel, profile: StockVolatilityProfile | None = None) -> AllocationSignal:
        profile = profile or stock_model.get_profile()
        if profile is None:
            raise RuntimeError("Stock risk model is not warmed up")
        scalar = stock_model.calculate_scalar(profile, market_regime)
        final = round(float(np.clip(market_regime.base_multiplier * scalar, 0.0, 1.0)), 4)
        return AllocationSignal(
            ticker=stock_model.ticker,
            final_multiplier=final,
            market_regime=market_regime,
            base_multiplier=market_regime.base_multiplier,
            vol_scalar=scalar,
            regime_label=market_regime.volatility_bucket,
            beta=profile.beta,
            relative_vol=profile.relative_vol,
            is_regime_confirmed=market_regime.is_confirmed,
            is_flickering=market_regime.is_flickering,
            timestamp=profile.timestamp,
            reasoning=(f"Layer1 {market_regime.volatility_bucket} base={market_regime.base_multiplier:.4f}; "
                       f"Layer2 relative_vol={profile.relative_vol:.4f}, beta={profile.beta:.4f}, scalar={scalar:.4f}"),
        )
