"""
Strategy layer.

This module converts Layer 1 + Layer 2 allocation context and market data into
strategy intent. It does not decide portfolio sizing, leverage, or quantity.

Architecture:
    AllocationSignal + OHLCV
            |
            v
         Strategy
            |
            v
      StrategySignal
            |
            v
        RiskManager
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core.types import (
    AllocationSignal,
    MarketRegimeState,
    StrategySignal,
    TradeDirection,
)


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for strategy logic only.

    Portfolio allocation, leverage and maximum risk are intentionally absent.
    Those concerns belong to RiskManager.
    """

    min_confidence_threshold: float = 0.7
    atr_period: int = 14
    ema_period: int = 50
    low_vol_reward_risk: float = 2.5
    mid_vol_reward_risk: float = 2.0
    high_vol_reward_risk: float = 1.5
    low_vol_atr_multiplier: float = 3.0
    mid_vol_atr_multiplier: float = 0.5
    high_vol_atr_multiplier: float = 1.0


class BaseStrategy(ABC):
    """Base class for strategies that produce trade intent only."""

    def __init__(self, config: StrategyConfig):
        self.config = config

    @abstractmethod
    def generate_signal(
        self,
        ticker: str,
        bars: pd.DataFrame,
        allocation: AllocationSignal,
    ) -> Optional[StrategySignal]:
        """Generate strategy intent for one ticker."""

    def _has_sufficient_history(self, bars: pd.DataFrame) -> bool:
        return len(bars) >= max(self.config.atr_period, self.config.ema_period)

    def _atr(self, bars: pd.DataFrame) -> float:
        high = bars["high"]
        low = bars["low"]
        prev_close = bars["close"].shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return float(true_range.rolling(self.config.atr_period).mean().iloc[-1])

    def _ema(self, bars: pd.DataFrame, period: int) -> float:
        if f"ema{period}" in bars.columns:
            return float(bars[f"ema{period}"].iloc[-1])
        return float(bars["close"].ewm(span=period, adjust=False).mean().iloc[-1])

    def calculate_stop_loss(
        self,
        bars: pd.DataFrame,
        atr_multiplier: float = 0.5,
    ) -> float:
        """Calculate a long-position stop from the latest close and ATR."""
        close = float(bars["close"].iloc[-1])
        atr = self._atr(bars)
        return close - (atr * atr_multiplier)

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        reward_risk: float = 2.0,
    ) -> float:
        """Calculate a long-position take-profit from the reward:risk ratio."""
        return entry_price + (entry_price - stop_loss) * reward_risk

    def _metadata(self, allocation: AllocationSignal) -> dict:
        regime = allocation.market_regime
        return {
            "market_regime": regime.label,
            "market_state_id": regime.state_id,
            "market_regime_probability": regime.probability,
            "market_regime_confirmed": regime.is_confirmed,
            "market_regime_flickering": regime.is_flickering,
            "allocation_multiplier": allocation.final_multiplier,
            "base_multiplier": allocation.base_multiplier,
            "stock_vol_scalar": allocation.vol_scalar,
            "beta": allocation.beta,
            "relative_vol": allocation.relative_vol,
        }


class LowVolBullStrategy(BaseStrategy):
    """Long trend strategy used in the calm volatility bucket.

    The strategy decides direction, entry and invalidation level. It does not
    prescribe how much capital may be allocated.
    """

    def generate_signal(
        self,
        ticker: str,
        bars: pd.DataFrame,
        allocation: AllocationSignal,
    ) -> Optional[StrategySignal]:
        if not self._has_sufficient_history(bars):
            return None

        close = float(bars["close"].iloc[-1])
        atr = self._atr(bars)
        ema_50 = self._ema(bars, self.config.ema_period)

        stop_from_price = close - atr * self.config.low_vol_atr_multiplier
        stop_from_ema = ema_50 - atr * (self.config.low_vol_atr_multiplier / 6)
        stop_loss = max(stop_from_price, stop_from_ema)
        take_profit = self.calculate_take_profit(
            close,
            stop_loss,
            self.config.low_vol_reward_risk,
        )

        trend = "above" if close > ema_50 else "below"
        regime = allocation.market_regime

        return StrategySignal(
            ticker=ticker,
            direction=TradeDirection.LONG,
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=regime.probability,
            strategy_name=type(self).__name__,
            timestamp=allocation.timestamp,
            reasoning=(
                f"Calm volatility regime ({regime.label}); price is {trend} "
                f"the {self.config.ema_period} EMA. Strategy produces long "
                "intent; portfolio sizing is delegated to RiskManager."
            ),
            metadata=self._metadata(allocation),
        )


class MidVolCautiousStrategy(BaseStrategy):
    """Long strategy for the moderate volatility bucket."""

    def generate_signal(
        self,
        ticker: str,
        bars: pd.DataFrame,
        allocation: AllocationSignal,
    ) -> Optional[StrategySignal]:
        if not self._has_sufficient_history(bars):
            return None

        close = float(bars["close"].iloc[-1])
        ema_50 = self._ema(bars, self.config.ema_period)
        atr = self._atr(bars)

        stop_loss = ema_50 - atr * self.config.mid_vol_atr_multiplier
        take_profit = self.calculate_take_profit(
            close,
            stop_loss,
            self.config.mid_vol_reward_risk,
        )
        trend_intact = close > ema_50
        trend_text = "intact" if trend_intact else "broken"
        regime = allocation.market_regime

        return StrategySignal(
            ticker=ticker,
            direction=TradeDirection.LONG,
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=regime.probability,
            strategy_name=type(self).__name__,
            timestamp=allocation.timestamp,
            reasoning=(
                f"Moderate volatility regime ({regime.label}); trend is "
                f"{trend_text} relative to the {self.config.ema_period} EMA. "
                "Strategy produces long intent; RiskManager determines size."
            ),
            metadata={
                **self._metadata(allocation),
                "trend_intact": trend_intact,
            },
        )


class HighVolDefensiveStrategy(BaseStrategy):
    """Long defensive strategy for the turbulent volatility bucket."""

    def generate_signal(
        self,
        ticker: str,
        bars: pd.DataFrame,
        allocation: AllocationSignal,
    ) -> Optional[StrategySignal]:
        if not self._has_sufficient_history(bars):
            return None

        close = float(bars["close"].iloc[-1])
        ema_50 = self._ema(bars, self.config.ema_period)
        atr = self._atr(bars)

        stop_loss = ema_50 - atr * self.config.high_vol_atr_multiplier
        take_profit = self.calculate_take_profit(
            close,
            stop_loss,
            self.config.high_vol_reward_risk,
        )
        regime = allocation.market_regime

        return StrategySignal(
            ticker=ticker,
            direction=TradeDirection.LONG,
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=regime.probability,
            strategy_name=type(self).__name__,
            timestamp=allocation.timestamp,
            reasoning=(
                f"Turbulent volatility regime ({regime.label}); using a wider "
                "invalidation level. Strategy remains directionally long, "
                "while RiskManager controls the allowed exposure."
            ),
            metadata=self._metadata(allocation),
        )


class StrategyOrchestrator:
    """Selects a strategy from the volatility bucket and emits intent.

    Layer 1/2 owns the exposure budget in AllocationSignal. The orchestrator
    never converts that budget into a position size.
    """

    _STRATEGY_BY_BUCKET = {
        "CALM": LowVolBullStrategy,
        "MODERATE": MidVolCautiousStrategy,
        "TURBULENT": HighVolDefensiveStrategy,
    }

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.low_vol_strategy = LowVolBullStrategy(self.config)
        self.mid_vol_strategy = MidVolCautiousStrategy(self.config)
        self.high_vol_strategy = HighVolDefensiveStrategy(self.config)

    def get_strategy_for_allocation(
        self,
        allocation: AllocationSignal,
    ) -> Optional[BaseStrategy]:
        """Return the strategy for the current volatility bucket."""
        bucket = allocation.market_regime.volatility_bucket
        strategy_type = self._STRATEGY_BY_BUCKET.get(bucket)
        if strategy_type is None:
            return None
        return strategy_type(self.config)

    def generate_signal(
        self,
        allocation: AllocationSignal,
        bars: pd.DataFrame,
    ) -> Optional[StrategySignal]:
        """Generate one strategy signal for one AllocationSignal."""
        if allocation.final_multiplier <= 0:
            return None

        strategy = self.get_strategy_for_allocation(allocation)
        if strategy is None:
            return None

        signal = strategy.generate_signal(allocation.ticker, bars, allocation)
        if signal is None:
            return None

        metadata = dict(signal.metadata)
        metadata["allocation_cap_pct"] = allocation.final_multiplier

        if (
            signal.confidence < self.config.min_confidence_threshold
            or allocation.is_flickering
            or not allocation.is_regime_confirmed
        ):
            metadata["uncertainty_mode"] = True
            signal = StrategySignal(
                ticker=signal.ticker,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                confidence=signal.confidence,
                strategy_name=signal.strategy_name,
                timestamp=signal.timestamp,
                reasoning=signal.reasoning + " [UNCERTAINTY CONTEXT]",
                metadata=metadata,
            )
        else:
            signal = StrategySignal(
                ticker=signal.ticker,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                confidence=signal.confidence,
                strategy_name=signal.strategy_name,
                timestamp=signal.timestamp,
                reasoning=signal.reasoning,
                metadata=metadata,
            )

        return signal

    def generate_signals(
        self,
        allocations: dict[str, AllocationSignal],
        bars_dict: dict[str, pd.DataFrame],
    ) -> list[StrategySignal]:
        """Generate signals for all tickers with a valid allocation."""
        signals: list[StrategySignal] = []

        for ticker, allocation in allocations.items():
            bars = bars_dict.get(ticker)
            if bars is None:
                continue

            signal = self.generate_signal(allocation, bars)
            if signal is not None:
                signals.append(signal)

        return signals

    def get_volatility_ranking(self) -> list[tuple[str, str]]:
        """Return the static strategy mapping by volatility bucket."""
        return [
            (bucket, strategy.__name__)
            for bucket, strategy in self._STRATEGY_BY_BUCKET.items()
        ]


def create_default_config() -> StrategyConfig:
    """Create default strategy configuration."""
    return StrategyConfig()


def create_orchestrator(
    config: Optional[StrategyConfig] = None,
) -> StrategyOrchestrator:
    """Create the strategy orchestrator using the canonical contracts."""
    return StrategyOrchestrator(config)
