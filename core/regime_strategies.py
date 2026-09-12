"""
Regime-Based Strategy Module

The allocation layer that sizes positions based on the HMM's volatility regime detection.
The HMM excels at detecting VOLATILITY ENVIRONMENTS, not market direction.

Design Insight:
- Low vol —> be fully invested (calm markets trend up)
- Mid vol —> stay invested if trend intact, reduce if not
- High vol —> Reduce but stay partially invested (catch V-shaped rebounds)

The edge comes from AVOIDING BIG DRAWDOWNS through vol-based sizing.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from core.hmm_model import RegimeInfo, RegimeState


class Direction(Enum):
    """Trade direction."""
    LONG = "LONG"
    FLAT = "FLAT"


@dataclass
class Signal:
    """Trading signal with position sizing information."""
    symbol: str
    direction: Direction
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: Optional[float] = None
    position_size_pct: float = 0.95
    leverage: float = 1.0
    regime_id: int = 0
    regime_name: str = "UNKNOWN"
    regime_probability: float = 0.0
    timestamp: Optional[datetime] = None
    reasoning: str = ""
    strategy_name: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Configuration for regime-based strategies."""
    min_confidence_threshold: float = 0.7
    rebalance_threshold: float = 0.10
    atr_period: int = 14
    ema_period: int = 50
    uncertainty_leverage: float = 1.0
    uncertainty_multiplier: float = 0.5


class BaseStrategy(ABC):
    """Abstract base class for regime-based strategies."""

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:
        """
        Generate a trading signal based on current regime and market conditions.

        Args:
            symbol: Ticker symbol
            bars: OHLCV data
            regime_state: Current regime detection state

        Returns:
            Signal if conditions are met, None otherwise
        """
        pass

    def calculate_stop_loss(
        self,
        bars: pd.DataFrame,
        method: str = "atr",
        atr_multiplier: float = 0.5,
    ) -> float:
        """Calculate stop loss price."""
        close = bars['close'].iloc[-1]

        if method == "atr" and 'atr' in bars.columns:
            atr = bars['atr'].iloc[-1]
            return close - (atr * atr_multiplier)
        elif method == "atr":
            high = bars['high']
            low = bars['low']
            prev_close = bars['close'].shift(1)

            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            return close - (atr * atr_multiplier)
        elif method == "ema":
            ema_50 = bars['ema50'].iloc[-1] if 'ema50' in bars.columns else close
            return ema_50 - (atr_multiplier * 0.5)
        return close * 0.95

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        reward_risk: float = 2.0,
    ) -> Optional[float]:
        """Calculate take profit price based on reward:risk ratio."""
        return entry_price + (entry_price - stop_loss) * reward_risk


class LowVolBullStrategy(BaseStrategy):
    """
    Low Volatility Bull Strategy.

    For regimes in the lowest third by expected_volatility.
    Calm markets trend upward - use leverage to compound returns.

    - Direction: LONG
    - Allocation: 95% of portfolio
    - Leverage: 1.25x (modest leverage in calm conditions)
    - Stop: max(price-3 ATR, 50 EMA -0.5 ATR)
    """

    def __init__(self, config: StrategyConfig):
        self.config = config

    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:
        """Generate signal for low volatility bull conditions."""
        close = bars['close'].iloc[-1]

        if len(bars) < max(self.config.atr_period, self.config.ema_period):
            return None

        stop_loss = self._calculate_stop_loss_low_vol(bars)
        take_profit = self.calculate_take_profit(close, stop_loss, reward_risk=2.5)

        return Signal(
            symbol=symbol,
            direction=Direction.LONG,
            confidence=regime_state.probability,
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=0.95,
            leverage=1.25,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=datetime.now(),
            reasoning=self._build_reasoning(bars, regime_state),
            strategy_name="LowVolBullStrategy",
        )

    def _calculate_stop_loss_low_vol(self, bars: pd.DataFrame) -> float:
        """Calculate stop loss for low volatility: max(price-3 ATR, 50 EMA -0.5 ATR)."""
        close = bars['close'].iloc[-1]
        ema_50 = bars['ema50'].iloc[-1] if 'ema50' in bars.columns else close

        high = bars['high']
        low = bars['low']
        prev_close = bars['close'].shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr_3 = tr.rolling(window=14).mean().iloc[-1] * 3

        stop_from_price = close - atr_3
        stop_from_ema = ema_50 - (atr_3 / 6)

        return max(stop_from_price, stop_from_ema)

    def _build_reasoning(self, bars: pd.DataFrame, regime_state: RegimeState) -> str:
        """Build reasoning string for the signal."""
        close = bars['close'].iloc[-1]
        ema_50 = bars['ema50'].iloc[-1] if 'ema50' in bars.columns else close
        trend = "above" if close > ema_50 else "below"

        return (
            f"Low volatility regime ({regime_state.label}) detected. "
            f"Calm markets trend upward - using 95% allocation with 1.25x leverage. "
            f"Price is {trend} 50 EMA. Confidence: {regime_state.probability:.2f}"
        )


class MidVolCautiousStrategy(BaseStrategy):
    """
    Mid Volatility Cautious Strategy.

    For regimes in the middle third by expected_volatility.
    Stay invested if trend intact, reduce if trend broken.

    - Direction: LONG
    - If price > 50 EMA: allocation 95%, leverage 1.0x (trend intact)
    - If price < 50 EMA: allocation 60%, leverage 1.0x (trend broken, reduce)
    - Stop: 50 EMA -0.5 ATR
    """

    def __init__(self, config: StrategyConfig):
        self.config = config

    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:
        """Generate signal for mid volatility cautious conditions."""
        close = bars['close'].iloc[-1]

        if len(bars) < max(self.config.atr_period, self.config.ema_period):
            return None

        ema_50 = bars['ema50'].iloc[-1] if 'ema50' in bars.columns else close

        if close > ema_50:
            position_size = 0.95
            leverage = 1.0
            reasoning_prefix = "Trend intact"
        else:
            position_size = 0.60
            leverage = 1.0
            reasoning_prefix = "Trend broken"

        stop_loss = self._calculate_stop_loss_mid_vol(bars, ema_50)
        take_profit = self.calculate_take_profit(close, stop_loss, reward_risk=2.0)

        return Signal(
            symbol=symbol,
            direction=Direction.LONG,
            confidence=regime_state.probability,
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size,
            leverage=leverage,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=datetime.now(),
            reasoning=self._build_reasoning(bars, regime_state, reasoning_prefix, close > ema_50),
            strategy_name="MidVolCautiousStrategy",
        )

    def _calculate_stop_loss_mid_vol(self, bars: pd.DataFrame, ema_50: float) -> float:
        """Calculate stop loss for mid volatility: 50 EMA -0.5 ATR."""
        high = bars['high']
        low = bars['low']
        prev_close = bars['close'].shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]

        return ema_50 - (atr * 0.5)

    def _build_reasoning(
        self,
        bars: pd.DataFrame,
        regime_state: RegimeState,
        prefix: str,
        trend_intact: bool,
    ) -> str:
        """Build reasoning string for the signal."""
        close = bars['close'].iloc[-1]
        ema_50 = bars['ema50'].iloc[-1] if 'ema50' in bars.columns else close
        trend = "above" if trend_intact else "below"

        return (
            f"Mid volatility regime ({regime_state.label}) detected. "
            f"{prefix} - price is {trend} 50 EMA. "
            f"Using {'95%' if trend_intact else '60%'} allocation. "
            f"Confidence: {regime_state.probability:.2f}"
        )


class HighVolDefensiveStrategy(BaseStrategy):
    """
    High Volatility Defensive Strategy.

    For regimes in the top third by expected_volatility.
    Stay partially invested to catch V-shaped rebounds.

    - Direction: LONG (NOT short)
    - Allocation: 60% of portfolio
    - Leverage: 1.0x
    - Stop: 50 EMA - 1.0 ATR (wider for volatile conditions)
    """

    def __init__(self, config: StrategyConfig):
        self.config = config

    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:
        """Generate signal for high volatility defensive conditions."""
        close = bars['close'].iloc[-1]

        if len(bars) < max(self.config.atr_period, self.config.ema_period):
            return None

        ema_50 = bars['ema50'].iloc[-1] if 'ema50' in bars.columns else close
        stop_loss = self._calculate_stop_loss_high_vol(bars, ema_50)
        take_profit = self.calculate_take_profit(close, stop_loss, reward_risk=1.5)

        return Signal(
            symbol=symbol,
            direction=Direction.LONG,
            confidence=regime_state.probability,
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=0.60,
            leverage=1.0,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=datetime.now(),
            reasoning=self._build_reasoning(bars, regime_state),
            strategy_name="HighVolDefensiveStrategy",
        )

    def _calculate_stop_loss_high_vol(self, bars: pd.DataFrame, ema_50: float) -> float:
        """Calculate stop loss for high volatility: 50 EMA -1.0 ATR (wider stop)."""
        high = bars['high']
        low = bars['low']
        prev_close = bars['close'].shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]

        return ema_50 - atr

    def _build_reasoning(self, bars: pd.DataFrame, regime_state: RegimeState) -> str:
        """Build reasoning string for the signal."""
        return (
            f"High volatility regime ({regime_state.label}) detected. "
            f"Reducing to 60% allocation with wider stop to catch rebounds. "
            f"Staying long to capture V-shaped recoveries. "
            f"Confidence: {regime_state.probability:.2f}"
        )


class StrategyOrchestrator:
    """
    Orchestrates regime-based strategies.

    Takes regime_infos from HMM, sorts by volatility (ascending),
    maps each regime to the appropriate strategy class based on vol rank.
    """

    def __init__(
        self,
        config: StrategyConfig,
        regime_infos: dict[int, RegimeInfo],
    ):
        """
        Initialize orchestrator.

        Args:
            config: Strategy configuration
            regime_infos: Dict of regime_id -> RegimeInfo from HMM
        """
        self.config = config
        self.low_vol_strategy = LowVolBullStrategy(config)
        self.mid_vol_strategy = MidVolCautiousStrategy(config)
        self.high_vol_strategy = HighVolDefensiveStrategy(config)
        self._vol_rank_to_strategy: dict[int, type[BaseStrategy]] = {}
        self._regime_id_to_vol_rank: dict[int, int] = {}
        self._current_allocation: dict[str, float] = {}

        self.update_regime_infos(regime_infos)

    def update_regime_infos(self, regime_infos: dict[int, RegimeInfo]) -> None:
        """
        Rebuild strategy mapping after HMM retrain.

        Sorts regimes by expected_volatility (ascending) and maps to strategies.
        This sort is INDEPENDENT of the label sort (which is by return).
        """
        self.regime_infos = regime_infos

        sorted_by_vol = sorted(
            regime_infos.items(),
            key=lambda x: x[1].expected_volatility
        )

        n_regimes = len(sorted_by_vol)
        self._regime_id_to_vol_rank = {}
        self._vol_rank_to_strategy = {}

        for rank, (regime_id, info) in enumerate(sorted_by_vol):
            position = rank / max(n_regimes - 1, 1)

            if position <= 0.33:
                strategy = self.low_vol_strategy
            elif position >= 0.67:
                strategy = self.high_vol_strategy
            else:
                strategy = self.mid_vol_strategy

            self._regime_id_to_vol_rank[regime_id] = rank
            self._vol_rank_to_strategy[rank] = type(strategy).__name__

    def get_strategy_for_regime(self, regime_id: int) -> BaseStrategy:
        """Get the strategy instance for a given regime ID."""
        vol_rank = self._regime_id_to_vol_rank.get(regime_id, 1)
        position = vol_rank / max(len(self._regime_id_to_vol_rank) - 1, 1)

        if position <= 0.33:
            return self.low_vol_strategy
        elif position >= 0.67:
            return self.high_vol_strategy
        else:
            return self.mid_vol_strategy

    def generate_signals(
        self,
        symbols: list[str],
        bars_dict: dict[str, pd.DataFrame],
        regime_state: RegimeState,
        is_flickering: bool = False,
    ) -> list[Signal]:
        """
        Generate signals for multiple symbols.

        Args:
            symbols: List of ticker symbols
            bars_dict: Dict of symbol -> OHLCV DataFrame
            regime_state: Current regime detection state
            is_flickering: Whether regime is flickering

        Returns:
            List of Signal objects
        """
        signals = []
        uncertainty_mode = self._check_uncertainty(regime_state, is_flickering)

        for symbol in symbols:
            if symbol not in bars_dict:
                continue

            bars = bars_dict[symbol]
            strategy = self.get_strategy_for_regime(regime_state.state_id)

            signal = strategy.generate_signal(symbol, bars, regime_state)

            if signal is None:
                continue

            if uncertainty_mode:
                signal = self._apply_uncertainty_adjustments(signal)

            if self._should_rebalance(symbol, signal.position_size_pct):
                signals.append(signal)
                self._current_allocation[symbol] = signal.position_size_pct * signal.leverage

        return signals

    def _check_uncertainty(self, regime_state: RegimeState, is_flickering: bool) -> bool:
        """Check if we should operate in uncertainty mode."""
        return (
            regime_state.probability < self.config.min_confidence_threshold
            or is_flickering
        )

    def _apply_uncertainty_adjustments(self, signal: Signal) -> Signal:
        """Apply uncertainty mode adjustments: halve position sizes, force leverage to 1.0."""
        signal.position_size_pct *= self.config.uncertainty_multiplier
        signal.leverage = self.config.uncertainty_leverage
        signal.reasoning += " [UNCERTAINTY - size halved]"
        signal.metadata['uncertainty_mode'] = True
        return signal

    def _should_rebalance(self, symbol: str, target_allocation: float) -> bool:
        """Check if we should rebalance based on allocation difference."""
        current = self._current_allocation.get(symbol, 0.0)
        diff = abs(target_allocation - current)
        return diff > self.config.rebalance_threshold

    def get_volatility_ranking(self) -> list[tuple[int, float, str]]:
        """Get regimes sorted by volatility with their strategy assignments."""
        results = []
        for regime_id, vol_rank in self._regime_id_to_vol_rank.items():
            strategy_name = self._vol_rank_to_strategy.get(vol_rank, "Unknown")
            exp_vol = self.regime_infos[regime_id].expected_volatility
            results.append((regime_id, exp_vol, strategy_name))
        return sorted(results, key=lambda x: x[1])


LABEL_TO_STRATEGY = {
    "CRASH": HighVolDefensiveStrategy,
    "STRONG_BEAR": HighVolDefensiveStrategy,
    "WEAK_BEAR": HighVolDefensiveStrategy,
    "BEAR": HighVolDefensiveStrategy,
    "NEUTRAL": MidVolCautiousStrategy,
    "WEAK_BULL": MidVolCautiousStrategy,
    "WEAK_BULL": MidVolCautiousStrategy,
    "STRONG_BULL": LowVolBullStrategy,
    "BULL": LowVolBullStrategy,
    "EUPHORIA": LowVolBullStrategy,
}


CrashDefensiveStrategy = HighVolDefensiveStrategy
BearTrendStrategy = HighVolDefensiveStrategy
MeanReversionStrategy = MidVolCautiousStrategy
BullTrendStrategy = LowVolBullStrategy
EuphoriaCautiousStrategy = LowVolBullStrategy
CalmBullStrategy = LowVolBullStrategy
VolatileCautiousStrategy = MidVolCautiousStrategy


def create_default_config() -> StrategyConfig:
    """Create default strategy configuration."""
    return StrategyConfig()


def create_orchestrator(
    regime_infos: dict[int, RegimeInfo],
    min_confidence: float = 0.7,
) -> StrategyOrchestrator:
    """Convenience function to create an orchestrator with default config."""
    config = StrategyConfig(min_confidence_threshold=min_confidence)
    return StrategyOrchestrator(config, regime_infos)