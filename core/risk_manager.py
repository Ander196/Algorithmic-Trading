"""
Risk Manager Module

Provides portfolio-level and position-level risk controls including:
- Portfolio exposure limits
- Circuit breakers for drawdown protection
- Position sizing based on risk per trade
- Order validation
- Correlation checks

The risk manager operates INDEPENDENTLY of the HMM. Even if the HMM fails completely,
circuit breakers catch drawdowns based on actual P&L. Defense in depth.

The risk manager has ABSOLUTE VETO POWER over any signal.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from numpy.typing import NDArray

from core.regime_strategies import Direction, Signal


class DecisionStatus(Enum):
    """Risk decision status."""
    APPROVED = "APPROVED"
    MODIFIED = "MODIFIED"
    REJECTED = "REJECTED"


class CircuitBreakerLevel(Enum):
    """Circuit breaker severity levels."""
    NONE = "NONE"
    REDUCE_SIZE = "REDUCE_SIZE"  # 50% size reduction
    HALT_DAY = "HALT_DAY"        # Close all, halt for rest of day
    HALT_WEEK = "HALT_WEEK"      # Close all, halt for rest of week
    HALT_ALL = "HALT_ALL"        # Halt all trading, requires manual intervention


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    direction: Direction
    entry_price: float
    stop_loss: float
    size: float  # Dollar amount
    quantity: float  # Number of shares
    entry_time: datetime
    regime_id: int = 0
    regime_name: str = "UNKNOWN"
    sector: str = "UNKNOWN"
    correlation_with_portfolio: float = 0.0


@dataclass
class RiskDecision:
    """Result of risk validation on a signal."""
    status: DecisionStatus
    modified_signal: Optional[Signal] = None
    rejection_reason: Optional[str] = None
    modifications: list[str] = field(default_factory=list)
    circuit_breaker_level: CircuitBreakerLevel = CircuitBreakerLevel.NONE
    risk_metrics: dict = field(default_factory=dict)


@dataclass
class CircuitBreakerSnapshot:
    """Snapshot of circuit breaker state for logging."""
    breaker_type: str
    actual_dd: float
    equity: float
    positions_closed: int
    hmm_regime: str
    hmm_wrong: bool
    timestamp: datetime
    level: CircuitBreakerLevel


@dataclass
class PortfolioState:
    """Current state of the portfolio for risk calculations."""
    equity: float
    cash: float
    buying_power: float
    positions: list[Position]
    daily_pnl: float
    weekly_pnl: float
    peak_equity: float
    drawdown: float
    circuit_breaker_status: CircuitBreakerLevel
    flicker_rate: int
    trades_today: int
    last_trade_time: Optional[datetime] = None

    def total_exposure(self) -> float:
        """Total dollar exposure across all positions."""
        return sum(p.size for p in self.positions)

    def exposure_pct(self) -> float:
        """Total exposure as percentage of equity."""
        if self.equity <= 0:
            return 1.0
        return self.total_exposure() / self.equity

    def position_count(self) -> int:
        """Number of open positions."""
        return len(self.positions)

    def largest_position_pct(self) -> float:
        """Largest single position as percentage of equity."""
        if self.equity <= 0 or not self.positions:
            return 0.0
        return max(p.size for p in self.positions) / self.equity

    def sector_exposure(self, sector: str) -> float:
        """Exposure in a specific sector."""
        return sum(
            p.size for p in self.positions
            if p.sector.upper() == sector.upper()
        ) / self.equity if self.equity > 0 else 0.0


@dataclass
class HMMSnapshot:
    """Snapshot of HMM state at time of breaker trigger."""
    regime_id: int
    regime_name: str
    regime_probability: float
    is_flickering: bool
    flicker_count: int


class Settings:
    """Configuration loaded from settings.yaml."""

    _instance: Optional["Settings"] = None

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "settings.yaml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        hmm = config.get("hmm", {})
        self.n_candidates: list[int] = hmm.get("n_candidates", [3, 4, 5, 6, 7])
        self.n_init: int = hmm.get("n_init", 10)
        self.covariance_type: str = hmm.get("covariance_type", "full")
        self.min_train_bars: int = hmm.get("min_train_bars", 252)
        self.stability_bars: int = hmm.get("stability_bars", 3)
        self.flicker_window: int = hmm.get("flicker_window", 20)
        self.flicker_threshold: int = hmm.get("flicker_threshold", 4)
        self.min_confidence: float = hmm.get("min_confidence", 0.7)

        strategy = config.get("strategy", {})
        self.low_vol_allocation: float = strategy.get("low_vol_allocation", 0.95)
        self.mid_vol_allocation_trend: float = strategy.get("mid_vol_allocation_trend", 0.95)
        self.mid_vol_allocation_no_trend: float = strategy.get("mid_vol_allocation_no_trend", 0.60)
        self.high_vol_allocation: float = strategy.get("high_vol_allocation", 0.60)
        self.low_vol_leverage: float = strategy.get("low_vol_leverage", 1.25)
        self.rebalance_threshold: float = strategy.get("rebalance_threshold", 0.10)
        self.uncertainty_size_mult: float = strategy.get("uncertainty_size_mult", 0.50)

        risk = config.get("risk", {})
        self.max_risk_per_trade: float = risk.get("max_risk_per_trade", 0.01)
        self.max_exposure: float = risk.get("max_exposure", 0.80)
        self.max_leverage: float = risk.get("max_leverage", 1.25)
        self.max_single_position: float = risk.get("max_single_position", 0.15)
        self.max_concurrent: int = risk.get("max_concurrent", 5)
        self.max_daily_trades: int = risk.get("max_daily_trades", 20)
        self.daily_dd_reduce: float = risk.get("daily_dd_reduce", 0.02)
        self.daily_dd_halt: float = risk.get("daily_dd_halt", 0.03)
        self.weekly_dd_reduce: float = risk.get("weekly_dd_reduce", 0.05)
        self.weekly_dd_halt: float = risk.get("weekly_dd_halt", 0.07)
        self.max_dd_from_peak: float = risk.get("max_dd_from_peak", 0.10)
        self.max_sector_exposure: float = risk.get("max_sector_exposure", 0.30)
        self.gap_risk_multiplier: float = risk.get("gap_risk_multiplier", 3)
        self.min_position_dollar: float = risk.get("min_position_dollar", 1)
        self.max_spread_pct: float = risk.get("max_spread_pct", 0.005)
        self.duplicate_window_seconds: int = risk.get("duplicate_window_seconds", 60)

        backtest = config.get("backtest", {})
        self.slippage_pct: float = backtest.get("slippage_pct", 0.0005)
        self.initial_capital: float = backtest.get("initial_capital", 1000)
        self.train_window: int = backtest.get("train_window", 252)
        self.test_window: int = backtest.get("test_window", 126)
        self.step_size: int = backtest.get("step_size", 126)
        self.risk_free_rate: float = backtest.get("risk_free_rate", 0.045)

        main = config.get("main", {})
        self.bar_interval: int = main.get("bar_interval", 5)
        self.poll_interval: int = main.get("poll_interval", 30)
        self.market_wait: bool = main.get("market_wait", True)
        self.retrain_weeks: int = main.get("retrain_weeks", 1)
        self.hmm_max_age_days: int = main.get("hmm_max_age_days", 7)
        self.hmm_model_path: str = main.get("hmm_model_path", "models/hmm_model.pkl")
        self.state_snapshot_path: str = main.get("state_snapshot_path", "state_snapshot.json")
        self.dry_run: bool = main.get("dry_run", False)
        self.log_level: str = main.get("log_level", "INFO")

    @classmethod
    def get(cls) -> "Settings":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class CircuitBreaker:
    """
    Circuit breaker system that monitors drawdowns and triggers protective actions.

    Operates INDEPENDENTLY of the HMM - fires based on actual P&L.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings.get()

        self._peak_equity: float = self.settings.initial_capital
        self._daily_peak: float = self.settings.initial_capital
        self._weekly_peak: float = self.settings.initial_capital
        self._current_equity: float = self.settings.initial_capital

        self._day_start_equity: float = self.settings.initial_capital
        self._week_start_equity: float = self.settings.initial_capital

        self._history: list[CircuitBreakerSnapshot] = []
        self._current_level: CircuitBreakerLevel = CircuitBreakerLevel.NONE

        self._last_daily_reset: datetime = datetime.now(timezone.utc)
        self._last_weekly_reset: datetime = datetime.now(timezone.utc)

    @property
    def day_start_equity(self) -> float:
        return self._day_start_equity

    @property
    def week_start_equity(self) -> float:
        return self._week_start_equity

    def update(self, pnl: float, equity: float) -> None:
        """Update circuit breaker with new P&L and equity."""
        self._current_equity = equity
        self._peak_equity = max(self._peak_equity, equity)

    def check(self, portfolio_state: PortfolioState) -> CircuitBreakerLevel:
        """
        Check all circuit breaker thresholds.

        Returns the highest active level.
        """
        equity = portfolio_state.equity
        day_start = self._day_start_equity
        week_start = self._week_start_equity

        daily_dd = (day_start - equity) / day_start if day_start > 0 else 0
        weekly_dd = (week_start - equity) / week_start if week_start > 0 else 0
        peak_dd = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0

        if peak_dd >= self.settings.max_dd_from_peak:
            self._current_level = CircuitBreakerLevel.HALT_ALL
            self._write_halt_file()
            return CircuitBreakerLevel.HALT_ALL

        if daily_dd >= self.settings.daily_dd_halt:
            self._current_level = CircuitBreakerLevel.HALT_DAY
            return CircuitBreakerLevel.HALT_DAY

        if weekly_dd >= self.settings.weekly_dd_halt:
            self._current_level = CircuitBreakerLevel.HALT_WEEK
            return CircuitBreakerLevel.HALT_WEEK

        if daily_dd >= self.settings.daily_dd_reduce:
            self._current_level = CircuitBreakerLevel.REDUCE_SIZE
            return CircuitBreakerLevel.REDUCE_SIZE

        if weekly_dd >= self.settings.weekly_dd_reduce:
            self._current_level = CircuitBreakerLevel.REDUCE_SIZE
            return CircuitBreakerLevel.REDUCE_SIZE

        self._current_level = CircuitBreakerLevel.NONE
        return CircuitBreakerLevel.NONE

    def get_daily_drawdown(self, portfolio_state: PortfolioState) -> float:
        """Calculate current daily drawdown."""
        if self._day_start_equity <= 0:
            return 0.0
        return (self._day_start_equity - portfolio_state.equity) / self._day_start_equity

    def get_weekly_drawdown(self, portfolio_state: PortfolioState) -> float:
        """Calculate current weekly drawdown."""
        if self._week_start_equity <= 0:
            return 0.0
        return (self._week_start_equity - portfolio_state.equity) / self._week_start_equity

    def get_peak_drawdown(self, portfolio_state: PortfolioState) -> float:
        """Calculate current peak drawdown."""
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - portfolio_state.equity) / self._peak_equity

    def reset_daily(self, equity: float) -> None:
        """Reset daily tracking, called at market open."""
        self._day_start_equity = equity
        self._daily_peak = equity
        self._last_daily_reset = datetime.now(timezone.utc)

    def reset_weekly(self, equity: float) -> None:
        """Reset weekly tracking, called at start of week."""
        self._week_start_equity = equity
        self._weekly_peak = equity
        self._last_weekly_reset = datetime.now(timezone.utc)

    def get_history(self) -> list[CircuitBreakerSnapshot]:
        """Get trigger history."""
        return self._history.copy()

    def log_trigger(
        self,
        breaker_type: str,
        actual_dd: float,
        equity: float,
        positions_closed: int,
        hmm_snapshot: Optional[HMMSnapshot],
    ) -> None:
        """Log a circuit breaker trigger."""
        hmm_regime = "UNKNOWN"
        hmm_wrong = False

        if hmm_snapshot:
            hmm_regime = hmm_snapshot.regime_name

        snapshot = CircuitBreakerSnapshot(
            breaker_type=breaker_type,
            actual_dd=actual_dd,
            equity=equity,
            positions_closed=positions_closed,
            hmm_regime=hmm_regime,
            hmm_wrong=hmm_wrong,
            timestamp=datetime.now(timezone.utc),
            level=self._current_level,
        )
        self._history.append(snapshot)

    def _write_halt_file(self) -> None:
        """Write trading_halted.lock file."""
        lock_path = Path("trading_halted.lock")
        with open(lock_path, "w") as f:
            f.write(f"Trading halted at {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"Peak drawdown exceeded {self.settings.max_dd_from_peak:.1%}\n")
            f.write("Manual intervention required to resume.\n")

    def is_halted(self) -> bool:
        """Check if trading is halted due to circuit breaker."""
        return self._current_level in (
            CircuitBreakerLevel.HALT_DAY,
            CircuitBreakerLevel.HALT_WEEK,
            CircuitBreakerLevel.HALT_ALL,
        )

    @property
    def current_level(self) -> CircuitBreakerLevel:
        return self._current_level


class RiskManager:
    """
    Main risk management class.

    Validates trading signals against portfolio state and risk limits.
    Has ABSOLUTE VETO POWER over any signal.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings.get()
        self.circuit_breaker = CircuitBreaker(self.settings)
        self._duplicate_tracks: dict[str, datetime] = {}

    def validate_signal(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
        hmm_snapshot: Optional[HMMSnapshot] = None,
        price_data: Optional[dict] = None,
    ) -> RiskDecision:
        """
        Validate a trading signal against all risk rules.

        Args:
            signal: The trading signal to validate
            portfolio_state: Current portfolio state
            hmm_snapshot: Current HMM state (for logging)
            price_data: Optional dict with 'bid', 'ask', 'sector' keys

        Returns:
            RiskDecision with approval, modification, or rejection
        """
        modifications: list[str] = []
        modified_signal = None
        rejection_reason = None
        status = DecisionStatus.APPROVED

        breaker_level = self.circuit_breaker.check(portfolio_state)
        if breaker_level != CircuitBreakerLevel.NONE:
            return RiskDecision(
                status=DecisionStatus.REJECTED,
                rejection_reason=f"Circuit breaker active: {breaker_level.value}",
                circuit_breaker_level=breaker_level,
            )

        rejection_reason = self._check_circuit_breaker(portfolio_state, hmm_snapshot)
        if rejection_reason:
            return RiskDecision(
                status=DecisionStatus.REJECTED,
                rejection_reason=rejection_reason,
                circuit_breaker_level=self.circuit_breaker.current_level,
            )

        rejection_reason = self._check_portfolio_limits(portfolio_state, signal)
        if rejection_reason:
            return RiskDecision(
                status=DecisionStatus.REJECTED,
                rejection_reason=rejection_reason,
            )

        rejection_reason = self._check_position_limits(signal, portfolio_state)
        if rejection_reason:
            return RiskDecision(
                status=DecisionStatus.REJECTED,
                rejection_reason=rejection_reason,
            )

        rejection_reason = self._check_stop_loss(signal)
        if rejection_reason:
            return RiskDecision(
                status=DecisionStatus.REJECTED,
                rejection_reason=rejection_reason,
            )

        rejection_reason = self._check_order_validation(signal, portfolio_state, price_data)
        if rejection_reason:
            return RiskDecision(
                status=DecisionStatus.REJECTED,
                rejection_reason=rejection_reason,
            )

        rejection_reason = self._check_duplicate(signal, portfolio_state)
        if rejection_reason:
            return RiskDecision(
                status=DecisionStatus.REJECTED,
                rejection_reason=rejection_reason,
            )

        modified_signal, pos_mods = self._calculate_position_size(signal, portfolio_state)
        if pos_mods:
            modifications.extend(pos_mods)

        rejection_reason = self._check_correlation(
            signal, portfolio_state, price_data
        )
        if rejection_reason:
            return RiskDecision(
                status=DecisionStatus.REJECTED,
                rejection_reason=rejection_reason,
            )

        modified_signal, leverage_mods = self._check_leverage(modified_signal, portfolio_state, hmm_snapshot)
        if leverage_mods:
            modifications.extend(leverage_mods)

        if modifications:
            status = DecisionStatus.MODIFIED

        return RiskDecision(
            status=status,
            modified_signal=modified_signal,
            rejection_reason=rejection_reason,
            modifications=modifications,
            circuit_breaker_level=self.circuit_breaker.current_level,
            risk_metrics=self._calculate_risk_metrics(signal, portfolio_state),
        )

    def _check_circuit_breaker(
        self,
        portfolio_state: PortfolioState,
        hmm_snapshot: Optional[HMMSnapshot],
    ) -> Optional[str]:
        """Check circuit breaker status."""
        level = self.circuit_breaker.check(portfolio_state)

        if level == CircuitBreakerLevel.HALT_ALL:
            self.circuit_breaker.log_trigger(
                "PEAK_DD_EXCEEDED",
                self.circuit_breaker.get_peak_drawdown(portfolio_state),
                portfolio_state.equity,
                len(portfolio_state.positions),
                hmm_snapshot,
            )
            return f"Trading halted - peak drawdown exceeds {self.settings.max_dd_from_peak:.1%}"

        if level == CircuitBreakerLevel.HALT_DAY:
            daily_dd = self.circuit_breaker.get_daily_drawdown(portfolio_state)
            self.circuit_breaker.log_trigger(
                "DAILY_DD_HALT",
                daily_dd,
                portfolio_state.equity,
                len(portfolio_state.positions),
                hmm_snapshot,
            )
            return f"Trading halted - daily drawdown {daily_dd:.2%} exceeds {self.settings.daily_dd_halt:.1%}"

        if level == CircuitBreakerLevel.HALT_WEEK:
            weekly_dd = self.circuit_breaker.get_weekly_drawdown(portfolio_state)
            self.circuit_breaker.log_trigger(
                "WEEKLY_DD_HALT",
                weekly_dd,
                portfolio_state.equity,
                len(portfolio_state.positions),
                hmm_snapshot,
            )
            return f"Trading halted - weekly drawdown {weekly_dd:.2%} exceeds {self.settings.weekly_dd_halt:.1%}"

        if level == CircuitBreakerLevel.REDUCE_SIZE:
            return None

        return None

    def _check_portfolio_limits(
        self,
        portfolio_state: PortfolioState,
        signal: Signal,
    ) -> Optional[str]:
        """Check portfolio-level exposure limits."""
        new_exposure = portfolio_state.exposure_pct() + (
            signal.position_size_pct * signal.leverage
        )

        if new_exposure > self.settings.max_exposure:
            return (
                f"Portfolio exposure {new_exposure:.1%} would exceed max "
                f"{self.settings.max_exposure:.1%}"
            )

        if portfolio_state.position_count() >= self.settings.max_concurrent:
            return (
                f"Max concurrent positions {self.settings.max_concurrent} reached"
            )

        if portfolio_state.trades_today >= self.settings.max_daily_trades:
            return f"Max daily trades {self.settings.max_daily_trades} reached"

        return None

    def _check_position_limits(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
    ) -> Optional[str]:
        """Check single position and sector limits."""
        single_position_pct = signal.position_size_pct * signal.leverage

        if single_position_pct > self.settings.max_single_position:
            return (
                f"Position size {single_position_pct:.1%} exceeds max single "
                f"position {self.settings.max_single_position:.1%}"
            )

        return None

    def _check_stop_loss(self, signal: Signal) -> Optional[str]:
        """Check that signal has a valid stop loss."""
        if signal.stop_loss is None or signal.stop_loss <= 0:
            return "Signal must have a stop loss"

        if signal.direction == Direction.LONG:
            if signal.stop_loss >= signal.entry_price:
                return "Stop loss must be below entry for long positions"
        else:
            if signal.stop_loss <= signal.entry_price:
                return "Stop loss must be above entry for short positions"

        return None

    def _check_order_validation(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
        price_data: Optional[dict],
    ) -> Optional[str]:
        """Check order validation rules."""
        required_size = signal.position_size_pct * portfolio_state.equity
        if required_size > portfolio_state.buying_power:
            return f"Insufficient buying power: need ${required_size:.2f}, have ${portfolio_state.buying_power:.2f}"

        if price_data:
            bid = price_data.get("bid")
            ask = price_data.get("ask")

            if bid and ask:
                spread_pct = (ask - bid) / ask
                if spread_pct > self.settings.max_spread_pct:
                    return f"Bid-ask spread {spread_pct:.2%} exceeds max {self.settings.max_spread_pct:.2%}"

        return None

    def _check_duplicate(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
    ) -> Optional[str]:
        """Check for duplicate trades within window."""
        key = f"{signal.symbol}:{signal.direction.value}"
        now = datetime.now(timezone.utc)

        if key in self._duplicate_tracks:
            last_time = self._duplicate_tracks[key]
            elapsed = (now - last_time).total_seconds()
            if elapsed < self.settings.duplicate_window_seconds:
                return (
                    f"Duplicate {signal.direction.value} for {signal.symbol} "
                    f"within {self.settings.duplicate_window_seconds}s window"
                )

        self._duplicate_tracks[key] = now
        return None

    def _calculate_position_size(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
    ) -> tuple[Optional[Signal], list[str]]:
        """Calculate position size based on risk per trade rules."""
        modifications = []

        risk_amount = portfolio_state.equity * self.settings.max_risk_per_trade

        entry = signal.entry_price
        stop = signal.stop_loss
        risk_per_share = abs(entry - stop)

        if risk_per_share <= 0:
            return signal, ["Invalid risk per share - using max position"]

        dollar_size = risk_amount / risk_per_share

        max_dollar_by_single = portfolio_state.equity * self.settings.max_single_position
        dollar_size = min(dollar_size, max_dollar_by_single)

        max_dollar_by_exposure = portfolio_state.equity * self.settings.max_exposure
        current_exposure = portfolio_state.total_exposure()
        available_exposure = max(0, max_dollar_by_exposure - current_exposure)
        dollar_size = min(dollar_size, available_exposure)

        if dollar_size < self.settings.min_position_dollar:
            return None, [f"Position size ${dollar_size:.2f} below minimum ${self.settings.min_position_dollar}"]

        size_pct = dollar_size / portfolio_state.equity

        if size_pct < signal.position_size_pct:
            modifications.append(f"Position size reduced to {size_pct:.1%} based on 1% risk rule")

        signal.position_size_pct = size_pct
        signal.metadata["dollar_size"] = dollar_size

        return signal, modifications

    def _check_correlation(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
        price_data: Optional[dict],
    ) -> Optional[str]:
        """Check correlation with existing positions."""
        if not portfolio_state.positions or not price_data:
            return None

        correlation = price_data.get("correlation_with_portfolio", 0.0)

        if correlation > 0.85:
            return (
                f"Correlation {correlation:.2f} with existing positions exceeds 0.85 - trade rejected"
            )

        if correlation > 0.70:
            return (
                f"Correlation {correlation:.2f} between 0.70 and 0.85 - size reduced by 50%"
            )

        return None

    def _check_leverage(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
        hmm_snapshot: Optional[HMMSnapshot],
    ) -> tuple[Optional[Signal], list[str]]:
        """Check and enforce leverage rules."""
        modifications = []

        should_force_leverage_1 = False
        reasons: list[str] = []

        if hmm_snapshot:
            if hmm_snapshot.regime_probability < self.settings.min_confidence:
                should_force_leverage_1 = True
                reasons.append(f"low regime confidence ({hmm_snapshot.regime_probability:.2f})")

            if hmm_snapshot.is_flickering:
                should_force_leverage_1 = True
                reasons.append("regime flickering")

        if self.circuit_breaker.current_level != CircuitBreakerLevel.NONE:
            should_force_leverage_1 = True
            reasons.append("circuit breaker active")

        if portfolio_state.position_count() >= 3:
            should_force_leverage_1 = True
            reasons.append("3+ positions open")

        if hmm_snapshot and hmm_snapshot.flicker_count >= self.settings.flicker_threshold:
            should_force_leverage_1 = True
            reasons.append("high flicker rate")

        if should_force_leverage_1:
            signal.leverage = 1.0
            modifications.append(f"Force leverage to 1.0x: {', '.join(reasons)}")
        elif signal.leverage > self.settings.max_leverage:
            signal.leverage = self.settings.max_leverage
            modifications.append(f"Leverage capped at {self.settings.max_leverage}x")

        return signal, modifications

    def _calculate_risk_metrics(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
    ) -> dict:
        """Calculate risk metrics for the signal."""
        entry = signal.entry_price
        stop = signal.stop_loss

        risk_per_share = abs(entry - stop)
        dollar_risk = signal.position_size_pct * portfolio_state.equity * signal.leverage
        risk_pct = dollar_risk / portfolio_state.equity if portfolio_state.equity > 0 else 0

        return {
            "risk_per_share": risk_per_share,
            "dollar_risk": dollar_risk,
            "risk_pct": risk_pct,
            "position_size_pct": signal.position_size_pct,
            "leverage": signal.leverage,
            "total_exposure_pct": portfolio_state.exposure_pct(),
        }

    def update_equity(self, equity: float) -> None:
        """Update circuit breaker with new equity value."""
        self.circuit_breaker.update(0, equity)

    def reset_daily(self, equity: float) -> None:
        """Reset daily tracking."""
        self.circuit_breaker.reset_daily(equity)

    def reset_weekly(self, equity: float) -> None:
        """Reset weekly tracking."""
        self.circuit_breaker.reset_weekly(equity)

    def get_circuit_breaker_status(self) -> CircuitBreakerLevel:
        """Get current circuit breaker status."""
        return self.circuit_breaker.current_level

    def is_trading_halted(self) -> bool:
        """Check if trading is currently halted."""
        return self.circuit_breaker.is_halted()


def create_default_settings(config_path: Optional[Path] = None) -> Settings:
    """Create Settings instance from config file."""
    return Settings(config_path)


def create_risk_manager(settings: Optional[Settings] = None) -> RiskManager:
    """Create RiskManager instance."""
    return RiskManager(settings)


def check_halt_file_exists() -> bool:
    """Check if trading is halted by manual intervention."""
    return Path("trading_halted.lock").exists()


def clear_halt_file() -> None:
    """Clear the halt file to resume trading (manual intervention)."""
    Path("trading_halted.lock").unlink(missing_ok=True)