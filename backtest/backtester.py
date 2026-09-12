"""
Walk-Forward Backtester with HMM Regime Detection

This is an ALLOCATION-BASED walk-forward backtester. It does NOT track individual
trade entries and exits. It sets a target portfolio allocation each bar based on
the detected volatility regime and rebalances when the allocation changes meaningfully.

Walk-forward windows:
- In-Sample (IS): 252 trading days (1 year) for HMM training + model selection
- Out-of-Sample (OOS): 126 trading days (6 months) for evaluation
- Step size: 126 trading days (6 months)
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from core.hmm_model import HMMVolatilityClassifier
from data.indicators import prepareFeaturesForHMM


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for walk-forward backtester."""
    initial_capital: float = 100000.0
    is_period_days: int = 252  # 1 year for training
    oos_period_days: int = 126  # 6 months for evaluation
    step_days: int = 126  # 6 month step
    min_periods_for_hmm: int = 504  # 2 years minimum for HMM
    rebalance_threshold: float = 0.10  # 10% allocation change triggers rebalance
    slippage_pct: float = 0.0005  # 0.05% slippage on rebalance
    commission: float = 0.0  # Commission per trade
    fill_delay_bars: int = 1  # Signal at bar N -> rebalance at bar N+1 open

    # Allocation mapping based on volatility rank (0 = lowest vol)
    # Maps vol_rank (0, 1, 2...) to target_allocation
    vol_rank_to_allocation: dict[int, float] = field(default_factory=lambda: {
        0: 1.00,  # Lowest volatility: 100% allocation
        1: 0.75,  # Low-mid volatility: 75% allocation
        2: 0.50,  # Mid-high volatility: 50% allocation
        3: 0.25,  # High volatility: 25% allocation
    })


@dataclass
class BacktestRecord:
    """Single bar record in backtest."""
    timestamp: pd.Timestamp
    regime_id: int
    regime_label: str
    regime_probability: float
    target_allocation: float
    current_allocation: float
    shares: int
    cash: float
    price: float
    equity: float
    daily_return: float
    is_rebalance: bool
    confidence_bucket: str


@dataclass
class BacktestResult:
    """Results from a complete backtest run."""
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    regime_history: pd.DataFrame
    records: list[BacktestRecord]
    config: BacktestConfig
    windows: list[dict] = field(default_factory=list)

    @property
    def total_return_pct(self) -> float:
        """Calculate total return percentage."""
        if len(self.equity_curve) < 2:
            return 0.0
        start = self.equity_curve['equity'].iloc[0]
        end = self.equity_curve['equity'].iloc[-1]
        return ((end - start) / start) * 100

    @property
    def final_equity(self) -> float:
        """Get final equity value."""
        return float(self.equity_curve['equity'].iloc[-1])


# =============================================================================
# Walk-Forward Engine
# =============================================================================

class WalkForwardBacktester:
    """
    Walk-forward backtester with HMM-based regime detection.

    Key characteristics:
    - Allocation-based (not trade-based)
    - Mark-to-market equity calculation
    - Rebalance only when allocation changes > threshold
    - 1-bar fill delay (signal at N, fill at N+1 open)
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> BacktestResult:
        """
        Run full walk-forward backtest on historical data.

        Args:
            data: DataFrame with OHLCV columns (indexed by date)
            symbol: Symbol being backtested

        Returns:
            BacktestResult with all records and metrics
        """
        df = data.copy()
        df = df.sort_index()

        # Validate we have enough data
        min_required = self.config.min_periods_for_hmm + self.config.oos_period_days
        if len(df) < min_required:
            raise ValueError(
                f"Insufficient data: {len(df)} bars, need at least {min_required} "
                f"({self.config.min_periods_for_hmm} for HMM + {self.config.oos_period_days} for OOS)"
            )

        # Generate walk-forward windows
        windows = self._generate_windows(df)
        if not windows:
            raise ValueError("No valid walk-forward windows generated")

        all_records: list[BacktestRecord] = []
        window_results: list[dict] = []

        # Run each window
        for i, window in enumerate(windows):
            is_data = df.loc[window['is_start']:window['is_end']]
            oos_data = df.loc[window['oos_start']:window['oos_end']]

            if len(is_data) < self.config.min_periods_for_hmm:
                continue

            # Train HMM on data that includes lookback before IS period
            # This ensures feature computation has enough historical data
            hmm_train_data = df.loc[:window['is_end']]
            hmm = HMMVolatilityClassifier(min_periods=self.config.min_periods_for_hmm)
            hmm.fit(hmm_train_data)

            # Get volatility ranking
            vol_ranking = hmm.get_volatility_sorting()
            # vol_ranking returns [(state_id, volatility), ...] - convert to dict
            vol_rank_to_info = {rank: (sid, hmm.regime_info[sid]) for rank, (sid, vol) in enumerate(vol_ranking)}

            # Get full history up to OOS end for feature computation
            # Include data before IS period for lookback
            full_history = df.loc[:window['oos_end']]

            # Walk forward through OOS
            records = self._walk_forward_window(
                oos_data, full_history, hmm, vol_rank_to_info, window, i
            )
            all_records.extend(records)

            window_results.append({
                'window_id': i,
                'is_start': window['is_start'],
                'is_end': window['is_end'],
                'oos_start': window['oos_start'],
                'oos_end': window['oos_end'],
                'n_regimes': hmm.n_regimes,
                'bic_score': hmm.bic_score,
            })

        if not all_records:
            raise ValueError("No records generated during backtest")

        # Convert to DataFrames
        return self._compile_results(all_records, window_results, symbol)

    def _generate_windows(self, df: pd.DataFrame) -> list[dict]:
        """Generate walk-forward windows."""
        windows = []
        dates = df.index.tolist()

        # Feature computation requires ~500 rows for valid output due to rolling lookbacks
        # Start from index 500 to ensure enough data for feature computation
        feature_lookback = 500
        start_idx = max(self.config.min_periods_for_hmm, feature_lookback)

        while True:
            is_start_idx = start_idx - self.config.min_periods_for_hmm
            is_end_idx = start_idx - 1

            oos_start_idx = start_idx
            oos_end_idx = start_idx + self.config.oos_period_days - 1

            if oos_end_idx >= len(dates):
                break

            windows.append({
                'is_start': dates[is_start_idx],
                'is_end': dates[is_end_idx],
                'oos_start': dates[oos_start_idx],
                'oos_end': dates[oos_end_idx],
            })

            start_idx += self.config.step_days

        return windows

    def _walk_forward_window(
        self,
        oos_data: pd.DataFrame,
        full_history: pd.DataFrame,
        hmm: HMMVolatilityClassifier,
        vol_rank_to_info: dict[int, tuple],
        window: dict,
        window_id: int,
    ) -> list[BacktestRecord]:
        """Walk through a single OOS window bar by bar."""
        records = []

        # Initialize portfolio
        cash = self.config.initial_capital
        shares = 0
        current_allocation = 0.0

        # For fill delay: buffer signals until next bar
        pending_rebalance = None

        oos_dates = oos_data.index.tolist()

        for bar_idx in range(len(oos_data)):
            current_date = oos_dates[bar_idx]
            current_bar = oos_data.iloc[bar_idx]
            current_price = float(current_bar['open'])  # Use open for fills

            # Get price for mark-to-market (use close for equity calculation)
            close_price = float(current_bar['close'])

            # Compute features using full history up to current point
            # This ensures we have enough lookback for feature computation
            lookback_data = full_history.loc[:current_date]
            if len(lookback_data) < 252:
                # Not enough data for features yet
                equity = cash + shares * close_price
                records.append(BacktestRecord(
                    timestamp=current_date,
                    regime_id=-1,
                    regime_label="WARMUP",
                    regime_probability=0.0,
                    target_allocation=current_allocation,
                    current_allocation=current_allocation,
                    shares=shares,
                    cash=float(cash),
                    price=close_price,
                    equity=equity,
                    daily_return=0.0,
                    is_rebalance=False,
                    confidence_bucket="N/A",
                ))
                continue

            # Run filtered HMM (forward algorithm)
            try:
                features = prepareFeaturesForHMM(lookback_data, min_periods=252)
                if len(features) < 1:
                    raise ValueError("No features generated")

                # Get regime using forward algorithm (filtered, no look-ahead)
                regime_states = hmm.predict_regime_filtered(features)
                current_regime_id = int(regime_states[-1])

                # Get probabilities
                proba = hmm.predict_regime_proba(features)
                regime_probability = float(proba[current_regime_id])

                # Get regime label
                regime_label = hmm._state_id_to_label.get(current_regime_id, "UNKNOWN")

            except Exception as e:
                # If HMM fails, maintain current position
                equity = cash + shares * close_price
                records.append(BacktestRecord(
                    timestamp=current_date,
                    regime_id=-1,
                    regime_label="ERROR",
                    regime_probability=0.0,
                    target_allocation=current_allocation,
                    current_allocation=current_allocation,
                    shares=shares,
                    cash=float(cash),
                    price=close_price,
                    equity=equity,
                    daily_return=0.0,
                    is_rebalance=False,
                    confidence_bucket="N/A",
                ))
                continue

            # Determine target allocation based on volatility rank
            target_allocation = self._get_target_allocation(current_regime_id, vol_rank_to_info)

            # Determine confidence bucket
            confidence_bucket = self._get_confidence_bucket(regime_probability)

            # Check if we need to rebalance
            allocation_change = abs(target_allocation - current_allocation)
            should_rebalance = allocation_change > self.config.rebalance_threshold

            # Apply pending rebalance from previous bar (fill delay)
            if pending_rebalance is not None:
                target_allocation, target_price = pending_rebalance
                should_rebalance = True
                current_price = target_price  # Execute at previous bar's close (next open)
                pending_rebalance = None

            is_rebalance = False

            if should_rebalance and bar_idx > 0:
                # Execute rebalance at next bar's open
                # Get execution price (open of next bar)
                if bar_idx < len(oos_data) - 1:
                    exec_price = float(oos_data.iloc[bar_idx + 1]['open'])
                else:
                    exec_price = close_price

                # ALLOCATION MATH (exact):
                # equity = cash + shares * current_price
                # target_shares = int(equity * target_allocation / current_price)
                # delta = target_shares - current_shares
                # cash = cash - delta * price (or cash + delta * price if selling)
                # shares = target_shares
                equity = cash + shares * exec_price
                target_shares = int(equity * target_allocation / exec_price)
                delta = target_shares - shares

                if delta != 0:
                    # Apply slippage
                    slippage = 1.0 + (self.config.slippage_pct if delta > 0 else -self.config.slippage_pct)
                    exec_price = exec_price * slippage

                    # Execute trade
                    trade_value = delta * exec_price
                    cash = cash - trade_value

                    # Apply commission
                    cash = cash - self.config.commission

                    shares = target_shares
                    current_allocation = target_allocation
                    is_rebalance = True

            # Mark-to-market equity
            equity = cash + shares * close_price

            # Calculate daily return
            if len(records) > 0:
                prev_equity = records[-1].equity
                if prev_equity > 0:
                    daily_return = (equity - prev_equity) / prev_equity
                else:
                    daily_return = 0.0
            else:
                daily_return = 0.0

            records.append(BacktestRecord(
                timestamp=current_date,
                regime_id=current_regime_id,
                regime_label=regime_label,
                regime_probability=regime_probability,
                target_allocation=target_allocation,
                current_allocation=current_allocation,
                shares=shares,
                cash=float(cash),
                price=close_price,
                equity=equity,
                daily_return=daily_return,
                is_rebalance=is_rebalance,
                confidence_bucket=confidence_bucket,
            ))

            # Queue rebalance for next bar if allocation changed
            if should_rebalance and bar_idx < len(oos_data) - 1:
                pending_rebalance = (target_allocation, close_price)

        return records

    def _get_target_allocation(
        self,
        regime_id: int,
        vol_rank_to_info: dict[int, tuple],
    ) -> float:
        """Get target allocation based on volatility rank."""
        # Find the volatility rank for this regime
        for vol_rank, (sid, info) in vol_rank_to_info.items():
            if sid == regime_id:
                return self.config.vol_rank_to_allocation.get(
                    vol_rank,
                    self.config.vol_rank_to_allocation.get(max(self.config.vol_rank_to_allocation.keys()), 0.25)
                )

        # Default to lowest allocation if regime not found
        return 0.25

    def _get_confidence_bucket(self, probability: float) -> str:
        """Bucket regime probability for analysis."""
        if probability < 0.50:
            return "<50%"
        elif probability < 0.60:
            return "50-60%"
        elif probability < 0.70:
            return "60-70%"
        else:
            return "70+%"
        return "N/A"

    def _compile_results(
        self,
        records: list[BacktestRecord],
        window_results: list[dict],
        symbol: str,
    ) -> BacktestResult:
        """Convert records to DataFrames and create result."""
        # Equity curve
        equity_df = pd.DataFrame({
            'timestamp': [r.timestamp for r in records],
            'equity': [r.equity for r in records],
        })
        equity_df.set_index('timestamp', inplace=True)

        # Trade log (rebalance events)
        trades = []
        for r in records:
            if r.is_rebalance:
                trades.append({
                    'timestamp': r.timestamp,
                    'regime_id': r.regime_id,
                    'regime_label': r.regime_label,
                    'regime_probability': r.regime_probability,
                    'target_allocation': r.target_allocation,
                    'shares': r.shares,
                    'cash': r.cash,
                    'price': r.price,
                    'equity': r.equity,
                    'confidence_bucket': r.confidence_bucket,
                })

        trade_df = pd.DataFrame(trades)
        if len(trade_df) > 0:
            trade_df.set_index('timestamp', inplace=True)

        # Regime history
        regime_df = pd.DataFrame({
            'timestamp': [r.timestamp for r in records],
            'regime_id': [r.regime_id for r in records],
            'regime_label': [r.regime_label for r in records],
            'regime_probability': [r.regime_probability for r in records],
            'target_allocation': [r.target_allocation for r in records],
            'current_allocation': [r.current_allocation for r in records],
            'equity': [r.equity for r in records],
        })
        regime_df.set_index('timestamp', inplace=True)

        return BacktestResult(
            equity_curve=equity_df,
            trade_log=trade_df,
            regime_history=regime_df,
            records=records,
            config=self.config,
            windows=window_results,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def run_walk_forward_backtest(
    data: pd.DataFrame,
    initial_capital: float = 100000.0,
    is_period_days: int = 252,
    oos_period_days: int = 126,
    step_days: int = 126,
    rebalance_threshold: float = 0.10,
    slippage_pct: float = 0.0005,
    commission: float = 0.0,
) -> BacktestResult:
    """
    Convenience function to run walk-forward backtest.

    Args:
        data: DataFrame with OHLCV columns (indexed by date)
        initial_capital: Starting capital
        is_period_days: In-sample period for HMM training
        oos_period_days: Out-of-sample period for evaluation
        step_days: Step size between windows
        rebalance_threshold: Allocation change threshold to trigger rebalance
        slippage_pct: Slippage percentage on each rebalance
        commission: Commission per trade

    Returns:
        BacktestResult with all records and metrics
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        is_period_days=is_period_days,
        oos_period_days=oos_period_days,
        step_days=step_days,
        rebalance_threshold=rebalance_threshold,
        slippage_pct=slippage_pct,
        commission=commission,
    )

    backtester = WalkForwardBacktester(config)
    return backtester.run(data)