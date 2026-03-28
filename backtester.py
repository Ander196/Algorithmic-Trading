"""
Backtester Module - Strategy simulation with full risk management
"""
import pandas as pd
import numpy as np
from decimal import Decimal
from typing import Dict, List, Tuple
from indicators import calculateAllIndicators


class TradingBacktester:
    """Backtester with 8-confirmation voting system and risk management."""

    def __init__(
        self,
        initial_capital: float = 1000.0,
        leverage: float = 2.5,
        cooldown_hours: int = 48,
        min_confirmations: int = 7
    ):
        self.initial_capital = Decimal(str(initial_capital))
        self.leverage = Decimal(str(leverage))
        self.cooldown_hours = cooldown_hours
        self.min_confirmations = min_confirmations

    def calculateConfirmations(self, row: pd.Series) -> Tuple[int, List[str]]:
        """
        Calculate number of confirmation conditions met.

        Returns:
            Tuple of (count, list of met condition names)
        """
        confirmations = []
        met = 0

        if row['rsi'] < 90:
            met += 1
            confirmations.append('RSI < 90')

        if row['momentum'] > 1.0:
            met += 1
            confirmations.append('Momentum > 1%')

        if row['volatility'] < 6.0:
            met += 1
            confirmations.append('Volatility < 6%')

        if row['volumeAboveSma']:
            met += 1
            confirmations.append('Volume > SMA20')

        if row['adx'] > 25:
            met += 1
            confirmations.append('ADX > 25')

        if row['priceAboveEma50']:
            met += 1
            confirmations.append('Price > EMA50')

        if row['priceAboveEma200']:
            met += 1
            confirmations.append('Price > EMA200')

        if row['macdAboveSignal']:
            met += 1
            confirmations.append('MACD > Signal')

        return met, confirmations

    def runBacktest(self, df: pd.DataFrame) -> Dict:
        """
        Run full backtest on historical data.

        Returns:
            Dictionary with trades, equity curve, and metrics
        """
        df = df.copy()
        if 'rsi' not in df.columns or df['rsi'].isna().all():
            df = calculateAllIndicators(df)

        trades = []
        equity_curve = [self.initial_capital]
        timestamps = [df.index[0]]

        capital = self.initial_capital
        position = Decimal('0')
        entry_price = Decimal('0')
        entry_time = None
        cooldown_until = None

        for i in range(50, len(df)):
            row = df.iloc[i]
            current_time = df.index[i]

            if pd.isna(row['rsi']) or pd.isna(row['adx']):
                equity_curve.append(capital)
                timestamps.append(current_time)
                continue

            met, confs = self.calculateConfirmations(row)

            in_position = position > 0

            # Check cooldown expiry
            if cooldown_until is not None and current_time >= cooldown_until:
                cooldown_until = None

            # Handle cooldown period - skip to next iteration but still track equity
            if cooldown_until is not None:
                equity_curve.append(capital)
                timestamps.append(current_time)
                continue

            # Exit logic - regime flipped to Bear/Crash
            if in_position:
                if row['regime'] in ['Bear', 'Crash']:
                    exit_price = Decimal(str(row['open']))
                    raw_pnl = (exit_price - entry_price) / entry_price * capital
                    pnl = raw_pnl * self.leverage
                    capital += pnl

                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': float(entry_price),
                        'exit_price': float(exit_price),
                        'pnl': float(pnl),
                        'exit_reason': row['regime'],
                        'regime': row['regime']
                    })

                    position = Decimal('0')
                    entry_price = Decimal('0')
                    entry_time = None
                    cooldown_until = current_time + pd.Timedelta(hours=self.cooldown_hours)

                    equity_curve.append(capital)
                    timestamps.append(current_time)
                    continue

            # Entry logic
            if not in_position and met >= self.min_confirmations and row['regime'] == 'Bull':
                position = capital
                entry_price = Decimal(str(row['open']))
                entry_time = current_time

            # Track equity
            if in_position:
                current_value = capital + (Decimal(str(row['close'])) - entry_price) / entry_price * capital * self.leverage
                equity_curve.append(current_value)
            else:
                equity_curve.append(capital)
            timestamps.append(current_time)

        equity_df = pd.DataFrame({'equity': equity_curve}, index=timestamps)

        metrics = self.calculateMetrics(equity_df, trades, df)

        return {
            'trades': trades,
            'equity_curve': equity_df,
            'metrics': metrics,
            'data': df
        }

    def calculateMetrics(
        self,
        equity_df: pd.DataFrame,
        trades: List[Dict],
        data: pd.DataFrame
    ) -> Dict:
        """Calculate performance metrics."""
        final_capital = equity_df['equity'].iloc[-1]
        total_return = (float(final_capital) - float(self.initial_capital)) / float(self.initial_capital) * 100

        if len(data) > 1:
            buy_hold_return = (data['close'].iloc[-1] - data['close'].iloc[50]) / data['close'].iloc[50] * 100
        else:
            buy_hold_return = 0

        alpha = total_return - buy_hold_return

        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

        equity_series = equity_df['equity'].apply(float)
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'alpha': alpha,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'final_capital': float(final_capital),
            'total_trades': len(trades),
            'winning_trades': len(winning_trades)
        }

    def getCurrentSignal(self, df: pd.DataFrame, regime: str) -> Tuple[str, int, List[str]]:
        """
        Get current trading signal based on latest data.

        Returns:
            Tuple of (signal, confirmations_met, list of met conditions)
        """
        latest = df.iloc[-1]

        if pd.isna(latest['rsi']) or pd.isna(latest['adx']):
            return 'HOLD', 0, []

        met, confs = self.calculateConfirmations(latest)

        if regime == 'Bull' and met >= self.min_confirmations:
            return 'LONG', met, confs
        else:
            return 'CASH', met, confs
