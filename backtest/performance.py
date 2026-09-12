"""
Performance Metrics Module

Computes comprehensive performance metrics for walk-forward backtests:
- Core metrics: Total return, CAGR, Sharpe, Sortino, Calmar, Max DD
- Regime-specific metrics
- Confidence-bucketed metrics
- Benchmark comparisons
- Worst-case metrics
- CSV output
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

    def tabulate(table_data, headers=None, tablefmt="grid"):
        """Simple fallback tabulate function."""
        if not table_data:
            return ""
        lines = []
        if headers:
            lines.append(" | ".join(headers))
            lines.append("-" * len(lines[0]))
        for row in table_data:
            lines.append(" | ".join(str(x) for x in row))
        return "\n".join(lines)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Core metrics
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    max_drawdown_days: int

    # Trade metrics
    total_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    avg_holding_period_days: float

    # Worst case
    worst_day_pct: float
    worst_week_pct: float
    worst_month_pct: float
    max_consecutive_losses: int
    longest_underwater_days: int

    # Additional
    volatility_annualized_pct: float
    skewness: float
    kurtosis: float


@dataclass
class RegimeMetrics:
    """Metrics broken down by regime."""
    regime_label: str
    time_in_pct: float
    return_contribution_pct: float
    avg_trade_pnl_pct: float
    win_rate_pct: float
    sharpe_ratio: float


@dataclass
class ConfidenceMetrics:
    """Metrics bucketed by confidence level."""
    confidence_bucket: str
    num_trades: int
    sharpe_ratio: float
    win_rate_pct: float
    avg_pnl_pct: float


@dataclass
class BenchmarkMetrics:
    """Benchmark comparison metrics."""
    name: str
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float


@dataclass
class FullPerformanceReport:
    """Complete performance report."""
    core: PerformanceMetrics
    regime_metrics: list[RegimeMetrics]
    confidence_metrics: list[ConfidenceMetrics]
    benchmarks: list[BenchmarkMetrics]
    worst_case: dict


# =============================================================================
# Core Metric Calculations
# =============================================================================

def calculate_core_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """
    Calculate core performance metrics.

    Args:
        equity_curve: DataFrame with 'equity' column indexed by timestamp
        trades: DataFrame with trade records
        risk_free_rate: Annual risk-free rate (default 0.0)

    Returns:
        PerformanceMetrics object
    """
    equity = equity_curve['equity'].values

    # Total return
    total_return_pct = ((equity[-1] - equity[0]) / equity[0]) * 100

    # Calculate returns
    returns = np.diff(equity) / equity[:-1]
    returns = returns[~np.isnan(returns)]

    # Time calculations
    if len(equity_curve) > 1:
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
    else:
        years = 1.0

    # CAGR
    if years > 0 and equity[0] > 0:
        cagr_pct = (((equity[-1] / equity[0]) ** (1 / years)) - 1) * 100
    else:
        cagr_pct = 0.0

    # Annualized volatility
    if len(returns) > 0:
        daily_vol = np.std(returns)
        volatility_annualized_pct = daily_vol * np.sqrt(252) * 100
    else:
        volatility_annualized_pct = 0.0

    # Sharpe ratio
    if volatility_annualized_pct > 0:
        excess_return = (cagr_pct / 100) - risk_free_rate
        sharpe_ratio = excess_return / (volatility_annualized_pct / 100)
    else:
        sharpe_ratio = 0.0

    # Sortino ratio (using downside deviation)
    if len(returns) > 0:
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_dev = np.std(negative_returns)
            downside_annualized = downside_dev * np.sqrt(252)
            if downside_annualized > 0:
                sortino_ratio = (cagr_pct / 100) / downside_annualized
            else:
                sortino_ratio = 0.0
        else:
            sortino_ratio = 0.0
    else:
        sortino_ratio = 0.0

    # Max drawdown
    rolling_max = np.maximum.accumulate(equity)
    drawdowns = (equity - rolling_max) / rolling_max * 100
    max_drawdown_pct = np.min(drawdowns)

    # Max drawdown duration
    underwater = equity < rolling_max
    underwater_streak = 0
    max_underwater = 0
    in_underwater = False

    for uw in underwater:
        if uw:
            if in_underwater:
                underwater_streak += 1
            else:
                underwater_streak = 1
                in_underwater = True
        else:
            in_underwater = False

        max_underwater = max(max_underwater, underwater_streak)

    max_drawdown_days = max_underwater

    # Calmar ratio
    if max_drawdown_pct > 0:
        calmar_ratio = cagr_pct / abs(max_drawdown_pct)
    else:
        calmar_ratio = 0.0

    # Trade metrics
    if len(trades) > 0:
        total_trades = len(trades)

        # Calculate P&L for each trade
        trade_pnls = []
        for i in range(1, len(trades)):
            prev_equity = trades.iloc[i-1]['equity']
            curr_equity = trades.iloc[i]['equity']
            if prev_equity > 0:
                pnl_pct = (curr_equity - prev_equity) / prev_equity * 100
                trade_pnls.append(pnl_pct)

        if trade_pnls:
            winning_trades = [p for p in trade_pnls if p > 0]
            losing_trades = [p for p in trade_pnls if p < 0]

            win_rate_pct = len(winning_trades) / len(trade_pnls) * 100 if trade_pnls else 0
            avg_win_pct = np.mean(winning_trades) if winning_trades else 0
            avg_loss_pct = np.mean(losing_trades) if losing_trades else 0

            total_wins = sum(winning_trades) if winning_trades else 0
            total_losses = abs(sum(losing_trades)) if losing_trades else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        else:
            win_rate_pct = 0
            avg_win_pct = 0
            avg_loss_pct = 0
            profit_factor = 0
            total_trades = 0

        # Average holding period
        if len(trades) > 1:
            avg_holding_period_days = 1.0  # Simplified - rebalances happen at bars
        else:
            avg_holding_period_days = 0
    else:
        total_trades = 0
        win_rate_pct = 0
        avg_win_pct = 0
        avg_loss_pct = 0
        profit_factor = 0
        avg_holding_period_days = 0

    # Worst case metrics
    if len(returns) > 0:
        worst_day_pct = np.min(returns) * 100

        # Worst week (5 trading days)
        if len(returns) >= 5:
            weekly_returns = []
            for i in range(0, len(returns) - 4, 5):
                week_ret = (equity[i + 5] - equity[i]) / equity[i] * 100 if equity[i] > 0 else 0
                weekly_returns.append(week_ret)
            worst_week_pct = min(weekly_returns) if weekly_returns else 0
        else:
            worst_week_pct = 0

        # Worst month (21 trading days)
        if len(returns) >= 21:
            monthly_returns = []
            for i in range(0, len(returns) - 20, 21):
                month_ret = (equity[i + 21] - equity[i]) / equity[i] * 100 if equity[i] > 0 else 0
                monthly_returns.append(month_ret)
            worst_month_pct = min(monthly_returns) if monthly_returns else 0
        else:
            worst_month_pct = 0

        # Max consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for r in returns:
            if r < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        # Longest time underwater
        longest_underwater_days = max_underwater
    else:
        worst_day_pct = 0
        worst_week_pct = 0
        worst_month_pct = 0
        max_consecutive_losses = 0
        longest_underwater_days = 0

    # Higher moments
    if len(returns) > 0:
        skewness = float(pd.Series(returns).skew())
        kurtosis = float(pd.Series(returns).kurtosis())
    else:
        skewness = 0
        kurtosis = 0

    return PerformanceMetrics(
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_days=max_drawdown_days,
        total_trades=total_trades,
        win_rate_pct=win_rate_pct,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
        profit_factor=profit_factor,
        avg_holding_period_days=avg_holding_period_days,
        worst_day_pct=worst_day_pct,
        worst_week_pct=worst_week_pct,
        worst_month_pct=worst_month_pct,
        max_consecutive_losses=max_consecutive_losses,
        longest_underwater_days=longest_underwater_days,
        volatility_annualized_pct=volatility_annualized_pct,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def calculate_regime_metrics(
    equity_curve: pd.DataFrame,
    regime_history: pd.DataFrame,
) -> list[RegimeMetrics]:
    """
    Calculate metrics broken down by regime.

    Args:
        equity_curve: DataFrame with equity values
        regime_history: DataFrame with regime labels

    Returns:
        List of RegimeMetrics
    """
    metrics = []

    # Merge data
    merged = equity_curve.copy()
    merged['regime_label'] = regime_history['regime_label']
    merged['regime_id'] = regime_history['regime_id']

    # Calculate returns
    merged['return'] = merged['equity'].pct_change()

    # Group by regime
    for regime in merged['regime_label'].unique():
        if regime in ['WARMUP', 'ERROR', 'N/A']:
            continue

        regime_data = merged[merged['regime_label'] == regime]

        if len(regime_data) < 2:
            continue

        # Time in regime
        time_in_pct = len(regime_data) / len(merged) * 100

        # Return contribution
        start_equity = merged['equity'].iloc[0]
        end_equity = regime_data['equity'].iloc[-1]
        regime_return = ((end_equity - start_equity) / start_equity) * 100 if start_equity > 0 else 0

        # Calculate trades in this regime
        regime_trades = regime_data[ regime_data['regime_id'].shift(1) != regime_data['regime_id']]
        num_trades = len(regime_trades)

        if num_trades > 0:
            trade_returns = []
            for i in range(1, len(regime_data)):
                prev = regime_data.iloc[i-1]['equity']
                curr = regime_data.iloc[i]['equity']
                if prev > 0:
                    trade_returns.append((curr - prev) / prev)

            if trade_returns:
                winning = [r for r in trade_returns if r > 0]
                win_rate = len(winning) / len(trade_returns) * 100 if trade_returns else 0

                # Sharpe-like metric
                if np.std(trade_returns) > 0:
                    sharpe = (np.mean(trade_returns) * 252) / (np.std(trade_returns) * np.sqrt(252))
                else:
                    sharpe = 0

                avg_pnl = np.mean(trade_returns) * 100
            else:
                win_rate = 0
                sharpe = 0
                avg_pnl = 0
        else:
            win_rate = 0
            sharpe = 0
            avg_pnl = 0

        metrics.append(RegimeMetrics(
            regime_label=regime,
            time_in_pct=time_in_pct,
            return_contribution_pct=regime_return,
            avg_trade_pnl_pct=avg_pnl,
            win_rate_pct=win_rate,
            sharpe_ratio=sharpe,
        ))

    return metrics


def calculate_confidence_metrics(
    trade_log: pd.DataFrame,
) -> list[ConfidenceMetrics]:
    """
    Calculate metrics bucketed by confidence level.

    Args:
        trade_log: DataFrame with trade records including confidence_bucket

    Returns:
        List of ConfidenceMetrics
    """
    metrics = []

    if len(trade_log) == 0:
        return metrics

    # Calculate P&L for each trade
    trade_log = trade_log.copy()
    trade_log['pnl_pct'] = trade_log['equity'].pct_change() * 100

    for bucket in ['<50%', '50-60%', '60-70%', '70+%']:
        bucket_data = trade_log[trade_log['confidence_bucket'] == bucket]

        if len(bucket_data) < 2:
            continue

        returns = bucket_data['pnl_pct'].dropna()
        if len(returns) == 0:
            continue

        winning = returns[returns > 0]
        win_rate = len(winning) / len(returns) * 100 if len(returns) > 0 else 0
        avg_pnl = np.mean(returns)

        # Sharpe ratio
        if np.std(returns) > 0:
            sharpe = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))
        else:
            sharpe = 0

        metrics.append(ConfidenceMetrics(
            confidence_bucket=bucket,
            num_trades=len(returns),
            sharpe_ratio=sharpe,
            win_rate_pct=win_rate,
            avg_pnl_pct=avg_pnl,
        ))

    return metrics


def calculate_benchmark_metrics(
    data: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_capital: float,
    risk_free_rate: float = 0.0,
) -> list[BenchmarkMetrics]:
    """
    Calculate benchmark comparison metrics.

    Benchmarks:
    1. Buy-and-hold: hold the asset entire period
    2. 200 SMA trend: long above 200 SMA, cash below
    3. Random entry: random allocation changes at same frequency

    Args:
        data: DataFrame with OHLCV data
        equity_curve: Strategy equity curve
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate

    Returns:
        List of BenchmarkMetrics
    """
    benchmarks = []

    close = data['close']
    start_idx = 200  # Need data for 200 SMA

    if len(data) < start_idx + 1:
        return benchmarks

    # Buy and hold benchmark
    bh_start_price = close.iloc[start_idx]
    bh_end_price = close.iloc[-1]
    bh_return = ((bh_end_price - bh_start_price) / bh_start_price) * 100

    # CAGR for buy-hold
    days = (data.index[-1] - data.index[start_idx]).days
    years = days / 365.25
    bh_cagr = (((bh_end_price / bh_start_price) ** (1 / years)) - 1) * 100 if years > 0 else 0

    # Max drawdown for buy-hold
    bh_equity = close.iloc[start_idx:].values
    bh_rolling_max = np.maximum.accumulate(bh_equity)
    bh_drawdowns = (bh_equity - bh_rolling_max) / bh_rolling_max * 100
    bh_max_dd = np.min(bh_drawdowns)

    # Sharpe for buy-hold
    bh_returns = np.diff(bh_equity) / bh_equity[:-1]
    bh_vol = np.std(bh_returns) * np.sqrt(252) * 100 if len(bh_returns) > 0 else 0
    bh_sharpe = bh_cagr / bh_vol if bh_vol > 0 else 0

    benchmarks.append(BenchmarkMetrics(
        name="Buy & Hold",
        total_return_pct=bh_return,
        cagr_pct=bh_cagr,
        sharpe_ratio=bh_sharpe,
        max_drawdown_pct=bh_max_dd,
    ))

    # 200 SMA trend benchmark
    sma200 = close.rolling(window=200).mean()

    # Generate signals
    in_position = False
    sma_trades = []
    sma_equity = [initial_capital]

    for i in range(start_idx, len(data)):
        price = close.iloc[i]
        sma_val = sma200.iloc[i]

        if pd.isna(sma_val):
            sma_equity.append(sma_equity[-1])
            continue

        above_sma = price > sma_val

        if above_sma and not in_position:
            # Go long
            in_position = True
            entry_price = price
        elif not above_sma and in_position:
            # Exit to cash
            pnl = (price - entry_price) / entry_price * sma_equity[-1]
            sma_equity[-1] += pnl
            in_position = False

        # Mark to market
        if in_position:
            mtm = (price - entry_price) / entry_price * initial_capital
            sma_equity.append(initial_capital + mtm)
        else:
            sma_equity.append(sma_equity[-1])

    sma_return = ((sma_equity[-1] - initial_capital) / initial_capital) * 100

    # Calculate metrics for SMA
    sma_returns_arr = np.diff(sma_equity) / np.array(sma_equity[:-1])
    sma_vol = np.std(sma_returns_arr) * np.sqrt(252) * 100 if len(sma_returns_arr) > 0 else 0
    sma_cagr = (((sma_equity[-1] / initial_capital) ** (1 / years)) - 1) * 100 if years > 0 else 0
    sma_sharpe = sma_cagr / sma_vol if sma_vol > 0 else 0

    sma_rolling_max = np.maximum.accumulate(sma_equity)
    sma_drawdowns = (np.array(sma_equity) - sma_rolling_max) / sma_rolling_max * 100
    sma_max_dd = np.min(sma_drawdowns)

    benchmarks.append(BenchmarkMetrics(
        name="200 SMA Trend",
        total_return_pct=sma_return,
        cagr_pct=sma_cagr,
        sharpe_ratio=sma_sharpe,
        max_drawdown_pct=sma_max_dd,
    ))

    # Random benchmark (100 seeds, report mean/std)
    random_returns = []
    random_sharpes = []
    random_max_dds = []

    np.random.seed(42)
    for seed in range(100):
        np.random.seed(seed)

        # Generate random allocation changes at similar frequency to strategy
        n_rebalances = len(equity_curve) // 20  # Roughly similar frequency
        rebalance_indices = np.random.choice(len(equity_curve), n_rebalances, replace=False)
        rebalance_indices = sorted(rebalance_indices)

        random_equity = [initial_capital]
        in_position = False

        for i in range(start_idx, len(data)):
            if i in rebalance_indices:
                in_position = not in_position

            price = close.iloc[i]
            if i == start_idx:
                entry = price

            if in_position:
                mtm = (price - entry) / entry * initial_capital
                random_equity.append(initial_capital + mtm)
            else:
                random_equity.append(random_equity[-1])

            entry = price

        random_returns.append(((random_equity[-1] - initial_capital) / initial_capital) * 100)

        r_rets = np.diff(random_equity) / np.array(random_equity[:-1])
        r_vol = np.std(r_rets) * np.sqrt(252) * 100 if len(r_rets) > 0 else 0
        r_cagr = (((random_equity[-1] / initial_capital) ** (1 / years)) - 1) * 100 if years > 0 else 0
        random_sharpes.append(r_cagr / r_vol if r_vol > 0 else 0)

        r_rolling_max = np.maximum.accumulate(random_equity)
        r_drawdowns = (np.array(random_equity) - r_rolling_max) / r_rolling_max * 100
        random_max_dds.append(np.min(r_drawdowns))

    benchmarks.append(BenchmarkMetrics(
        name="Random Entry (mean)",
        total_return_pct=np.mean(random_returns),
        cagr_pct=np.mean(random_sharpes) * np.mean(random_sharpes),  # Approximate
        sharpe_ratio=np.mean(random_sharpes),
        max_drawdown_pct=np.mean(random_max_dds),
    ))

    return benchmarks


# =============================================================================
# Formatting and Output
# =============================================================================

def format_core_metrics(metrics: PerformanceMetrics) -> str:
    """Format core metrics as a table."""
    table = [
        ["Total Return", f"{metrics.total_return_pct:.2f}%"],
        ["CAGR", f"{metrics.cagr_pct:.2f}%"],
        ["Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}"],
        ["Sortino Ratio", f"{metrics.sortino_ratio:.2f}"],
        ["Calmar Ratio", f"{metrics.calmar_ratio:.2f}"],
        ["Max Drawdown", f"{metrics.max_drawdown_pct:.2f}%"],
        ["Max DD Duration", f"{metrics.max_drawdown_days} days"],
        ["", ""],
        ["Total Trades", str(metrics.total_trades)],
        ["Win Rate", f"{metrics.win_rate_pct:.1f}%"],
        ["Avg Win", f"{metrics.avg_win_pct:.2f}%"],
        ["Avg Loss", f"{metrics.avg_loss_pct:.2f}%"],
        ["Profit Factor", f"{metrics.profit_factor:.2f}"],
        ["", ""],
        ["Annualized Vol", f"{metrics.volatility_annualized_pct:.2f}%"],
        ["Skewness", f"{metrics.skewness:.2f}"],
        ["Kurtosis", f"{metrics.kurtosis:.2f}"],
    ]

    return tabulate(table, headers=["Metric", "Value"], tablefmt="grid")


def format_regime_table(regime_metrics: list[RegimeMetrics]) -> str:
    """Format regime metrics as a table."""
    if not regime_metrics:
        return "No regime metrics available"

    table = [
        [
            m.regime_label,
            f"{m.time_in_pct:.1f}%",
            f"{m.return_contribution_pct:.2f}%",
            f"{m.avg_trade_pnl_pct:.2f}%",
            f"{m.win_rate_pct:.1f}%",
            f"{m.sharpe_ratio:.2f}",
        ]
        for m in regime_metrics
    ]

    return tabulate(
        table,
        headers=["Regime", "% Time", "Return Contrib", "Avg P&L", "Win Rate", "Sharpe"],
        tablefmt="grid",
    )


def format_confidence_table(confidence_metrics: list[ConfidenceMetrics]) -> str:
    """Format confidence bucket metrics as a table."""
    if not confidence_metrics:
        return "No confidence metrics available"

    table = [
        [
            m.confidence_bucket,
            str(m.num_trades),
            f"{m.sharpe_ratio:.2f}",
            f"{m.win_rate_pct:.1f}%",
            f"{m.avg_pnl_pct:.2f}%",
        ]
        for m in confidence_metrics
    ]

    return tabulate(
        table,
        headers=["Confidence", "Trades", "Sharpe", "Win Rate", "Avg P&L"],
        tablefmt="grid",
    )


def format_benchmark_table(benchmarks: list[BenchmarkMetrics]) -> str:
    """Format benchmark comparisons as a table."""
    if not benchmarks:
        return "No benchmarks available"

    table = [
        [
            b.name,
            f"{b.total_return_pct:.2f}%",
            f"{b.cagr_pct:.2f}%",
            f"{b.sharpe_ratio:.2f}",
            f"{b.max_drawdown_pct:.2f}%",
        ]
        for b in benchmarks
    ]

    return tabulate(
        table,
        headers=["Benchmark", "Total Return", "CAGR", "Sharpe", "Max DD"],
        tablefmt="grid",
    )


def format_worst_case(metrics: PerformanceMetrics) -> str:
    """Format worst-case metrics."""
    table = [
        ["Worst Day", f"{metrics.worst_day_pct:.2f}%"],
        ["Worst Week", f"{metrics.worst_week_pct:.2f}%"],
        ["Worst Month", f"{metrics.worst_month_pct:.2f}%"],
        ["Max Consecutive Losses", str(metrics.max_consecutive_losses)],
        ["Longest Underwater", f"{metrics.longest_underwater_days} days"],
    ]

    return tabulate(table, headers=["Metric", "Value"], tablefmt="grid")


def generate_full_report(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
    regime_history: pd.DataFrame,
    benchmark_data: Optional[pd.DataFrame] = None,
    initial_capital: float = 100000.0,
    include_benchmarks: bool = False,
) -> FullPerformanceReport:
    """
    Generate a complete performance report.

    Args:
        equity_curve: DataFrame with equity values
        trade_log: DataFrame with trade records
        regime_history: DataFrame with regime labels
        benchmark_data: DataFrame with OHLCV data for benchmarks
        initial_capital: Starting capital
        include_benchmarks: Whether to calculate benchmarks

    Returns:
        FullPerformanceReport
    """
    # Core metrics
    core = calculate_core_metrics(equity_curve, trade_log)

    # Regime metrics
    regime_metrics = calculate_regime_metrics(equity_curve, regime_history)

    # Confidence metrics
    confidence_metrics = calculate_confidence_metrics(trade_log)

    # Benchmarks
    benchmarks = []
    if include_benchmarks and benchmark_data is not None:
        benchmarks = calculate_benchmark_metrics(
            benchmark_data, equity_curve, initial_capital
        )

    # Worst case
    worst_case = {
        'worst_day_pct': core.worst_day_pct,
        'worst_week_pct': core.worst_week_pct,
        'worst_month_pct': core.worst_month_pct,
        'max_consecutive_losses': core.max_consecutive_losses,
        'longest_underwater_days': core.longest_underwater_days,
    }

    return FullPerformanceReport(
        core=core,
        regime_metrics=regime_metrics,
        confidence_metrics=confidence_metrics,
        benchmarks=benchmarks,
        worst_case=worst_case,
    )


def print_report(report: FullPerformanceReport):
    """Print a complete performance report to console."""
    print("\n" + "=" * 60)
    print("PERFORMANCE REPORT")
    print("=" * 60 + "\n")

    print("CORE METRICS")
    print("-" * 40)
    print(format_core_metrics(report.core))

    print("\n\nREGIME BREAKDOWN")
    print("-" * 40)
    print(format_regime_table(report.regime_metrics))

    print("\n\nCONFIDENCE BUCKETS")
    print("-" * 40)
    print(format_confidence_table(report.confidence_metrics))

    if report.benchmarks:
        print("\n\nBENCHMARK COMPARISON")
        print("-" * 40)
        print(format_benchmark_table(report.benchmarks))

    print("\n\nWORST CASE")
    print("-" * 40)
    print(format_worst_case(report.core))

    print("\n" + "=" * 60)


def save_results(
    result,
    output_dir: str = ".",
    prefix: str = "backtest",
):
    """
    Save backtest results to CSV files.

    Args:
        result: BacktestResult from backtester.py
        output_dir: Directory to save files
        prefix: Filename prefix
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Equity curve
    equity_path = os.path.join(output_dir, f"{prefix}_equity_curve.csv")
    result.equity_curve.to_csv(equity_path)

    # Trade log
    if len(result.trade_log) > 0:
        trade_path = os.path.join(output_dir, f"{prefix}_trade_log.csv")
        result.trade_log.to_csv(trade_path)

    # Regime history
    regime_path = os.path.join(output_dir, f"{prefix}_regime_history.csv")
    result.regime_history.to_csv(regime_path)

    print(f"Saved results to {output_dir}/")


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_backtest(
    result,
    include_benchmarks: bool = False,
    benchmark_data: Optional[pd.DataFrame] = None,
) -> FullPerformanceReport:
    """
    Analyze a backtest result and generate report.

    Args:
        result: BacktestResult from backtester.py
        include_benchmarks: Whether to calculate benchmarks
        benchmark_data: DataFrame with OHLCV data for benchmarks

    Returns:
        FullPerformanceReport
    """
    return generate_full_report(
        equity_curve=result.equity_curve,
        trade_log=result.trade_log,
        regime_history=result.regime_history,
        benchmark_data=benchmark_data,
        initial_capital=result.config.initial_capital,
        include_benchmarks=include_benchmarks,
    )