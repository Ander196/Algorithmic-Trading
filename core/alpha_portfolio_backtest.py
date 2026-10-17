"""Cost-aware out-of-sample Alpha portfolio backtest.

Research-only. A score observed at close t is not assumed executable at that
same close: targets are set at close t+1 and earn returns from close t+1 to
close t+2. Adjusted prices must include distributions/splits and be expressed
in one base currency. Missing prices are not filled by default; a held asset
without a valid next-session return raises an error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    top_n: int = 5
    rebalance_every: int = 5
    transaction_cost_bps: float = 5.0
    min_cross_section: int = 20
    momentum_lookback: int = 60
    max_stale_sessions: int = 0
    annualization: int = 252

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be > 0")
        if self.rebalance_every <= 0:
            raise ValueError("rebalance_every must be > 0")
        if not np.isfinite(self.transaction_cost_bps) or self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be finite and >= 0")
        if self.min_cross_section < self.top_n:
            raise ValueError("min_cross_section must be >= top_n")
        if self.momentum_lookback <= 0:
            raise ValueError("momentum_lookback must be > 0")
        if self.max_stale_sessions < 0:
            raise ValueError("max_stale_sessions must be >= 0")
        if self.annualization <= 0:
            raise ValueError("annualization must be > 0")


@dataclass(frozen=True)
class PortfolioBacktestReport:
    daily_returns: pd.DataFrame
    summary: pd.DataFrame
    holdings: pd.DataFrame


_STRATEGIES = ("ridge_top_n", "momentum_top_n", "equal_weight_universe", "benchmark")


def _naive_dates(values: pd.Index) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(values))
    if dates.tz is not None:
        dates = dates.tz_convert(None)
    return dates.normalize()


def _validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("predictions must be a pandas DataFrame")
    if not isinstance(predictions.index, pd.MultiIndex):
        raise ValueError("predictions must use a MultiIndex of (date, ticker)")
    if list(predictions.index.names) != ["date", "ticker"]:
        raise ValueError("predictions index levels must be named ('date', 'ticker')")
    if "expected_excess_return" not in predictions.columns:
        raise ValueError("predictions missing required column: expected_excess_return")
    rows = predictions.reset_index().copy()
    rows["date"] = _naive_dates(pd.Index(rows["date"]))
    rows["ticker"] = rows["ticker"].astype(str)
    if rows.duplicated(["date", "ticker"]).any():
        raise ValueError("predictions contain duplicate (date, ticker) rows")
    rows["expected_excess_return"] = pd.to_numeric(
        rows["expected_excess_return"], errors="coerce"
    )
    return rows


def _prepare_prices(
    close_prices: pd.DataFrame,
    benchmark_ticker: str,
    config: PortfolioBacktestConfig,
) -> pd.DataFrame:
    if not isinstance(close_prices, pd.DataFrame) or close_prices.empty:
        raise ValueError("close_prices must be a non-empty DataFrame")
    if close_prices.columns.has_duplicates:
        raise ValueError("close_prices columns must be unique")
    prices = close_prices.copy()
    prices.index = _naive_dates(prices.index)
    if prices.index.has_duplicates:
        raise ValueError("close_prices must have unique session dates")
    prices = prices.sort_index()
    prices.columns = prices.columns.astype(str)
    if benchmark_ticker not in prices.columns:
        raise ValueError(f"benchmark ticker {benchmark_ticker!r} is missing from close_prices")
    for column in prices.columns:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    values = prices.to_numpy(dtype=float)
    prices = prices.where(np.isfinite(values) & (values > 0))
    if prices[benchmark_ticker].isna().any():
        raise ValueError("benchmark price history contains missing or non-positive values")
    # Tolerating stale observations is opt-in because it can distort daily risk
    # metrics and move a multi-session price move onto the wrong session.
    if config.max_stale_sessions:
        prices = prices.ffill(limit=config.max_stale_sessions)
    return prices


def _weights_for_date(
    execution_date: pd.Timestamp,
    signal_date: pd.Timestamp,
    rows: pd.DataFrame,
    prices: pd.DataFrame,
    momentum: pd.DataFrame,
    *,
    top_n: int,
    min_cross_section: int,
    score_column: str | None,
) -> dict[str, float] | None:
    if rows.empty:
        return None
    execution_prices = prices.loc[execution_date]
    candidates = rows.loc[
        rows["ticker"].isin(prices.columns)
        & np.isfinite(rows["expected_excess_return"])
    ].copy()
    if candidates.empty:
        return None
    candidates = candidates.loc[candidates["ticker"].map(execution_prices).notna()]
    if score_column == "momentum":
        candidates["__score"] = candidates["ticker"].map(momentum.loc[signal_date])
        candidates = candidates.loc[np.isfinite(candidates["__score"])]
    elif score_column == "expected_excess_return":
        candidates["__score"] = candidates["expected_excess_return"]
        candidates = candidates.loc[np.isfinite(candidates["__score"])]
    if len(candidates) < min_cross_section:
        return None
    if score_column is None:
        selected = sorted(candidates["ticker"].unique().tolist())
    else:
        selected = (
            candidates.sort_values(
                ["__score", "ticker"], ascending=[False, True], kind="mergesort"
            )
            .head(top_n)["ticker"].tolist()
        )
    if not selected:
        return None
    weight = 1.0 / len(selected)
    return {ticker: weight for ticker in selected}


def _portfolio_summary(
    name: str,
    returns: pd.Series,
    turnover: pd.Series,
    costs: pd.Series,
    annualization: int,
) -> dict[str, float | int | str]:
    returns = returns.astype(float)
    equity = (1.0 + returns).cumprod()
    periods = len(returns)
    ending_equity = float(equity.iloc[-1]) if periods else float("nan")
    total_return = ending_equity - 1.0 if periods else float("nan")
    annualized_return = (
        ending_equity ** (annualization / periods) - 1.0
        if periods and ending_equity > 0 else float("nan")
    )
    volatility = (
        float(returns.std(ddof=1) * np.sqrt(annualization))
        if periods > 1 else float("nan")
    )
    mean_return = float(returns.mean()) if periods else float("nan")
    daily_std = float(returns.std(ddof=1)) if periods > 1 else float("nan")
    sharpe = (
        mean_return / daily_std * np.sqrt(annualization)
        if np.isfinite(daily_std) and daily_std > 0 else float("nan")
    )
    running_peak = pd.Series(np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:], index=equity.index) if periods else pd.Series(dtype=float)
    drawdown = equity / running_peak - 1.0 if periods else pd.Series(dtype=float)
    return {
        "strategy": name,
        "periods": periods,
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": volatility,
        "sharpe_ratio_rf0": float(sharpe),
        "max_drawdown": float(drawdown.min()) if periods else float("nan"),
        "average_session_turnover": float(turnover.mean()) if periods else float("nan"),
        "sum_cost_rate": float(costs.sum()) if periods else float("nan"),
    }


def run_alpha_portfolio_backtest(
    oos_predictions: pd.DataFrame,
    close_prices: pd.DataFrame,
    *,
    benchmark_ticker: str = "SPY",
    config: PortfolioBacktestConfig | None = None,
) -> PortfolioBacktestReport:
    """Compare Ridge Top-N, Momentum Top-N, equal-weight universe, and benchmark.

    Signals from close t are executed no earlier than close t+1. The strategy
    holds between scheduled rebalances; weights drift with asset returns rather
    than silently assuming free daily rebalancing. Turnover is gross traded
    notional, sum(abs(new weight - pre-trade drifted weight)), including buys
    and sells. Transaction cost equals turnover * basis-points / 10,000.
    The benchmark is buy-and-hold and pays its initial purchase cost once.
    """
    config = config or PortfolioBacktestConfig()
    predictions = _validate_predictions(oos_predictions)
    prices = _prepare_prices(close_prices, benchmark_ticker, config)
    if predictions.empty:
        raise ValueError("oos_predictions must not be empty")
    missing_tickers = sorted(set(predictions["ticker"]) - set(prices.columns))
    if missing_tickers:
        sample = ", ".join(missing_tickers[:10])
        raise ValueError(f"price panel is missing predicted tickers: {sample}")
    prediction_dates = set(predictions["date"])
    missing_dates = sorted(prediction_dates - set(prices.index))
    if missing_dates:
        sample = ", ".join(pd.Timestamp(date).date().isoformat() for date in missing_dates[:5])
        raise ValueError(f"price panel is missing OOS signal dates: {sample}")
    if predictions.empty:
        raise ValueError("oos_predictions must not be empty")

    signal_dates = sorted(prediction_dates)
    if len(signal_dates) < 2:
        raise ValueError("at least two OOS signal dates are required")
    first_signal_pos = int(prices.index.get_loc(signal_dates[0]))
    last_signal_pos = int(prices.index.get_loc(signal_dates[-1]))
    if first_signal_pos < config.momentum_lookback:
        raise ValueError(
            "close_prices must include at least momentum_lookback sessions before the first OOS signal"
        )

    # Signals are not tradable at the same close used to calculate them. Need
    # an execution close and one later close for the last OOS signal as well.
    first_pos = first_signal_pos + 1
    last_pos = last_signal_pos + 1
    if last_pos >= len(prices.index) - 1:
        raise ValueError(
            "close_prices must include at least two sessions after the latest OOS signal date"
        )
    if first_pos > last_pos:
        raise ValueError("no OOS signal date has a subsequent executable session and return")

    prediction_groups = {
        pd.Timestamp(date): group.copy()
        for date, group in predictions.groupby("date", sort=True)
    }
    momentum = prices.pct_change(
        periods=config.momentum_lookback, fill_method=None
    )
    forward_returns = prices.pct_change(fill_method=None).shift(-1)
    cost_rate = config.transaction_cost_bps / 10_000.0

    holdings: dict[str, dict[str, float]] = {
        "ridge_top_n": {},
        "momentum_top_n": {},
        "equal_weight_universe": {},
    }
    daily_rows: list[dict[str, float | pd.Timestamp]] = []
    holding_rows: list[dict[str, str | float | pd.Timestamp | int]] = []
    session_dates = prices.index[first_pos : last_pos + 1]

    for session_number, execution_date in enumerate(session_dates):
        execution_pos = int(prices.index.get_loc(execution_date))
        signal_date = pd.Timestamp(prices.index[execution_pos - 1])
        day_returns = forward_returns.loc[execution_date]
        if pd.isna(day_returns.get(benchmark_ticker)):
            raise ValueError(f"benchmark has a missing next-session return after {execution_date.date()}")
        rows = prediction_groups.get(signal_date, pd.DataFrame())
        is_rebalance = session_number % config.rebalance_every == 0
        turnover_by_strategy: dict[str, float] = {}
        cost_by_strategy: dict[str, float] = {}

        for name, score_column in (
            ("ridge_top_n", "expected_excess_return"),
            ("momentum_top_n", "momentum"),
            ("equal_weight_universe", None),
        ):
            if is_rebalance or not holdings[name]:
                target = _weights_for_date(
                    pd.Timestamp(execution_date), signal_date, rows, prices, momentum,
                    top_n=config.top_n,
                    min_cross_section=config.min_cross_section,
                    score_column=score_column,
                ) if not rows.empty else None
                if target:
                    previous = holdings[name]
                    all_tickers = set(previous) | set(target)
                    turnover = float(sum(
                        abs(target.get(ticker, 0.0) - previous.get(ticker, 0.0))
                        for ticker in all_tickers
                    ))
                    holdings[name] = target
                else:
                    turnover = 0.0
                turnover_by_strategy[name] = turnover
                cost_by_strategy[name] = turnover * cost_rate
            else:
                turnover_by_strategy[name] = 0.0
                cost_by_strategy[name] = 0.0

        if not daily_rows:
            turnover_by_strategy["benchmark"] = 1.0
            cost_by_strategy["benchmark"] = cost_rate
        else:
            turnover_by_strategy["benchmark"] = 0.0
            cost_by_strategy["benchmark"] = 0.0

        daily_row: dict[str, float | pd.Timestamp] = {"date": pd.Timestamp(execution_date)}
        gross_by_strategy: dict[str, float] = {}
        for name in _STRATEGIES:
            if name == "benchmark":
                gross = float(day_returns[benchmark_ticker])
            elif not holdings[name]:
                gross = 0.0
            else:
                missing = [
                    ticker for ticker in holdings[name]
                    if pd.isna(day_returns.get(ticker))
                ]
                if missing:
                    raise ValueError(
                        f"{name} has missing next-session returns after {execution_date.date()} "
                        f"for held tickers: {', '.join(sorted(missing))}"
                    )
                gross = float(sum(
                    weight * float(day_returns[ticker])
                    for ticker, weight in holdings[name].items()
                ))
            gross_by_strategy[name] = gross
            cost = float(cost_by_strategy.get(name, 0.0))
            daily_row[f"{name}_gross_return"] = gross
            daily_row[f"{name}_turnover"] = float(turnover_by_strategy.get(name, 0.0))
            daily_row[f"{name}_cost_rate"] = cost
            daily_row[f"{name}_net_return"] = gross - cost
            for ticker, weight in sorted(holdings.get(name, {}).items()):
                holding_rows.append({
                    "date": pd.Timestamp(execution_date),
                    "signal_date": signal_date,
                    "strategy": name,
                    "ticker": ticker,
                    "weight": float(weight),
                    "session_number": session_number,
                })
        daily_rows.append(daily_row)

        # Drift relative weights with returns so rebalances alone create trades.
        for name, weights in holdings.items():
            denominator = 1.0 + gross_by_strategy[name]
            if weights and denominator > 0:
                holdings[name] = {
                    ticker: weight * (1.0 + float(day_returns[ticker])) / denominator
                    for ticker, weight in weights.items()
                }

    if not daily_rows:
        raise ValueError("no benchmark sessions with valid forward returns were available")

    daily = pd.DataFrame(daily_rows).set_index("date").sort_index()
    for name in _STRATEGIES:
        daily[f"{name}_equity"] = (1.0 + daily[f"{name}_net_return"]).cumprod()
    summary = pd.DataFrame([
        _portfolio_summary(
            name,
            daily[f"{name}_net_return"],
            daily[f"{name}_turnover"],
            daily[f"{name}_cost_rate"],
            config.annualization,
        )
        for name in _STRATEGIES
    ])
    holdings_frame = pd.DataFrame(
        holding_rows,
        columns=["date", "signal_date", "strategy", "ticker", "weight", "session_number"],
    )
    return PortfolioBacktestReport(daily, summary, holdings_frame)


__all__ = [
    "PortfolioBacktestConfig",
    "PortfolioBacktestReport",
    "run_alpha_portfolio_backtest",
]
