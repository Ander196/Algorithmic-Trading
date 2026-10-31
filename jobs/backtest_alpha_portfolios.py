"""Run a cost-aware backtest on exported OOS scores and adjusted prices.

Price CSV contract: long-format columns date,ticker,adjusted_close_base.
Adjusted prices must include distributions/splits and be converted into the
same base currency (for example USD), including FX changes for foreign listings.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path

import pandas as pd

from data.excluded_tickers import filter_excluded_ticker_rows
from data.sp500_universe import filter_sp500_membership_rows
from core.alpha_portfolio_backtest import (
    PortfolioBacktestConfig,
    run_alpha_portfolio_backtest,
)


def _parse_cost_sensitivity_bps(value: str) -> tuple[float, ...]:
    """Parse unique, non-negative transaction-cost scenarios in basis points."""
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("cost sensitivity must be a comma-separated list such as 5,10,25")
    try:
        scenarios = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("cost sensitivity values must be numeric") from exc
    if any(not math.isfinite(cost) or cost < 0 for cost in scenarios):
        raise ValueError("cost sensitivity values must be finite and >= 0")
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("cost sensitivity values must not contain duplicates")
    return scenarios


def _load_predictions(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    required = {"date", "ticker", "expected_excess_return"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(
            f"predictions CSV is missing required columns: {sorted(missing)}"
        )
    rows["date"] = pd.to_datetime(rows["date"], errors="raise").dt.normalize()
    rows["ticker"] = rows["ticker"].astype(str).str.strip().str.upper()
    rows, _excluded = filter_excluded_ticker_rows(
        rows, include_historical_retired=True
    )
    rows = filter_sp500_membership_rows(
        rows, ticker_column="ticker", date_column="date"
    )
    if rows.empty:
        raise ValueError(
            "predictions contain no tickers inside their historical S&P 500 membership windows"
        )
    return rows.set_index(["date", "ticker"]).sort_index()


def _load_adjusted_price_panel(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    required = {"date", "ticker", "adjusted_close_base"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(
            "prices CSV must be long-format with columns "
            f"date,ticker,adjusted_close_base; missing: {sorted(missing)}"
        )
    rows["date"] = pd.to_datetime(rows["date"], errors="raise")
    rows["ticker"] = rows["ticker"].astype(str)
    if rows.duplicated(["date", "ticker"]).any():
        raise ValueError("prices CSV contains duplicate (date, ticker) rows")
    rows["adjusted_close_base"] = pd.to_numeric(
        rows["adjusted_close_base"], errors="coerce"
    )
    return rows.pivot(
        index="date", columns="ticker", values="adjusted_close_base"
    ).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest Alpha OOS rankings against momentum, universe and benchmark"
    )
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--prices-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-ticker", default="SPY")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument(
        "--cost-sensitivity-bps",
        default="5,10,25",
        help="Comma-separated cost scenarios (default: 5,10,25); writes a comparison CSV.",
    )
    parser.add_argument("--min-cross-section", type=int, default=20)
    parser.add_argument("--momentum-lookback", type=int, default=60)
    parser.add_argument("--max-stale-sessions", type=int, default=0)
    args = parser.parse_args()

    try:
        config = PortfolioBacktestConfig(
            top_n=args.top_n,
            rebalance_every=args.rebalance_every,
            transaction_cost_bps=args.transaction_cost_bps,
            min_cross_section=args.min_cross_section,
            momentum_lookback=args.momentum_lookback,
            max_stale_sessions=args.max_stale_sessions,
        )
        sensitivity_costs = _parse_cost_sensitivity_bps(args.cost_sensitivity_bps)
        predictions = _load_predictions(args.predictions_csv)
        prices = _load_adjusted_price_panel(args.prices_csv)
        report = run_alpha_portfolio_backtest(
            predictions,
            prices,
            benchmark_ticker=args.benchmark_ticker,
            config=config,
        )
        sensitivity_rows = []
        for cost_bps in sensitivity_costs:
            scenario_config = replace(config, transaction_cost_bps=cost_bps)
            scenario_report = (
                report
                if cost_bps == config.transaction_cost_bps
                else run_alpha_portfolio_backtest(
                    predictions,
                    prices,
                    benchmark_ticker=args.benchmark_ticker,
                    config=scenario_config,
                )
            )
            scenario_summary = scenario_report.summary.copy()
            scenario_summary.insert(0, "transaction_cost_bps", cost_bps)
            sensitivity_rows.append(scenario_summary)
        sensitivity = pd.concat(sensitivity_rows, ignore_index=True)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oos_start = report.daily_returns.index.min().date().isoformat()
    oos_end = report.daily_returns.index.max().date().isoformat()
    report.summary.to_csv(args.output_dir / "alpha_portfolio_summary.csv", index=False)
    report.daily_returns.to_csv(args.output_dir / "alpha_portfolio_daily_returns.csv")
    report.holdings.to_csv(args.output_dir / "alpha_portfolio_holdings.csv", index=False)
    sensitivity.to_csv(
        args.output_dir / "alpha_portfolio_cost_sensitivity.csv", index=False
    )
    print(f"\nOOS execution-return window: {oos_start} through {oos_end} ({len(report.daily_returns):,} sessions)")
    print("\n=== OOS Portfolio Backtest (net of configured costs) ===")
    print(report.summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n=== Transaction-Cost Sensitivity ===")
    display_columns = [
        "transaction_cost_bps", "strategy", "annualized_return",
        "annualized_volatility", "sharpe_ratio_rf0", "max_drawdown",
    ]
    print(sensitivity[display_columns].to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))
    print(f"CSV reports written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
