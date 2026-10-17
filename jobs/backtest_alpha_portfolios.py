"""Run a cost-aware backtest on exported OOS scores and adjusted prices.

Price CSV contract: long-format columns date,ticker,adjusted_close_base.
Adjusted prices must include distributions/splits and be converted into the
same base currency (for example USD), including FX changes for foreign listings.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core.alpha_portfolio_backtest import (
    PortfolioBacktestConfig,
    run_alpha_portfolio_backtest,
)


def _load_predictions(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    required = {"date", "ticker", "expected_excess_return"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(
            f"predictions CSV is missing required columns: {sorted(missing)}"
        )
    rows["date"] = pd.to_datetime(rows["date"], errors="raise")
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
    parser.add_argument("--min-cross-section", type=int, default=20)
    parser.add_argument("--momentum-lookback", type=int, default=60)
    parser.add_argument("--max-stale-sessions", type=int, default=3)
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
        predictions = _load_predictions(args.predictions_csv)
        prices = _load_adjusted_price_panel(args.prices_csv)
        report = run_alpha_portfolio_backtest(
            predictions,
            prices,
            benchmark_ticker=args.benchmark_ticker,
            config=config,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.summary.to_csv(args.output_dir / "alpha_portfolio_summary.csv", index=False)
    report.daily_returns.to_csv(args.output_dir / "alpha_portfolio_daily_returns.csv")
    report.holdings.to_csv(args.output_dir / "alpha_portfolio_holdings.csv", index=False)
    print("\n=== OOS Portfolio Backtest (net of configured costs) ===")
    print(report.summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"CSV reports written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
