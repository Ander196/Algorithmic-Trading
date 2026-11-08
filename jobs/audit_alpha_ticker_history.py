"""Read-only Yahoo vs Supabase price-history audit for one Alpha ticker.

The command compares adjusted OHLCV histories, flags large observations and
price-basis breaks, and records Yahoo corporate-action metadata. It never
writes to Supabase or emits a repair SQL script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from core.alpha_price_anomaly_audit import compare_adjusted_price_histories
from jobs.audit_alpha_price_coverage import (
    _fetch_supabase_prices,
    _normalize_yahoo_history,
)
from jobs.common import get_supabase_client


def _action_series(raw_history: pd.DataFrame, column: str, index: pd.DatetimeIndex) -> pd.Series:
    if column not in raw_history.columns:
        return pd.Series(0.0, index=index)
    values = pd.to_numeric(raw_history[column], errors="coerce")
    dates = pd.DatetimeIndex(raw_history.index)
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    values.index = dates.normalize()
    return values.reindex(index).fillna(0.0)


def audit_ticker_history(
    ticker: str,
    output_dir: Path,
    *,
    client=None,
) -> dict[str, object]:
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("ticker must not be empty")
    client = client if client is not None else get_supabase_client()

    stored = _fetch_supabase_prices(client, ticker, "1900-01-01")
    if stored.empty:
        raise RuntimeError(f"No Supabase price history found for {ticker}")

    start = stored.index.min().strftime("%Y-%m-%d")
    end = (stored.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw_reference = yf.Ticker(ticker).history(
        start=start,
        end=end,
        auto_adjust=False,
        actions=True,
    )
    if raw_reference.empty:
        raise RuntimeError(f"Yahoo Finance returned no history for {ticker}")

    reference = _normalize_yahoo_history(raw_reference)
    reference.index = pd.DatetimeIndex(reference.index).normalize()
    reference["dividends"] = _action_series(
        raw_reference, "Dividends", reference.index
    )
    reference["stock_splits"] = _action_series(
        raw_reference, "Stock Splits", reference.index
    )

    comparison, summary = compare_adjusted_price_histories(
        stored,
        reference,
        ticker=ticker,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / f"{ticker.lower()}_price_history_comparison.csv"
    summary_path = output_dir / f"{ticker.lower()}_price_history_summary.csv"
    comparison.to_csv(comparison_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print(f"Ticker: {ticker}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Daily comparison: {comparison_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print("Read-only audit: Supabase was not modified and no repair SQL was generated.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare one ticker's stored adjusted price history with Yahoo Finance"
    )
    parser.add_argument("--ticker", default="CPWR")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/alpha_price_anomalies"),
    )
    args = parser.parse_args()
    try:
        audit_ticker_history(args.ticker, args.output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
