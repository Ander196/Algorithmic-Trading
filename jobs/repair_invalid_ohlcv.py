"""Refresh invalid OHLCV rows from a consistent adjusted provider history.

By default the command only creates a CSV preview. Database updates require
explicit --apply and are restricted to dates present in the input audit CSV for
which yfinance currently returns a valid daily bar.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv(".env.secrets")

PRICE_COLUMNS = ("open", "high", "low", "close")
INPUT_REQUIRED_COLUMNS = {"ticker", "price_date", *PRICE_COLUMNS, "volume"}
REPORT_COLUMNS = [
    "ticker",
    "price_date",
    "reasons",
    "repair_status",
    "repair_error",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "proposed_open",
    "proposed_high",
    "proposed_low",
    "proposed_close",
    "proposed_adj_close",
    "proposed_volume",
]


def fetch_adjusted_history(
    ticker: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch adjusted OHLCV bars; all OHLC fields use one adjustment basis."""
    return yf.Ticker(ticker).history(
        start=start,
        end=end,
        auto_adjust=True,
    )


def normalize_provider_history(history: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize yfinance history to a frame indexed by YYYY-MM-DD."""
    columns = ["open", "high", "low", "close", "volume"]
    if history is None or history.empty:
        return pd.DataFrame(columns=columns)

    frame = history.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    renamed = {}
    for column in frame.columns:
        normalized = str(column).strip().lower().replace(" ", "_")
        if normalized == "adj_close":
            normalized = "adj_close"
        renamed[column] = normalized
    frame = frame.rename(columns=renamed)

    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(
            "provider history missing required columns: "
            + ", ".join(sorted(missing))
        )

    frame["price_date"] = pd.to_datetime(frame.index).strftime("%Y-%m-%d")
    frame = frame.reset_index(drop=True)
    frame = frame.drop_duplicates("price_date", keep="last").set_index("price_date")

    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # With auto_adjust=True the returned Close is already adjusted on the same
    # basis as Open/High/Low, so retain it as adj_close for downstream consistency.
    frame["adj_close"] = frame["close"]
    return frame


def source_bar_invalid_reasons(bar: pd.Series) -> list[str]:
    """Return quality reasons for one candidate replacement daily bar."""
    numeric: dict[str, float] = {}
    for column in (*PRICE_COLUMNS, "volume"):
        try:
            numeric[column] = float(bar[column])
        except (TypeError, ValueError, KeyError):
            return [f"missing_or_non_numeric_{column}"]

    if not all(np.isfinite(numeric[column]) for column in PRICE_COLUMNS):
        return ["non_finite_ohlc_values"]
    if any(numeric[column] <= 0 for column in PRICE_COLUMNS):
        return ["non_positive_ohlc_values"]
    if not np.isfinite(numeric["volume"]):
        return ["non_finite_volume"]
    if numeric["volume"] < 0:
        return ["negative_volume"]
    if numeric["high"] < max(numeric["open"], numeric["low"], numeric["close"]):
        return ["high_below_ohlc_max"]
    if numeric["low"] > min(numeric["open"], numeric["high"], numeric["close"]):
        return ["low_above_ohlc_min"]
    return []


def _normalize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    missing = INPUT_REQUIRED_COLUMNS - set(candidates.columns)
    if missing:
        raise ValueError(
            "input audit CSV missing required columns: " + ", ".join(sorted(missing))
        )

    frame = candidates.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["price_date"] = pd.to_datetime(
        frame["price_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    if frame["ticker"].eq("").any() or frame["price_date"].isna().any():
        raise ValueError("input audit CSV contains blank ticker or invalid price_date")
    return frame


def repair_candidates(
    candidates: pd.DataFrame,
    *,
    fetcher: Callable[[str, datetime, datetime], pd.DataFrame] = fetch_adjusted_history,
    client=None,
    apply: bool = False,
) -> pd.DataFrame:
    """Create repair proposals and optionally update verified rows in Supabase.

    The caller provides the exact rows from the read-only SQL audit export.
    Candidate data are refreshed per ticker/date-range, but only input rows are
    eligible for update. If the provider cannot supply a valid replacement,
    the existing database row is left untouched and reported as unresolved.
    """
    if apply and client is None:
        raise ValueError("client is required when apply=True")

    frame = _normalize_candidates(candidates)
    reports: list[dict] = []

    for ticker, group in frame.groupby("ticker", sort=True):
        dates = pd.to_datetime(group["price_date"])
        start = dates.min().to_pydatetime()
        end = dates.max().to_pydatetime() + timedelta(days=1)

        try:
            provider_raw = fetcher(ticker, start, end)
            provider = normalize_provider_history(provider_raw)
            fetch_error = None
        except Exception as exc:  # provider/network failures should be reported per ticker
            provider = pd.DataFrame()
            fetch_error = str(exc)

        for _, original in group.iterrows():
            date = original["price_date"]
            report = {
                "ticker": ticker,
                "price_date": date,
                "reasons": original.get("reasons", ""),
                "repair_status": "",
                "repair_error": "",
            }
            for column in ("open", "high", "low", "close", "adj_close", "volume"):
                report[column] = original.get(column, np.nan)
                report[f"proposed_{column}"] = np.nan

            if fetch_error is not None:
                report["repair_status"] = "provider_fetch_error"
                report["repair_error"] = fetch_error
                reports.append(report)
                continue

            if date not in provider.index:
                report["repair_status"] = "provider_bar_not_found"
                reports.append(report)
                continue

            source_bar = provider.loc[date]
            if isinstance(source_bar, pd.DataFrame):
                source_bar = source_bar.iloc[-1]
            reasons = source_bar_invalid_reasons(source_bar)
            if reasons:
                report["repair_status"] = "provider_bar_invalid"
                report["repair_error"] = ",".join(reasons)
                reports.append(report)
                continue

            proposed = {
                "open": float(source_bar["open"]),
                "high": float(source_bar["high"]),
                "low": float(source_bar["low"]),
                "close": float(source_bar["close"]),
                "adj_close": float(source_bar["adj_close"]),
                "volume": int(source_bar["volume"]),
            }
            for column, value in proposed.items():
                report[f"proposed_{column}"] = value

            report["repair_status"] = "would_update"
            if apply:
                try:
                    (
                        client.table("stock_prices")
                        .update(proposed)
                        .eq("ticker", ticker)
                        .eq("price_date", date)
                        .execute()
                    )
                    report["repair_status"] = "updated"
                except Exception as exc:
                    report["repair_status"] = "update_error"
                    report["repair_error"] = str(exc)

            reports.append(report)

    return pd.DataFrame(reports, columns=REPORT_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview or apply provider-verified repairs for invalid OHLCV rows"
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="CSV export from scripts/sql/audit_ohlcv_invalid_rows.sql",
    )
    parser.add_argument(
        "--report-csv",
        default="ohlcv_repair_preview.csv",
        help="Path for the repair proposal/result CSV",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Update only audit rows with a valid replacement bar from yfinance",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.is_file():
        parser.error(f"input CSV does not exist: {input_path}")

    candidates = pd.read_csv(input_path)
    client = None
    if args.apply:
        from jobs.common import get_supabase_client

        client = get_supabase_client()

    result = repair_candidates(candidates, client=client, apply=args.apply)
    result.to_csv(args.report_csv, index=False)

    print(f"Audit rows processed: {len(result)}")
    print(f"Report written: {args.report_csv}")
    if "repair_status" in result:
        print(result["repair_status"].value_counts(dropna=False).to_string())
    if not args.apply:
        print("DRY RUN: Supabase was not modified. Review the report before using --apply.")


if __name__ == "__main__":
    main()
