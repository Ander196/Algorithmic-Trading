"""
Historical OHLCV loader for active and curated retired tickers.

The loader:
- reads active tickers and registered historical-only symbols from stocks;
- preserves existing history in stock_prices;
- backfills from 2000-01-01 where history is missing;
- fetches recent data from the latest stored date through today;
- excludes tickers known to have invalid historical prices.

Usage:
    python -m data.data_loader_historical
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import supabase
import yfinance as yf
from dotenv import load_dotenv

from data.excluded_tickers import normalize_ticker
from data.sp500_universe import (
    MembershipPeriod,
    SP500_BENCHMARK_TICKER,
    SP500_HISTORY_START,
    filter_sp500_price_rows,
    get_sp500_membership_periods,
    get_sp500_tickers,
    get_yfinance_symbol,
)
from data.price_validation import validate_ohlcv_frame

load_dotenv(".env.secrets")

# Retain enough history for Alpha while preventing future max-history backfills
# from reintroducing observations that were intentionally removed.
HISTORICAL_START_DATE = datetime.combine(SP500_HISTORY_START, datetime.min.time()).replace(tzinfo=timezone.utc)


def getSupabaseClient() -> supabase.Client:
    """Initialize and return Supabase client."""
    supabaseUrl = os.getenv("SUPABASE_URL")
    supabaseKey = os.getenv("SUPABASE_KEY")

    if not supabaseUrl or not supabaseKey:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY must be set in .env.secrets"
        )

    return supabase.create_client(supabaseUrl, supabaseKey)


def getActiveTickers(client: supabase.Client) -> list[str]:
    """Return every S&P 500 ticker from the historical point-in-time catalog.

    ticker_loader.py must be run first so every symbol exists in stocks and
    satisfies the stock_prices foreign key. SPY is additionally loaded as the
    benchmark series; it is not treated as an index constituent.
    """
    return sorted(set(get_sp500_tickers()) | {SP500_BENCHMARK_TICKER})

def getLatestPriceDate(
    client: supabase.Client,
    ticker: str,
    start_date: str | None = None,
    end_date_exclusive: str | None = None,
) -> datetime | None:
    """Get the latest stored price date inside an optional membership window."""
    query = (
        client.table("stock_prices")
        .select("price_date")
        .eq("ticker", ticker.upper())
    )
    if start_date:
        query = query.gte("price_date", start_date)
    if end_date_exclusive:
        query = query.lt("price_date", end_date_exclusive)
    response = query.order("price_date", desc=True).limit(1).execute()
    if not response.data:
        return None
    return datetime.strptime(
        response.data[0]["price_date"], "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc)


def getEarliestPriceDate(
    client: supabase.Client,
    ticker: str,
    start_date: str | None = None,
    end_date_exclusive: str | None = None,
) -> datetime | None:
    """Get the earliest stored price date inside an optional membership window."""
    query = (
        client.table("stock_prices")
        .select("price_date")
        .eq("ticker", ticker.upper())
    )
    if start_date:
        query = query.gte("price_date", start_date)
    if end_date_exclusive:
        query = query.lt("price_date", end_date_exclusive)
    response = query.order("price_date", desc=False).limit(1).execute()
    if not response.data:
        return None
    return datetime.strptime(
        response.data[0]["price_date"], "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc)

def fetchPriceData(
    ticker: str,
    startDate: datetime | None,
    endDate: datetime,
    retries: int = 2,
) -> pd.DataFrame | None:
    """Fetch OHLCV data from yfinance.

    History is bounded to HISTORICAL_START_DATE or the later requested start date.
    """
    for attempt in range(retries):
        try:
            stock = yf.Ticker(get_yfinance_symbol(ticker))

            effectiveStart = startDate or HISTORICAL_START_DATE
            if effectiveStart < HISTORICAL_START_DATE:
                effectiveStart = HISTORICAL_START_DATE
            df = stock.history(start=effectiveStart, end=endDate)

            if df.empty:
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                    "Adj Close": "adj_close",
                }
            )

            keepCols = ["open", "high", "low", "close", "volume"]
            missing = sorted(set(keepCols) - set(df.columns))
            if missing:
                print(
                    f"  {ticker}: rejecting provider response; missing required "
                    f"columns: {', '.join(missing)}"
                )
                return None

            df = df[keepCols].copy()
            df["ticker"] = ticker.upper()
            df["price_date"] = pd.to_datetime(df.index).strftime("%Y-%m-%d")
            df = df.reset_index(drop=True)

            try:
                adjusted_start = max(
                    startDate or HISTORICAL_START_DATE,
                    HISTORICAL_START_DATE,
                )
                adjusted = stock.history(
                    start=adjusted_start,
                    end=endDate,
                    auto_adjust=False,
                )
                if isinstance(adjusted.columns, pd.MultiIndex):
                    adjusted.columns = adjusted.columns.get_level_values(0)
                adjusted.index = pd.to_datetime(adjusted.index).strftime("%Y-%m-%d")
                adj_series = pd.to_numeric(adjusted["Adj Close"], errors="coerce")
                adj_map = adj_series.to_dict()
                df["adj_close"] = [
                    adj_map.get(date, close)
                    for date, close in zip(df["price_date"], df["close"])
                ]
            except Exception:
                df["adj_close"] = df["close"]

            df = validate_ohlcv_frame(df, ticker)
            if df is None or df.empty:
                return None
            return df

        except Exception:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
            continue

    return None


def uploadToSupabase(
    client: supabase.Client,
    ticker: str,
    df: pd.DataFrame,
    batchSize: int = 500,
) -> int:
    """Upload only rows that fall inside the ticker's S&P membership window."""
    ticker = normalize_ticker(ticker)
    if ticker == SP500_BENCHMARK_TICKER:
        # SPY is the market benchmark and trades independently of index membership.
        eligible = df.copy()
    else:
        eligible = filter_sp500_price_rows(df, ticker)

    eligible = validate_ohlcv_frame(eligible, ticker)
    if eligible is None or eligible.empty:
        return 0

    records = []
    now = datetime.now(timezone.utc).isoformat()
    for _, row in eligible.iterrows():
        records.append({
            "ticker": ticker,
            "price_date": row["price_date"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "adj_close": (
                float(row["adj_close"])
                if pd.notna(row["adj_close"])
                else float(row["close"])
            ),
            "volume": int(row["volume"]),
            "inserted_at": now,
        })

    inserted = 0
    for i in range(0, len(records), batchSize):
        batch = records[i:i + batchSize]
        try:
            client.table("stock_prices").insert(batch).execute()
            inserted += len(batch)
        except Exception as exc:
            print(f"    Error inserting batch for {ticker}: {exc}")
            for record in batch:
                try:
                    client.table("stock_prices").insert(record).execute()
                    inserted += 1
                except Exception:
                    # Existing logical rows are skipped; unique-key duplicates
                    # are expected when overlapping backfill ranges are retried.
                    pass
    return inserted

def processTicker(
    client: supabase.Client,
    ticker: str,
    years_of_history: int | None = None,
    force: bool = False,
    max_days_stale: int = 3,
) -> dict[str, Any]:
    """Backfill each S&P membership interval without crossing index exits.

    Each re-entry is processed independently. This prevents provider rows and
    stored observations outside the constituent windows from being ingested.
    """
    ticker = normalize_ticker(ticker)
    result = {
        "ticker": ticker,
        "fetched": 0,
        "inserted": 0,
        "skipped": 0,
        "error": None,
    }

    if ticker == SP500_BENCHMARK_TICKER:
        periods = (MembershipPeriod(
            ticker=ticker,
            start_date=SP500_HISTORY_START,
            end_date_exclusive=None,
        ),)
    else:
        periods = get_sp500_membership_periods(ticker)

    if not periods:
        result["skipped"] = 1
        print(f"  {ticker}: not present in the S&P 500 membership file")
        return result

    try:
        today = datetime.now(timezone.utc)
        stale_date = today - timedelta(days=max_days_stale)
        requested_start = (
            HISTORICAL_START_DATE
            if years_of_history is None
            else max(
                HISTORICAL_START_DATE,
                today - timedelta(days=365 * years_of_history),
            )
        )
        fetched_any = False

        for period in periods:
            period_start = datetime.combine(
                period.start_date, datetime.min.time(), tzinfo=timezone.utc
            )
            period_end = (
                datetime.combine(
                    period.end_date_exclusive,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                )
                if period.end_date_exclusive is not None
                else today
            )
            window_start = max(period_start, requested_start)
            window_end = min(period_end, today)

            if window_start >= window_end:
                continue

            start_text = window_start.date().isoformat()
            end_text = (
                period.end_date_exclusive.isoformat()
                if period.end_date_exclusive is not None
                else (today.date() + timedelta(days=1)).isoformat()
            )

            earliest = getEarliestPriceDate(
                client, ticker, start_text, end_text
            )
            latest = getLatestPriceDate(
                client, ticker, start_text, end_text
            )

            # Missing interval: request the complete membership window.
            if earliest is None or latest is None or force:
                fetch_start = window_start
                fetch_end = (
                    min(
                        datetime.combine(
                            period.end_date_exclusive,
                            datetime.min.time(),
                            tzinfo=timezone.utc,
                        ),
                        today,
                    )
                    if period.end_date_exclusive is not None
                    else today
                )
                label = "full membership window" if force or earliest is None else "refresh"
                print(
                    f"  {ticker}: fetching {label} "
                    f"[{fetch_start.date()}, {end_text})"
                )
            else:
                # Fill a missing prefix inside this same membership interval.
                if earliest > window_start:
                    backfill = fetchPriceData(
                        ticker,
                        startDate=window_start,
                        endDate=earliest,
                    )
                    if backfill is not None and not backfill.empty:
                        result["fetched"] += len(backfill)
                        result["inserted"] += uploadToSupabase(
                            client, ticker, backfill
                        )
                        fetched_any = True

                # Closed membership intervals do not need daily refreshes.
                # Try a final tail only when coverage ends materially early.
                if period.end_date_exclusive is not None:
                    if latest.date() >= (
                        period.end_date_exclusive - timedelta(days=7)
                    ):
                        continue
                    fetch_start = latest + timedelta(days=1)
                    fetch_end = window_end
                    print(
                        f"  {ticker}: filling membership tail "
                        f"from {fetch_start.date()} to {end_text}"
                    )
                else:
                    if latest.date() >= stale_date.date():
                        continue
                    fetch_start = latest + timedelta(days=1)
                    fetch_end = today
                    print(
                        f"  {ticker}: updating active membership from "
                        f"{fetch_start.date()} to {today.date()}"
                    )

            if fetch_start >= fetch_end:
                continue

            frame = fetchPriceData(
                ticker,
                startDate=fetch_start,
                endDate=fetch_end,
            )
            if frame is None or frame.empty:
                continue

            result["fetched"] += len(frame)
            result["inserted"] += uploadToSupabase(client, ticker, frame)
            fetched_any = True

        if not fetched_any and result["fetched"] == 0:
            result["skipped"] = 1
            print(f"  {ticker}: no missing S&P membership prices to load")
        else:
            print(
                f"  {ticker}: fetched={result['fetched']}, "
                f"inserted={result['inserted']}"
            )

    except Exception as exc:
        result["error"] = str(exc)
        print(f"  {ticker}: ERROR - {exc}")

    return result

def main() -> None:
    """Load maximum available history for active and registered retired tickers."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load stock history from 2000-01-01 onwards"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="Optional shorter history window in years; default starts at 2000-01-01.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh the requested range; existing rows are preserved.",
    )
    args = parser.parse_args()

    if args.years is not None and args.years <= 0:
        parser.error("--years must be greater than zero")

    client = getSupabaseClient()
    tickers = getActiveTickers(client)
    print(f"Found {len(tickers)} eligible tickers (active plus registered retired)\n")

    totalInserted = 0
    totalFetched = 0
    errors = []

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Processing {ticker}...")
        result = processTicker(
            client,
            ticker,
            years_of_history=args.years,
            force=args.force,
        )

        totalFetched += result["fetched"]
        totalInserted += result["inserted"]

        if result["error"]:
            errors.append(result)

        time.sleep(0.3)

    print(f"\n{'=' * 50}")
    print("SUMMARY:")
    print(f"  Total tickers processed: {len(tickers)}")
    print(f"  Total records fetched: {totalFetched}")
    print(f"  Total records inserted: {totalInserted}")
    print(f"  Errors: {len(errors)}")
    print(f"{'=' * 50}")

    if errors:
        print("\nFailed tickers:")
        for error in errors:
            print(f"  {error['ticker']}: {error['error']}")


if __name__ == "__main__":
    main()
