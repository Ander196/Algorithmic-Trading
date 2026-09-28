"""
Historical OHLCV loader for all active tickers.

The loader:
- reads active tickers from the Supabase stocks table;
- preserves existing history in stock_prices;
- backfills older data from yfinance when available;
- fetches recent data from the latest stored date through today;
- uses maximum available yfinance history by default.

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

load_dotenv(".env.secrets")


def getSupabaseClient() -> supabase.Client:
    """Initialize and return Supabase client."""
    supabaseUrl = os.getenv("SUPABASE_URL")
    supabaseKey = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not supabaseUrl or not supabaseKey:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY (or SUPABASE_ANON_KEY) "
            "must be set in .env.secrets"
        )

    return supabase.create_client(supabaseUrl, supabaseKey)


def getActiveTickers(client: supabase.Client) -> list[str]:
    """Fetch all active tickers from stocks table."""
    response = (
        client.table("stocks")
        .select("ticker")
        .eq("is_active", True)
        .execute()
    )
    return [
        str(row["ticker"]).upper()
        for row in response.data
        if row.get("ticker")
    ]


def getLatestPriceDate(
    client: supabase.Client,
    ticker: str,
) -> datetime | None:
    """Get the latest price_date existing in stock_prices for a ticker."""
    response = (
        client.table("stock_prices")
        .select("price_date")
        .eq("ticker", ticker.upper())
        .order("price_date", desc=True)
        .limit(1)
        .execute()
    )
    if not response.data:
        return None

    return datetime.strptime(
        response.data[0]["price_date"], "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc)


def getEarliestPriceDate(
    client: supabase.Client,
    ticker: str,
) -> datetime | None:
    """Get the earliest price_date existing in stock_prices for a ticker."""
    response = (
        client.table("stock_prices")
        .select("price_date")
        .eq("ticker", ticker.upper())
        .order("price_date", desc=False)
        .limit(1)
        .execute()
    )
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

    When startDate is None, yfinance requests maximum available history.
    """
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)

            if startDate is None:
                df = stock.history(period="max", end=endDate)
            else:
                df = stock.history(start=startDate, end=endDate)

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
            for col in keepCols:
                if col not in df.columns:
                    df[col] = 0

            df = df[keepCols].copy()
            df["ticker"] = ticker.upper()
            df["price_date"] = pd.to_datetime(df.index).strftime("%Y-%m-%d")
            df = df.reset_index(drop=True)

            try:
                adjusted = stock.history(
                    **({"period": "max"} if startDate is None else {"start": startDate, "end": endDate}),
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
    """Upload price data to Supabase, skipping duplicate rows."""
    if df.empty:
        return 0

    records = []
    now = datetime.now(timezone.utc).isoformat()

    for _, row in df.iterrows():
        records.append(
            {
                "ticker": ticker.upper(),
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
            }
        )

    inserted = 0
    for i in range(0, len(records), batchSize):
        batch = records[i : i + batchSize]
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
                    # Existing logical rows are skipped.
                    pass

    return inserted


def processTicker(
    client: supabase.Client,
    ticker: str,
    years_of_history: int | None = None,
    force: bool = False,
    max_days_stale: int = 3,
) -> dict[str, Any]:
    """Backfill and update one ticker.

    years_of_history=None is the default and means maximum available
    yfinance history. A positive value keeps the bounded historical mode.
    """
    ticker = ticker.upper()
    result = {
        "ticker": ticker,
        "fetched": 0,
        "inserted": 0,
        "skipped": 0,
        "error": None,
    }

    try:
        today = datetime.now(timezone.utc)
        staleDate = today - timedelta(days=max_days_stale)
        requestedStart = (
            None
            if years_of_history is None
            else today - timedelta(days=365 * years_of_history)
        )

        earliestDate = getEarliestPriceDate(client, ticker)
        latestDate = getLatestPriceDate(client, ticker)

        if latestDate is None:
            startDate = requestedStart
            label = (
                "maximum available history"
                if requestedStart is None
                else f"since {requestedStart.date()}"
            )
            print(f"  {ticker}: No existing data, fetching {label} to {today.date()}")

        else:
            startDate = latestDate + timedelta(days=1)

            # With maximum-history mode, backfill the missing prefix without
            # deleting existing rows. With bounded mode, only backfill to the
            # requested start when the DB starts too late.
            if earliestDate is not None and (
                requestedStart is None or earliestDate > requestedStart
            ):
                print(f"  {ticker}: Backfilling history before {earliestDate.date()}")
                backfill = fetchPriceData(
                    ticker,
                    startDate=requestedStart,
                    endDate=earliestDate,
                )
                if backfill is not None and not backfill.empty:
                    result["fetched"] += len(backfill)
                    result["inserted"] += uploadToSupabase(
                        client, ticker, backfill
                    )

            if force:
                startDate = requestedStart
            elif latestDate.date() >= staleDate.date():
                print(f"  {ticker}: Up to date through {latestDate.date()}")
                result["skipped"] = 1
                return result
            else:
                print(
                    f"  {ticker}: Updating recent data from "
                    f"{startDate.date()} to {today.date()}"
                )

        if startDate is not None and startDate.date() >= today.date():
            print(f"  {ticker}: Already up to date")
            result["skipped"] = 1
            return result

        df = fetchPriceData(
            ticker,
            startDate=startDate,
            endDate=today,
        )
        if df is None or df.empty:
            print(f"  {ticker}: No data fetched")
            return result

        result["fetched"] += len(df)
        result["inserted"] += uploadToSupabase(client, ticker, df)
        print(
            f"  {ticker}: fetched={result['fetched']}, "
            f"inserted={result['inserted']}"
        )

    except Exception as exc:
        result["error"] = str(exc)
        print(f"  {ticker}: ERROR - {exc}")

    return result


def main() -> None:
    """Load maximum available history for every active ticker."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load maximum available historical stock data"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="Optional bounded history in years; default is maximum available yfinance history.",
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
    print(f"Found {len(tickers)} active tickers in stocks table
")

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

    print(f"
{'=' * 50}")
    print("SUMMARY:")
    print(f"  Total tickers processed: {len(tickers)}")
    print(f"  Total records fetched: {totalFetched}")
    print(f"  Total records inserted: {totalInserted}")
    print(f"  Errors: {len(errors)}")
    print(f"{'=' * 50}")

    if errors:
        print("
Failed tickers:")
        for error in errors:
            print(f"  {error['ticker']}: {error['error']}")


if __name__ == "__main__":
    main()
