"""
data_loader.py

Incremental daily ingestion script for OHLCV data.
- Fetches active tickers from `stocks` table
- Detects the last date with data in `stock_prices` per ticker
- Fetches and inserts data from (last_date + 1) to today

Usage:
    python data_loader.py

Environment variables required in .env.secrets:
    SUPABASE_URL=https://your-project.supabase.co
    SUPABASE_ANON_KEY=your-anon-key
"""

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
    supabaseKey = os.getenv("SUPABASE_ANON_KEY")

    if not supabaseUrl or not supabaseKey:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env.secrets")

    return supabase.create_client(supabaseUrl, supabaseKey)


def getActiveTickers(client: supabase.Client) -> list[str]:
    """Fetch all active tickers from stocks table."""
    response = (
        client.table("stocks")
        .select("ticker")
        .eq("is_active", True)
        .execute()
    )
    return [row["ticker"] for row in response.data]


def getLatestPriceDate(client: supabase.Client, ticker: str) -> datetime | None:
    """Get the latest price_date existing in stock_prices for a given ticker."""
    response = (
        client.table("stock_prices")
        .select("price_date")
        .eq("ticker", ticker)
        .order("price_date", desc=True)
        .limit(1)
        .execute()
    )
    if not response.data:
        return None
    latestStr = response.data[0]["price_date"]
    return datetime.strptime(latestStr, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def fetchPriceData(
    ticker: str, startDate: datetime, endDate: datetime, retries: int = 2
) -> pd.DataFrame | None:
    """Fetch OHLCV data from yfinance for a date range."""
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=startDate, end=endDate)

            if df.empty:
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Dividends": "dividends",
                "Stock Splits": "stock_splits",
            })

            keepCols = ["open", "high", "low", "close", "volume"]
            for col in keepCols:
                if col not in df.columns:
                    df[col] = 0

            df = df[keepCols].copy()
            df["ticker"] = ticker
            df["price_date"] = df.index.strftime("%Y-%m-%d")
            df = df.reset_index(drop=True)

            try:
                adj_close = stock.info.get("adj_close") or stock.info.get("adjClose")
                df["adj_close"] = adj_close if adj_close is not None else df["close"]
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
    """Upload price data to Supabase."""
    if df.empty:
        return 0

    records = []
    now = datetime.now(timezone.utc).isoformat()

    for _, row in df.iterrows():
        records.append({
            "ticker": ticker,
            "price_date": row["price_date"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "adj_close": float(row["adj_close"]) if pd.notna(row["adj_close"]) else float(row["close"]),
            "volume": int(row["volume"]),
            "inserted_at": now,
        })

    inserted = 0
    for i in range(0, len(records), batchSize):
        batch = records[i:i + batchSize]
        try:
            client.table("stock_prices").insert(batch).execute()
            inserted += len(batch)
        except Exception as e:
            print(f"    Error inserting batch for {ticker}: {e}")
            for record in batch:
                try:
                    client.table("stock_prices").insert(record).execute()
                    inserted += 1
                except Exception:
                    pass

    return inserted


def processTicker(client: supabase.Client, ticker: str) -> dict[str, Any]:
    """Process a single ticker: fetch incremental data and upload to Supabase."""
    result = {
        "ticker": ticker,
        "fetched": 0,
        "inserted": 0,
        "skipped": 0,
        "error": None,
    }

    try:
        latestDate = getLatestPriceDate(client, ticker)
        today = datetime.now(timezone.utc)

        if latestDate is None:
            startDate = today.replace(year=today.year - 5)
            print(f"  {ticker}: No existing data, fetching from {startDate.date()} to {today.date()}")
        else:
            startDate = latestDate + timedelta(days=1)
            print(f"  {ticker}: Last date={latestDate.date()}, fetching from {startDate.date()} to {today.date()}")

        if startDate.date() >= today.date():
            print(f"  {ticker}: Already up to date")
            result["skipped"] = 1
            return result

        df = fetchPriceData(ticker, startDate, today)
        if df is None or df.empty:
            print(f"  {ticker}: No new data fetched")
            return result

        result["fetched"] = len(df)
        inserted = uploadToSupabase(client, ticker, df)
        result["inserted"] = inserted
        print(f"  {ticker}: fetched={result['fetched']}, inserted={inserted}")

    except Exception as e:
        result["error"] = str(e)
        print(f"  {ticker}: ERROR - {e}")

    return result


def main() -> None:
    """Main entry point."""
    client = getSupabaseClient()
    tickers = getActiveTickers(client)
    print(f"Found {len(tickers)} active tickers in stocks table\n")

    totalInserted = 0
    totalFetched = 0
    errors = []

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Processing {ticker}...")
        result = processTicker(client, ticker)

        totalFetched += result["fetched"]
        totalInserted += result["inserted"]

        if result["error"]:
            errors.append(result)

        time.sleep(0.3)

    print(f"\n{'=' * 50}")
    print(f"SUMMARY:")
    print(f"  Total tickers processed: {len(tickers)}")
    print(f"  Total records fetched: {totalFetched}")
    print(f"  Total records inserted: {totalInserted}")
    print(f"  Errors: {len(errors)}")
    print(f"{'=' * 50}")

    if errors:
        print("\nFailed tickers:")
        for e in errors:
            print(f"  {e['ticker']}: {e['error']}")


if __name__ == "__main__":
    main()