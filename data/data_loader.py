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

from data.excluded_tickers import is_excluded_ticker, normalize_ticker
from data.price_validation import validate_ohlcv_frame

load_dotenv(".env.secrets")


class PriceUploadError(RuntimeError):
    """Raised when one or more price rows could not be persisted."""


def _is_duplicate_key_error(error: Exception) -> bool:
    """Recognize PostgreSQL unique-key conflicts without hiding other failures."""
    if getattr(error, "code", None) == "23505":
        return True
    response = getattr(error, "details", None)
    message = " ".join(
        str(value) for value in (error, response, getattr(error, "message", None))
        if value is not None
    )
    return "23505" in message or "duplicate key value violates unique constraint" in message.lower()


def getSupabaseClient() -> supabase.Client:
    """Initialize and return Supabase client."""
    supabaseUrl = os.getenv("SUPABASE_URL")
    supabaseKey = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not supabaseUrl or not supabaseKey:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY (or SUPABASE_ANON_KEY) must be set in .env.secrets")

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
        normalize_ticker(row["ticker"])
        for row in (response.data or [])
        if row.get("ticker") and not is_excluded_ticker(row["ticker"])
    ]


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
            # Request the total-return-adjusted OHLC basis explicitly. Older
            # rows were ingested using yfinance's version-dependent default;
            # in this project's current environment that default adjusted OHLC.
            # Keeping this explicit avoids storing a raw-price tail in an
            # otherwise adjusted historical series.
            df = stock.history(
                start=startDate,
                end=endDate,
                auto_adjust=True,
                actions=False,
            )

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
            missing = sorted(set(keepCols) - set(df.columns))
            if missing:
                raise ValueError(
                    f"{ticker}: provider response is missing required columns: "
                    f"{', '.join(missing)}"
                )

            df = df[keepCols].copy()
            df["ticker"] = ticker
            df["price_date"] = df.index.strftime("%Y-%m-%d")

            # yfinance's end parameter is exclusive. Reject unexpected dates
            # rather than allowing provider boundary behavior to create stale
            # or out-of-window records in the database.
            start_day = startDate.date().isoformat()
            end_day = endDate.date().isoformat()
            outside = ~df["price_date"].between(start_day, end_day, inclusive="left")
            if outside.any():
                bad_dates = df.loc[outside, "price_date"].astype(str).tolist()[:5]
                raise ValueError(
                    f"{ticker}: provider returned date(s) outside requested "
                    f"range [{start_day}, {end_day}): {bad_dates}"
                )

            df = df.reset_index(drop=True)

            # With auto_adjust=True, OHLC already includes corporate-action
            # adjustments. Keep adj_close on that same basis.
            df["adj_close"] = df["close"]

            return validate_ohlcv_frame(df, ticker)

        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            raise RuntimeError(
                f"{ticker}: failed to fetch/validate provider prices after "
                f"{retries} attempt(s): {exc}"
            ) from exc

    return None


def uploadToSupabase(
    client: supabase.Client,
    ticker: str,
    df: pd.DataFrame,
    batchSize: int = 500,
) -> int:
    """Upload price data to Supabase."""
    ticker = normalize_ticker(ticker)
    if is_excluded_ticker(ticker):
        print(f"  {ticker}: refusing to upload excluded price history")
        return 0

    df = validate_ohlcv_frame(df, ticker)
    if df is None or df.empty:
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
    duplicates = 0
    failures: list[str] = []
    for i in range(0, len(records), batchSize):
        batch = records[i:i + batchSize]
        try:
            client.table("stock_prices").insert(batch).execute()
            inserted += len(batch)
            continue
        except Exception as exc:
            # Retry individually to isolate unique-key conflicts from genuine
            # persistence errors; a failed batch is not itself proof every row
            # failed, because some rows may already exist.
            print(f"    Batch insert for {ticker} failed; retrying rows individually: {exc}")

        for record in batch:
            try:
                client.table("stock_prices").insert(record).execute()
                inserted += 1
            except Exception as exc:
                if _is_duplicate_key_error(exc):
                    duplicates += 1
                else:
                    failures.append(f"{record['price_date']}: {exc}")

    if duplicates:
        print(f"    {ticker}: skipped {duplicates} row(s) already present (unique-key conflicts)")
    if failures:
        sample = "; ".join(failures[:5])
        remainder = len(failures) - min(len(failures), 5)
        suffix = f"; and {remainder} more" if remainder else ""
        raise PriceUploadError(
            f"{ticker}: failed to insert {len(failures)} row(s); {sample}{suffix}"
        )

    return inserted


def _build_daily_ingestion_window(
    now: datetime, latest_date: datetime | None
) -> tuple[datetime, datetime]:
    """Return naive exchange-local date boundaries for completed daily bars.

    yfinance interprets naive datetimes in each ticker's exchange timezone.
    Passing UTC-aware midnight can shift the start date back to the prior
    local calendar day and make an otherwise valid row fail boundary checks.

    The regular workflow runs at 22:30 UTC, after cash-market close in the
    supported markets. Before 22:00 UTC, today's date stays exclusive so a
    partial daily candle is not ingested. At or after 22:00 UTC, tomorrow's
    date is used as the exclusive end boundary so today's completed candle
    can be included.
    """
    now_utc = now.astimezone(timezone.utc) if now.tzinfo else now.replace(tzinfo=timezone.utc)
    today = now_utc.date()
    end_day = today + timedelta(days=1) if now_utc.hour >= 22 else today

    if latest_date is None:
        target_year = today.year - 5
        try:
            start_day = today.replace(year=target_year)
        except ValueError:
            # February 29 does not exist in every target year.
            start_day = today.replace(year=target_year, day=28)
    else:
        start_day = latest_date.date() + timedelta(days=1)

    return (
        datetime.combine(start_day, datetime.min.time()),
        datetime.combine(end_day, datetime.min.time()),
    )


def processTicker(client: supabase.Client, ticker: str) -> dict[str, Any]:
    """Process one ticker unless it is on the permanent invalid-data blocklist."""
    ticker = normalize_ticker(ticker)
    result = {
        "ticker": ticker,
        "fetched": 0,
        "inserted": 0,
        "skipped": 0,
        "error": None,
    }

    if is_excluded_ticker(ticker):
        print(f"  {ticker}: excluded due to known invalid price history")
        result["skipped"] = 1
        return result

    try:
        latestDate = getLatestPriceDate(client, ticker)
        now = datetime.now(timezone.utc)
        startDate, endDate = _build_daily_ingestion_window(now, latestDate)

        if latestDate is None:
            print(
                f"  {ticker}: No existing data, fetching from {startDate.date()} "
                f"to {endDate.date()} (end exclusive)"
            )
        else:
            print(
                f"  {ticker}: Last date={latestDate.date()}, fetching from "
                f"{startDate.date()} to {endDate.date()} (end exclusive)"
            )

        if startDate.date() >= endDate.date():
            print(f"  {ticker}: Already up to date (no completed daily bars in range)")
            result["skipped"] = 1
            return result

        df = fetchPriceData(ticker, startDate, endDate)
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
