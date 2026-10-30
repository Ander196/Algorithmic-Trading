"""Build the stock catalog from all historical S&P 500 constituents.

The S&P 500 point-in-time CSV is the source of truth. Current constituents are
active, former constituents remain registered but inactive so historical OHLCV
can be stored under the stocks foreign key. SPY is retained as the benchmark.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pandas as pd
import supabase
import yfinance as yf
from dotenv import load_dotenv

from data.retired_tickers import RETIRED_TICKERS
from data.sp500_universe import (
    SP500_BENCHMARK_TICKER,
    get_current_sp500_tickers,
    get_sp500_tickers,
    get_yfinance_symbol,
)

load_dotenv(".env.secrets")


def _fallback_metadata(ticker: str) -> dict:
    """Return safe metadata when Yahoo quote metadata is unavailable."""
    curated = RETIRED_TICKERS.get(ticker, {})
    return {
        "ticker": ticker,
        "name": curated.get("name") or ticker,
        "exchange": curated.get("exchange") or "Unknown",
        "sector": curated.get("sector") or "Unknown",
        "industry": curated.get("industry") or "Unknown",
    }


def fetchTickerInfo(ticker: str, retries: int = 2) -> dict:
    """Fetch descriptive metadata, falling back without dropping constituents."""
    ticker = str(ticker).strip().upper()
    fallback = _fallback_metadata(ticker)
    for attempt in range(max(1, retries)):
        try:
            info = yf.Ticker(get_yfinance_symbol(ticker)).info or {}
            return {
                "ticker": ticker,
                "name": info.get("shortName") or info.get("longName") or fallback["name"],
                "exchange": info.get("exchange") or fallback["exchange"],
                "sector": info.get("sector") or fallback["sector"],
                "industry": info.get("industry") or fallback["industry"],
            }
        except Exception:
            if attempt < retries - 1:
                time.sleep(attempt + 1)
    # A provider metadata outage must never remove a valid point-in-time member.
    return fallback


def fetchAllTickers(
    tickers: list[str],
    batchSize: int = 50,
    delay: float = 0.3,
) -> pd.DataFrame:
    """Fetch current-constituent metadata, retaining all symbols on failure."""
    normalized = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    results = []
    total = len(normalized)
    print(f"Fetching metadata for {total} current S&P 500/benchmark tickers...")

    for i in range(0, total, batchSize):
        batch = normalized[i:i + batchSize]
        for ticker in batch:
            results.append(fetchTickerInfo(ticker))
            if delay:
                time.sleep(delay)
        progress = min(i + batchSize, total)
        print(f"  Progress: {progress}/{total} ({100 * progress / total:.1f}%)")

    return pd.DataFrame(results)


def buildTickerCatalog() -> pd.DataFrame:
    """Build the full catalog of historical S&P members plus the SPY benchmark."""
    current = set(get_current_sp500_tickers())
    all_members = set(get_sp500_tickers())
    current_with_benchmark = sorted(current | {SP500_BENCHMARK_TICKER})
    former = sorted(all_members - current)

    current_df = fetchAllTickers(current_with_benchmark)
    current_df["historical_only"] = False

    former_records = []
    for ticker in former:
        former_records.append({
            **_fallback_metadata(ticker),
            "historical_only": True,
        })

    former_df = pd.DataFrame(former_records)
    combined = pd.concat([current_df, former_df], ignore_index=True)
    combined["ticker"] = combined["ticker"].astype(str).str.strip().str.upper()

    if combined["ticker"].duplicated().any():
        duplicates = sorted(combined.loc[
            combined["ticker"].duplicated(keep=False), "ticker"
        ].unique())
        raise ValueError(f"Duplicate tickers in S&P 500 catalog: {duplicates}")

    expected = all_members | {SP500_BENCHMARK_TICKER}
    observed = set(combined["ticker"])
    if observed != expected:
        raise ValueError(
            "S&P 500 catalog mismatch; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    return combined


def getExistingTickers(client: supabase.Client) -> set[str]:
    """Get all tickers already present in the stocks table."""
    response = client.table("stocks").select("ticker").execute()
    return {
        str(row["ticker"]).strip().upper()
        for row in (response.data or [])
        if row.get("ticker")
    }


def getNowTimestamptz() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def uploadToSupabase(
    df: pd.DataFrame,
    client: supabase.Client,
) -> tuple[int, int]:
    """Upload current and historical-only catalog rows without resetting history."""
    required = {"ticker", "name", "exchange", "sector", "industry", "historical_only"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ticker catalog missing columns: {sorted(missing)}")

    existing = getExistingTickers(client)
    now = getNowTimestamptz()
    current_records = []
    historical_existing = []
    historical_new = []
    seen = set()

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        if ticker in seen:
            raise ValueError(f"Duplicate ticker in upload frame: {ticker}")
        seen.add(ticker)

        historical_only = bool(row["historical_only"])
        record = {
            "ticker": ticker,
            "name": row["name"] or ticker,
            "exchange": row["exchange"] or "Unknown",
            "sector": row["sector"] or "Unknown",
            "industry": row["industry"] or "Unknown",
            "is_active": not historical_only,
        }

        if historical_only:
            if ticker in existing:
                historical_existing.append(record)
            else:
                historical_new.append(record)
        else:
            # Preserve the established current-catalog refresh semantics.
            record["created_at"] = now
            current_records.append(record)

    if current_records:
        client.table("stocks").upsert(
            current_records, on_conflict="ticker"
        ).execute()
        print(f"  Upserted {len(current_records)} current constituents/benchmark")

    for record in historical_existing:
        client.table("stocks").update(record).eq(
            "ticker", record["ticker"]
        ).execute()
    if historical_existing:
        print(f"  Updated {len(historical_existing)} former constituents")

    if historical_new:
        client.table("stocks").insert(historical_new).execute()
        print(f"  Inserted {len(historical_new)} former constituents")

    # Keep non-S&P and non-benchmark rows for auditability, but mark them inactive.
    missing_tickers = existing - seen
    for ticker in sorted(missing_tickers):
        client.table("stocks").update({"is_active": False}).eq(
            "ticker", ticker
        ).execute()

    print(f"  Deactivated {len(missing_tickers)} tickers outside the configured universe")
    return len(current_records) + len(historical_new), len(missing_tickers)


def getSupabaseClient() -> supabase.Client:
    """Initialize Supabase with the configured privileged project key."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env.secrets")
    return supabase.create_client(supabase_url, supabase_key)


def main() -> None:
    """Refresh the current S&P catalog and register all former constituents."""
    client = getSupabaseClient()
    catalog = buildTickerCatalog()
    current_count = int((~catalog["historical_only"]).sum())
    historical_count = int(catalog["historical_only"].sum())

    print(
        f"Prepared {current_count} current constituents/benchmark and "
        f"{historical_count} former S&P 500 constituents "
        f"({catalog['ticker'].nunique()} unique tickers total)"
    )
    new_count, deactivated_count = uploadToSupabase(catalog, client)
    print(f"\nDone! New: {new_count}, Deactivated: {deactivated_count}")


if __name__ == "__main__":
    main()
