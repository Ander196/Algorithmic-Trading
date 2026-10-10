"""Shared data-access helpers for scheduled jobs."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
from supabase import Client, create_client


def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY/SUPABASE_ANON_KEY must be configured")
    return create_client(url, key)


def fetch_price_history(
    client: Client,
    ticker: str,
    limit: int | None = 1500,
) -> pd.DataFrame:
    """Read OHLCV history using Supabase pagination.

    By default, fetch the most recent 1500 rows for compatibility with other
    jobs. Pass ``limit=None`` to fetch every available row. An explicit positive
    limit caps the result to the most recent observations. Results are returned
    chronologically for rolling features and models.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")

    rows: list[dict] = []
    page_size = 1000
    offset = 0

    while limit is None or len(rows) < limit:
        remaining = page_size if limit is None else min(page_size, limit - len(rows))
        if remaining <= 0:
            break

        response = (
            client.table("stock_prices")
            .select("price_date,open,high,low,close,volume")
            .eq("ticker", ticker.upper())
            .order("price_date", desc=True)
            .range(offset, offset + remaining - 1)
            .execute()
        )
        batch = response.data or []
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < remaining:
            break

    if not rows:
        raise RuntimeError(f"No price history found for {ticker}")

    df = pd.DataFrame(rows)
    df["price_date"] = pd.to_datetime(df["price_date"], utc=True)
    df = df.sort_values("price_date").drop_duplicates("price_date", keep="last")
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    if limit is not None:
        df = df.tail(limit)
    return df.reset_index(drop=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
