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
    limit: int = 1500,
) -> pd.DataFrame:
    """Read the most recent OHLCV history, handling Supabase pagination.

    The query is ordered newest-first so the requested ``limit`` contains the
    latest observations. The returned DataFrame is then sorted chronologically
    because downstream rolling features and stateful models expect time order.
    """
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    rows: list[dict] = []
    page_size = 1000
    offset = 0

    while len(rows) < limit:
        end = min(offset + page_size, limit) - 1
        response = (
            client.table("stock_prices")
            .select("price_date,open,high,low,close,volume")
            .eq("ticker", ticker.upper())
            .order("price_date", desc=True)
            .range(offset, end)
            .execute()
        )
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    if not rows:
        raise RuntimeError(f"No price history found for {ticker}")

    df = pd.DataFrame(rows)
    df["price_date"] = pd.to_datetime(df["price_date"], utc=True)
    df = df.sort_values("price_date").drop_duplicates("price_date", keep="last")
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df.tail(limit).reset_index(drop=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
