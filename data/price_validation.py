"""Validation helpers for daily stock OHLCV bars."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")


def validate_ohlcv_frame(
    data: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame | None:
    """Return only valid OHLCV rows, or None if the frame cannot be used.

    Required columns are never fabricated. Price fields must be finite and
    positive, volume must be finite and non-negative, and intraday OHLC ranges
    must be internally consistent. Invalid rows are logged and excluded.
    """
    if data is None or data.empty:
        return None

    missing = sorted(set(REQUIRED_OHLCV_COLUMNS) - set(data.columns))
    if missing:
        print(
            f"  {ticker}: rejecting OHLCV data; missing required columns: "
            f"{', '.join(missing)}"
        )
        return None

    frame = data.copy()
    for column in REQUIRED_OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    prices = frame.loc[:, list(PRICE_COLUMNS)]
    valid = prices.notna().all(axis=1)
    valid &= np.isfinite(prices).all(axis=1)
    valid &= (prices > 0).all(axis=1)

    volume = frame["volume"]
    valid &= volume.notna() & np.isfinite(volume) & (volume >= 0)

    # Adj Close is optional in the provider response, but if supplied it must
    # be either missing (uploader falls back to Close) or a finite positive price.
    if "adj_close" in frame.columns:
        adj_close = pd.to_numeric(frame["adj_close"], errors="coerce")
        valid &= adj_close.isna() | (
            np.isfinite(adj_close) & (adj_close > 0)
        )

    valid &= frame["high"] >= frame[["open", "low", "close"]].max(axis=1)
    valid &= frame["low"] <= frame[["open", "high", "close"]].min(axis=1)

    invalid_count = int((~valid).sum())
    if invalid_count:
        print(
            f"  {ticker}: excluding {invalid_count} invalid OHLCV "
            f"row(s) before upload"
        )

    frame = frame.loc[valid].copy()
    if frame.empty:
        print(f"  {ticker}: no valid OHLCV rows remain; nothing will be uploaded")
        return None

    return frame
