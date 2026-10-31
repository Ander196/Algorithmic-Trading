"""Transient pre-membership prices used only to warm up live risk estimates.

These observations are fetched from Yahoo Finance and kept in memory. They are
never inserted into stock_prices and must not be used by the point-in-time
Alpha dataset or historical portfolio backtests.
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from data.price_validation import validate_ohlcv_frame
from data.sp500_universe import get_yfinance_symbol

logger = logging.getLogger(__name__)

DEFAULT_PRE_MEMBERSHIP_LOOKBACK_DAYS = 180


def fetch_pre_membership_price_history(
    ticker: str,
    membership_start: date | str,
    *,
    lookback_days: int = DEFAULT_PRE_MEMBERSHIP_LOOKBACK_DAYS,
    retries: int = 2,
) -> pd.DataFrame:
    """Fetch adjusted OHLCV strictly before the current S&P membership start.

    The returned price_date values are UTC-normalized trading dates, matching
    the representation returned by jobs.common.fetch_price_history.
    auto_adjust=True mirrors the adjusted OHLC basis persisted by ingestion.
    The caller may use this frame for an in-memory Layer 2 warm-up only.
    """
    ticker = str(ticker).strip().upper()
    if not ticker:
        raise ValueError("ticker must be non-empty")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be greater than zero")
    if retries <= 0:
        raise ValueError("retries must be greater than zero")

    start_day = pd.Timestamp(membership_start).date()
    end_day = start_day
    request_start_day = start_day - timedelta(days=lookback_days)
    request_start = pd.Timestamp(request_start_day, tz="UTC").to_pydatetime()
    request_end = pd.Timestamp(end_day, tz="UTC").to_pydatetime()

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            history = yf.Ticker(get_yfinance_symbol(ticker)).history(
                start=request_start,
                end=request_end,
                auto_adjust=True,
                actions=False,
            )
            if history is None or history.empty:
                raise RuntimeError(
                    f"Yahoo Finance returned no pre-membership history for {ticker} "
                    f"before {end_day.isoformat()}"
                )

            if isinstance(history.columns, pd.MultiIndex):
                history.columns = history.columns.get_level_values(0)

            renamed = history.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            required = ["open", "high", "low", "close", "volume"]
            missing = sorted(set(required) - set(renamed.columns))
            if missing:
                raise ValueError(
                    f"{ticker}: Yahoo Finance response missing required columns: "
                    f"{', '.join(missing)}"
                )

            frame = renamed[required].copy()
            # Preserve the exchange-local session date before normalizing to UTC.
            frame["price_date"] = pd.to_datetime(
                history.index.strftime("%Y-%m-%d"),
                utc=True,
            )
            frame = frame.reset_index(drop=True)

            # Yahoo's end parameter is exclusive. Enforce the boundary locally
            # too, so a provider boundary quirk can never leak post-entry prices
            # into the auxiliary pre-membership segment.
            dates = pd.to_datetime(frame["price_date"], utc=True)
            start_utc = pd.Timestamp(request_start_day, tz="UTC")
            end_utc = pd.Timestamp(end_day, tz="UTC")
            frame = frame.loc[(dates >= start_utc) & (dates < end_utc)].copy()
            frame = frame.sort_values("price_date").drop_duplicates(
                "price_date", keep="last"
            )

            validated = validate_ohlcv_frame(
                frame,
                ticker,
                context="Yahoo pre-membership risk warm-up",
            )
            if validated is None or validated.empty:
                raise RuntimeError(
                    f"{ticker}: no valid pre-membership OHLCV rows before "
                    f"{end_day.isoformat()}"
                )

            result = validated[
                ["price_date", "open", "high", "low", "close", "volume"]
            ].reset_index(drop=True)
            logger.info(
                "%s: fetched %d transient pre-membership risk bars (%s to %s); "
                "not persisted",
                ticker,
                len(result),
                result["price_date"].min().date(),
                result["price_date"].max().date(),
            )
            return result

        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                logger.warning(
                    "%s: pre-membership risk-history request failed "
                    "(attempt %d/%d): %s",
                    ticker,
                    attempt + 1,
                    retries,
                    exc,
                )
                time.sleep(attempt + 1)

    raise RuntimeError(
        f"{ticker}: failed to fetch pre-membership risk history after "
        f"{retries} attempt(s): {last_error}"
    ) from last_error


__all__ = [
    "DEFAULT_PRE_MEMBERSHIP_LOOKBACK_DAYS",
    "fetch_pre_membership_price_history",
]
