"""Point-in-time S&P 500 membership universe.

The checked-in CSV is the source of truth for membership since 2000.
Membership intervals use a start-inclusive, end-exclusive convention.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

SP500_HISTORY_START = date(2000, 1, 1)
SP500_BENCHMARK_TICKER = "SPY"
_MEMBERSHIP_PATH = Path(__file__).with_name("sp500_membership.csv")

# Yahoo Finance uses hyphens for these two share-class symbols.
YFINANCE_TICKER_ALIASES = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}


@dataclass(frozen=True, order=True)
class MembershipPeriod:
    ticker: str
    start_date: date
    end_date_exclusive: date | None


def _as_date(value: date | str | pd.Timestamp) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value).date()


@lru_cache(maxsize=1)
def load_sp500_membership() -> tuple[MembershipPeriod, ...]:
    """Load and validate all point-in-time membership intervals."""
    if not _MEMBERSHIP_PATH.exists():
        raise FileNotFoundError(
            f"Missing S&P 500 membership file: {_MEMBERSHIP_PATH}"
        )

    periods: list[MembershipPeriod] = []
    with _MEMBERSHIP_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ticker", "start_date", "end_date_exclusive"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                "S&P 500 membership CSV must contain columns "
                "'ticker', 'start_date', and 'end_date_exclusive'"
            )

        for line_number, row in enumerate(reader, start=2):
            ticker = (row.get("ticker") or "").strip().upper()
            if not ticker:
                raise ValueError(f"Empty ticker in membership CSV line {line_number}")
            start = _as_date(row["start_date"])
            raw_end = (row.get("end_date_exclusive") or "").strip()
            end = _as_date(raw_end) if raw_end else None

            if start < SP500_HISTORY_START:
                raise ValueError(
                    f"{ticker}: membership starts before 2000 on line {line_number}"
                )
            if end is not None and end <= start:
                raise ValueError(
                    f"{ticker}: invalid membership interval on line {line_number}"
                )
            periods.append(MembershipPeriod(ticker, start, end))

    periods.sort(key=lambda item: (item.ticker, item.start_date))
    by_ticker: dict[str, list[MembershipPeriod]] = {}
    for period in periods:
        previous = by_ticker.setdefault(period.ticker, [])
        if previous:
            prior = previous[-1]
            if prior.end_date_exclusive is None:
                raise ValueError(f"{period.ticker}: overlapping open membership intervals")
            if period.start_date < prior.end_date_exclusive:
                raise ValueError(f"{period.ticker}: overlapping membership intervals")
        previous.append(period)

    return tuple(periods)


def get_sp500_tickers() -> list[str]:
    """Return every distinct ticker that was an S&P 500 constituent since 2000."""
    return sorted({period.ticker for period in load_sp500_membership()})


def get_current_sp500_tickers(as_of: date | str | pd.Timestamp | None = None) -> list[str]:
    """Return S&P 500 constituents for one date (today when omitted)."""
    day = _as_date(as_of or date.today())
    current = {
        period.ticker
        for period in load_sp500_membership()
        if period.start_date <= day
        and (period.end_date_exclusive is None or day < period.end_date_exclusive)
    }
    return sorted(current)


def get_sp500_membership_periods(ticker: str) -> tuple[MembershipPeriod, ...]:
    """Return all membership intervals for a ticker, including re-entries."""
    normalized = str(ticker).strip().upper()
    return tuple(
        period for period in load_sp500_membership()
        if period.ticker == normalized
    )


def is_sp500_member(
    ticker: str,
    day: date | str | pd.Timestamp,
) -> bool:
    """Return whether a ticker was an index constituent on a given date."""
    target = _as_date(day)
    return any(
        period.start_date <= target
        and (period.end_date_exclusive is None or target < period.end_date_exclusive)
        for period in get_sp500_membership_periods(ticker)
    )


def filter_sp500_price_rows(
    frame: pd.DataFrame,
    ticker: str,
    date_column: str = "price_date",
) -> pd.DataFrame:
    """Keep only price rows whose dates fall inside a ticker's S&P membership."""
    if date_column not in frame.columns:
        raise ValueError(f"Missing date column '{date_column}'")
    periods = get_sp500_membership_periods(ticker)
    if not periods:
        return frame.iloc[0:0].copy()

    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.date
    keep = pd.Series(False, index=frame.index)
    for period in periods:
        mask = dates >= period.start_date
        if period.end_date_exclusive is not None:
            mask &= dates < period.end_date_exclusive
        keep |= mask.fillna(False)
    return frame.loc[keep].copy()


def get_yfinance_symbol(ticker: str) -> str:
    """Translate a canonical S&P ticker to the Yahoo Finance symbol."""
    normalized = str(ticker).strip().upper()
    return YFINANCE_TICKER_ALIASES.get(normalized, normalized)


__all__ = [
    "MembershipPeriod",
    "SP500_BENCHMARK_TICKER",
    "SP500_HISTORY_START",
    "YFINANCE_TICKER_ALIASES",
    "filter_sp500_price_rows",
    "get_current_sp500_tickers",
    "get_sp500_membership_periods",
    "get_sp500_tickers",
    "get_yfinance_symbol",
    "is_sp500_member",
    "load_sp500_membership",
]
