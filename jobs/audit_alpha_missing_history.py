"""Read-only Yahoo coverage probe for historical S&P 500 tickers missing in Supabase.

This audit accepts the failed-ticker CSV emitted from an Alpha evaluation log.
It queries Yahoo only inside the ticker's recorded S&P 500 membership windows,
and optionally tests a dot-to-hyphen share-class spelling. It never writes to
Supabase, emits repair SQL, or treats provider availability as identity proof.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import yfinance as yf

from data.retired_tickers import RETIRED_TICKERS
from data.sp500_universe import (
    get_sp500_membership_periods,
    get_sp500_price_periods,
    get_yfinance_symbol,
)


HISTORY_COLUMNS = {"Close", "Adj Close"}
AUDIT_COLUMNS = (
    "ticker",
    "membership_period_count",
    "membership_windows",
    "last_membership_end_exclusive",
    "curated_last_trade_date",
    "ticker_shape_review",
    "primary_yahoo_symbol",
    "primary_sessions",
    "primary_first_date",
    "primary_last_date",
    "primary_requests",
    "primary_errors",
    "primary_dividend_events",
    "primary_split_events",
    "share_class_alias_candidate",
    "alias_sessions",
    "alias_first_date",
    "alias_last_date",
    "alias_requests",
    "alias_errors",
    "alias_dividend_events",
    "alias_split_events",
    "status",
    "identity_warning",
)


@dataclass(frozen=True)
class YahooProbe:
    """Coverage-only result for one Yahoo symbol across membership windows."""

    symbol: str
    sessions: int
    first_date: str | None
    last_date: str | None
    requests: int
    errors: int
    dividend_events: int
    split_events: int


def _summarize_yahoo_history(history: pd.DataFrame) -> tuple[int, str | None, str | None, int, int]:
    """Count valid price observations and corporate-action rows, without prices."""
    if history is None or history.empty:
        return 0, None, None, 0, 0

    frame = history.copy()
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    if index.tz is not None:
        # Yahoo's index is exchange-local; strip timezone without shifting dates.
        index = index.tz_localize(None)
    frame.index = index.normalize()
    if frame.index.has_duplicates:
        frame = frame.loc[~frame.index.duplicated(keep="last")]

    price_column = "Adj Close" if "Adj Close" in frame.columns else "Close"
    if price_column not in frame.columns:
        raise ValueError(
            f"Yahoo history has neither 'Adj Close' nor 'Close': {list(frame.columns)}"
        )
    prices = pd.to_numeric(frame[price_column], errors="coerce")
    valid = prices.notna() & (prices > 0) & prices.map(lambda value: pd.notna(value) and value != float("inf"))
    valid_dates = frame.index[valid.to_numpy()]
    dividends = (
        pd.to_numeric(frame["Dividends"], errors="coerce").fillna(0.0)
        if "Dividends" in frame.columns else pd.Series(0.0, index=frame.index)
    )
    splits = (
        pd.to_numeric(frame["Stock Splits"], errors="coerce").fillna(0.0)
        if "Stock Splits" in frame.columns else pd.Series(0.0, index=frame.index)
    )
    dividend_events = int((dividends != 0).sum())
    split_events = int((splits != 0).sum())
    if not len(valid_dates):
        return 0, None, None, dividend_events, split_events
    return (
        int(len(valid_dates)),
        valid_dates.min().date().isoformat(),
        valid_dates.max().date().isoformat(),
        dividend_events,
        split_events,
    )


def _probe_yahoo_symbol(
    symbol: str,
    windows,
    *,
    history_fetcher: Callable[[str, str, str], pd.DataFrame],
    retries: int,
    delay_seconds: float,
    sleeper: Callable[[float], None],
) -> YahooProbe:
    sessions = 0
    dates: list[pd.Timestamp] = []
    errors = 0
    requests = 0
    dividend_events = 0
    split_events = 0

    for window_index, window in enumerate(windows):
        start = window.start_date.isoformat()
        if window.end_date_exclusive is None:
            raise ValueError(
                f"{symbol}: an open-ended membership window cannot be bounded for this audit"
            )
        end = window.end_date_exclusive.isoformat()

        history = pd.DataFrame()
        for attempt in range(retries):
            try:
                history = history_fetcher(symbol, start, end)
                requests += 1
                break
            except Exception:
                errors += 1
                if attempt + 1 < retries:
                    sleeper(delay_seconds * (attempt + 1))

        count, first_date, last_date, dividends, splits = _summarize_yahoo_history(history)
        sessions += count
        dividend_events += dividends
        split_events += splits
        if first_date:
            dates.append(pd.Timestamp(first_date))
        if last_date:
            dates.append(pd.Timestamp(last_date))

        if delay_seconds > 0 and window_index + 1 < len(windows):
            sleeper(delay_seconds)

    return YahooProbe(
        symbol=symbol,
        sessions=sessions,
        first_date=min(dates).date().isoformat() if dates else None,
        last_date=max(dates).date().isoformat() if dates else None,
        requests=requests,
        errors=errors,
        dividend_events=dividend_events,
        split_events=split_events,
    )


def _ticker_shape_review(ticker: str) -> str:
    """Return weak triage hints; this is not an issuer classification."""
    hints: list[str] = []
    if ticker.endswith("Q") and len(ticker) > 2:
        hints.append("Q_suffix_may_be_OTC_or_distress_related")
    if "." in ticker:
        hints.append("dotted_share_class_symbol_check_dot_to_hyphen")
    if len(ticker) == 1:
        hints.append("single_character_symbol_requires_manual_identity_check")
    return ";".join(hints) if hints else "no_simple_symbol_shape_hint"


def audit_missing_ticker(
    ticker: str,
    *,
    history_fetcher: Callable[[str, str, str], pd.DataFrame] | None = None,
    retries: int = 2,
    delay_seconds: float = 0.15,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Probe the original and optional share-class spelling for one failed ticker."""
    normalized = str(ticker).strip().upper()
    if not normalized:
        raise ValueError("ticker must not be empty")
    if retries < 1:
        raise ValueError("retries must be >= 1")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be >= 0")

    membership = get_sp500_membership_periods(normalized)
    windows = get_sp500_price_periods(normalized)
    if not membership:
        return {
            "ticker": normalized,
            "membership_period_count": 0,
            "membership_windows": "",
            "last_membership_end_exclusive": None,
            "curated_last_trade_date": None,
            "ticker_shape_review": _ticker_shape_review(normalized),
            "primary_yahoo_symbol": get_yfinance_symbol(normalized),
            "primary_sessions": 0,
            "primary_first_date": None,
            "primary_last_date": None,
            "primary_requests": 0,
            "primary_errors": 0,
            "primary_dividend_events": 0,
            "primary_split_events": 0,
            "share_class_alias_candidate": None,
            "alias_sessions": 0,
            "alias_first_date": None,
            "alias_last_date": None,
            "alias_requests": 0,
            "alias_errors": 0,
            "alias_dividend_events": 0,
            "alias_split_events": 0,
            "status": "not_in_sp500_membership_file",
            "identity_warning": "No membership interval; inspect the input list and universe source.",
        }

    if any(window.end_date_exclusive is None for window in windows):
        raise ValueError(
            f"{normalized}: open-ended membership interval; audit window needs an explicit end"
        )

    fetcher = history_fetcher or (
        lambda symbol, start, end: yf.Ticker(symbol).history(
            start=start,
            end=end,
            auto_adjust=False,
            actions=True,
        )
    )
    primary_symbol = get_yfinance_symbol(normalized)
    primary = _probe_yahoo_symbol(
        primary_symbol,
        windows,
        history_fetcher=fetcher,
        retries=retries,
        delay_seconds=delay_seconds,
        sleeper=sleeper,
    )

    alias_symbol = (
        normalized.replace(".", "-")
        if "." in normalized and normalized.replace(".", "-") != primary_symbol
        else None
    )
    alias = (
        _probe_yahoo_symbol(
            alias_symbol,
            windows,
            history_fetcher=fetcher,
            retries=retries,
            delay_seconds=delay_seconds,
            sleeper=sleeper,
        )
        if alias_symbol and primary.sessions == 0
        else None
    )

    if primary.sessions:
        status = (
            "primary_history_available_with_provider_errors"
            if primary.errors else "primary_history_available_identity_review"
        )
        identity_warning = (
            "Yahoo has observations inside the historical membership windows. "
            "This is a backfill candidate only; confirm issuer/security identity, "
            "price basis, corporate actions and symbol reuse before any repair."
        )
    elif alias and alias.sessions:
        status = (
            "share_class_alias_available_with_provider_errors"
            if alias.errors else "possible_share_class_alias_identity_review"
        )
        identity_warning = (
            "A dot-to-hyphen Yahoo spelling returned observations. This is only a "
            "candidate mapping; independently confirm the exact share class and issuer."
        )
    elif primary.requests == 0 and primary.errors:
        status = "primary_provider_error"
        identity_warning = "Yahoo requests failed; no conclusion about data availability is possible."
    elif primary.errors and (alias is None or alias.errors):
        status = "provider_errors_no_conclusion"
        identity_warning = "One or more Yahoo requests failed; retry before classifying this symbol."
    else:
        status = "no_history_for_tested_yahoo_symbols"
        identity_warning = (
            "No Yahoo observations were returned for the tested original/share-class "
            "symbols in the membership windows. This does not prove the security never traded."
        )

    last_membership_end = membership[-1].end_date_exclusive
    windows_text = ";".join(
        f"{window.start_date.isoformat()}..{window.end_date_exclusive.isoformat()}"
        for window in windows
    )
    curated = RETIRED_TICKERS.get(normalized, {})

    return {
        "ticker": normalized,
        "membership_period_count": len(membership),
        "membership_windows": windows_text,
        "last_membership_end_exclusive": (
            last_membership_end.isoformat() if last_membership_end else None
        ),
        "curated_last_trade_date": curated.get("last_trade_date"),
        "ticker_shape_review": _ticker_shape_review(normalized),
        "primary_yahoo_symbol": primary_symbol,
        "primary_sessions": primary.sessions,
        "primary_first_date": primary.first_date,
        "primary_last_date": primary.last_date,
        "primary_requests": primary.requests,
        "primary_errors": primary.errors,
        "primary_dividend_events": primary.dividend_events,
        "primary_split_events": primary.split_events,
        "share_class_alias_candidate": alias_symbol,
        "alias_sessions": alias.sessions if alias else 0,
        "alias_first_date": alias.first_date if alias else None,
        "alias_last_date": alias.last_date if alias else None,
        "alias_requests": alias.requests if alias else 0,
        "alias_errors": alias.errors if alias else 0,
        "alias_dividend_events": alias.dividend_events if alias else 0,
        "alias_split_events": alias.split_events if alias else 0,
        "status": status,
        "identity_warning": identity_warning,
    }


def audit_missing_history(
    tickers_csv: Path,
    output_dir: Path,
    *,
    retries: int = 2,
    delay_seconds: float = 0.15,
    max_tickers: int | None = None,
) -> dict[str, int]:
    """Probe the missing-ticker list and write read-only audit CSVs."""
    source = pd.read_csv(tickers_csv)
    if "ticker" not in source.columns:
        raise ValueError("tickers CSV must contain a 'ticker' column")
    tickers = (
        source["ticker"].dropna().astype(str).str.strip().str.upper()
        .loc[lambda values: values.ne("")]
        .drop_duplicates().tolist()
    )
    if not tickers:
        raise ValueError("tickers CSV contains no ticker values")
    if max_tickers is not None:
        if max_tickers <= 0:
            raise ValueError("max_tickers must be > 0")
        tickers = tickers[:max_tickers]

    rows: list[dict[str, object]] = []
    for position, ticker in enumerate(tickers, start=1):
        result = audit_missing_ticker(
            ticker,
            retries=retries,
            delay_seconds=delay_seconds,
        )
        rows.append(result)
        primary_sessions = int(result["primary_sessions"])
        alias_sessions = int(result["alias_sessions"])
        print(
            f"[{position}/{len(tickers)}] {ticker}: "
            f"primary={primary_sessions}, alias={alias_sessions}, status={result['status']}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(rows, columns=AUDIT_COLUMNS).sort_values("ticker")
    audit.to_csv(output_dir / "alpha_missing_history_audit.csv", index=False)
    summary = (
        audit.groupby("status", dropna=False)
        .size().rename("tickers").reset_index()
        .sort_values(["tickers", "status"], ascending=[False, True])
    )
    summary.to_csv(output_dir / "alpha_missing_history_summary.csv", index=False)

    counts = {
        "tickers_checked": int(len(audit)),
        "primary_history_available": int(audit["primary_sessions"].gt(0).sum()),
        "possible_share_class_alias": int(audit["alias_sessions"].gt(0).sum()),
        "no_history_for_tested_symbols": int(
            audit["status"].eq("no_history_for_tested_yahoo_symbols").sum()
        ),
        "provider_errors": int(
            audit["status"].isin(
                ["primary_provider_error", "provider_errors_no_conclusion"]
            ).sum()
        ),
    }
    print("\n=== Alpha missing-history Yahoo probe ===")
    for key, value in counts.items():
        print(f"{key}: {value}")
    print(f"Reports: {output_dir.resolve()}")
    print("Read-only: Yahoo was queried for coverage only; Supabase was not modified.")
    print("No price repair or symbol mapping was approved automatically.")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Yahoo availability for historical Alpha tickers with no Supabase price history"
    )
    parser.add_argument(
        "--tickers-csv",
        type=Path,
        required=True,
        help="CSV containing a ticker column, e.g. the failed_tickers list from Alpha evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/alpha_missing_history_audit"),
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--delay-seconds", type=float, default=0.15)
    parser.add_argument("--max-tickers", type=int, default=None)
    args = parser.parse_args()
    try:
        audit_missing_history(
            args.tickers_csv,
            args.output_dir,
            retries=args.retries,
            delay_seconds=args.delay_seconds,
            max_tickers=args.max_tickers,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
