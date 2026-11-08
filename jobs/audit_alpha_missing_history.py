"""Read-only Yahoo coverage probe for historical S&P 500 tickers missing in Supabase.

The job first validates Yahoo access with a liquid, known control ticker. It
then queries missing historical symbols only inside their recorded S&P 500
membership windows, optionally testing dot-to-hyphen share-class spellings.
It never writes to Supabase, emits repair SQL, or treats provider availability
as proof of issuer/security identity.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFPricesMissingError

from data.retired_tickers import RETIRED_TICKERS
from data.sp500_universe import (
    get_sp500_membership_periods,
    get_sp500_price_periods,
    get_yfinance_symbol,
)


DEFAULT_CONTROL_TICKER = "AAPL"
DEFAULT_CONTROL_START = "2026-10-01"
DEFAULT_CONTROL_END = "2026-10-30"
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
    "primary_empty_responses",
    "primary_errors",
    "primary_dividend_events",
    "primary_split_events",
    "share_class_alias_candidate",
    "alias_sessions",
    "alias_first_date",
    "alias_last_date",
    "alias_requests",
    "alias_empty_responses",
    "alias_errors",
    "alias_dividend_events",
    "alias_split_events",
    "status",
    "identity_warning",
)
CONTROL_COLUMNS = (
    "control_ticker",
    "start_date",
    "end_date_exclusive",
    "sessions",
    "first_date",
    "last_date",
    "yfinance_version",
    "status",
)


@dataclass(frozen=True)
class YahooProbe:
    """Coverage-only result for one Yahoo symbol across membership windows."""

    symbol: str
    sessions: int
    first_date: str | None
    last_date: str | None
    requests: int
    empty_responses: int
    errors: int
    dividend_events: int
    split_events: int


def _fetch_yahoo_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch Yahoo history while surfacing errors otherwise hidden as empty frames."""
    return yf.Ticker(symbol).history(
        start=start,
        end=end,
        auto_adjust=False,
        actions=True,
        raise_errors=True,
    )


def _summarize_yahoo_history(
    history: pd.DataFrame,
) -> tuple[int, str | None, str | None, int, int]:
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
    price_values = prices.to_numpy(dtype=float)
    valid = np.isfinite(price_values) & (price_values > 0)
    valid_dates = frame.index[valid]
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


def _fetch_with_retries(
    symbol: str,
    start: str,
    end: str,
    *,
    history_fetcher: Callable[[str, str, str], pd.DataFrame],
    retries: int,
    delay_seconds: float,
    sleeper: Callable[[float], None],
) -> tuple[pd.DataFrame, int, int, int]:
    """Return history, request attempts, empty/no-data responses and other errors."""
    history = pd.DataFrame()
    requests = 0
    empty_responses = 0
    errors = 0

    for attempt in range(retries):
        requests += 1
        try:
            history = history_fetcher(symbol, start, end)
        except YFPricesMissingError:
            # Yahoo explicitly reports no prices for this symbol/range. Retry
            # once more before accepting the no-data response as stable.
            empty_responses += 1
            if attempt + 1 < retries:
                sleeper(delay_seconds * (attempt + 1))
                continue
            break
        except Exception:
            errors += 1
            if attempt + 1 < retries:
                sleeper(delay_seconds * (attempt + 1))
                continue
            break

        if history is None or history.empty:
            # Some yfinance paths still return an empty frame rather than raise.
            empty_responses += 1
            if attempt + 1 < retries:
                sleeper(delay_seconds * (attempt + 1))
                continue
        break

    return history, requests, empty_responses, errors


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
    empty_responses = 0
    dividend_events = 0
    split_events = 0

    for window_index, window in enumerate(windows):
        start = window.start_date.isoformat()
        if window.end_date_exclusive is None:
            raise ValueError(
                f"{symbol}: an open-ended membership window cannot be bounded for this audit"
            )
        end = window.end_date_exclusive.isoformat()

        history, attempts, empty_count, error_count = _fetch_with_retries(
            symbol,
            start,
            end,
            history_fetcher=history_fetcher,
            retries=retries,
            delay_seconds=delay_seconds,
            sleeper=sleeper,
        )
        requests += attempts
        empty_responses += empty_count
        errors += error_count

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
        empty_responses=empty_responses,
        errors=errors,
        dividend_events=dividend_events,
        split_events=split_events,
    )


def _validate_control_ticker(
    history_fetcher: Callable[[str, str, str], pd.DataFrame],
    control_ticker: str,
    start: str,
    end: str,
) -> dict[str, object]:
    """Fail closed unless Yahoo returns real observations for a liquid control."""
    try:
        history = history_fetcher(control_ticker, start, end)
    except Exception as exc:
        raise RuntimeError(
            f"Yahoo control check failed for {control_ticker} ({start}..{end}): {exc}. "
            "No missing-ticker results were classified; retry when Yahoo is available."
        ) from exc

    sessions, first_date, last_date, _, _ = _summarize_yahoo_history(history)
    if sessions <= 0:
        raise RuntimeError(
            f"Yahoo control check returned no valid observations for {control_ticker} "
            f"({start}..{end}). No missing-ticker results were classified; "
            "check Yahoo access and retry."
        )

    return {
        "control_ticker": control_ticker,
        "start_date": start,
        "end_date_exclusive": end,
        "sessions": sessions,
        "first_date": first_date,
        "last_date": last_date,
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "status": "passed",
    }


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
            "primary_empty_responses": 0,
            "primary_errors": 0,
            "primary_dividend_events": 0,
            "primary_split_events": 0,
            "share_class_alias_candidate": None,
            "alias_sessions": 0,
            "alias_first_date": None,
            "alias_last_date": None,
            "alias_requests": 0,
            "alias_empty_responses": 0,
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

    fetcher = history_fetcher or _fetch_yahoo_history
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
        if alias_symbol and primary.sessions == 0 and primary.errors == 0
        else None
    )

    if primary.sessions:
        status = (
            "primary_history_available_with_provider_issues"
            if primary.errors or primary.empty_responses
            else "primary_history_available_identity_review"
        )
        identity_warning = (
            "Yahoo has observations inside the historical membership windows. "
            "This is a backfill candidate only; confirm issuer/security identity, "
            "price basis, corporate actions and symbol reuse before any repair."
        )
    elif alias and alias.sessions:
        status = (
            "share_class_alias_available_with_provider_issues"
            if alias.errors or alias.empty_responses
            else "possible_share_class_alias_identity_review"
        )
        identity_warning = (
            "A dot-to-hyphen Yahoo spelling returned observations. This is only a "
            "candidate mapping; independently confirm the exact share class and issuer."
        )
    elif primary.errors or (alias and alias.errors):
        status = "provider_errors_no_conclusion"
        identity_warning = "One or more Yahoo requests failed; retry before classifying this symbol."
    else:
        status = "no_history_for_tested_yahoo_symbols"
        identity_warning = (
            "Yahoo returned no prices for the tested original/share-class symbols "
            "in the membership windows after retrying. This is not proof that the "
            "security never traded or that no other archive contains its history."
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
        "primary_empty_responses": primary.empty_responses,
        "primary_errors": primary.errors,
        "primary_dividend_events": primary.dividend_events,
        "primary_split_events": primary.split_events,
        "share_class_alias_candidate": alias_symbol,
        "alias_sessions": alias.sessions if alias else 0,
        "alias_first_date": alias.first_date if alias else None,
        "alias_last_date": alias.last_date if alias else None,
        "alias_requests": alias.requests if alias else 0,
        "alias_empty_responses": alias.empty_responses if alias else 0,
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
    control_ticker: str = DEFAULT_CONTROL_TICKER,
    control_start: str = DEFAULT_CONTROL_START,
    control_end: str = DEFAULT_CONTROL_END,
    history_fetcher: Callable[[str, str, str], pd.DataFrame] | None = None,
) -> dict[str, int]:
    """Validate Yahoo with a liquid control, then audit missing tickers."""
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
    if not control_ticker.strip():
        raise ValueError("control_ticker must not be empty")
    if pd.Timestamp(control_end) <= pd.Timestamp(control_start):
        raise ValueError("control_end must be later than control_start")

    fetcher = history_fetcher or _fetch_yahoo_history
    control = _validate_control_ticker(
        fetcher,
        control_ticker.strip().upper(),
        control_start,
        control_end,
    )
    print(
        f"Yahoo control passed: {control['control_ticker']} "
        f"{control['first_date']}..{control['last_date']} "
        f"({control['sessions']} sessions; yfinance {control['yfinance_version']})"
    )

    rows: list[dict[str, object]] = []
    for position, ticker in enumerate(tickers, start=1):
        result = audit_missing_ticker(
            ticker,
            history_fetcher=fetcher,
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
    pd.DataFrame([control], columns=CONTROL_COLUMNS).to_csv(
        output_dir / "alpha_missing_history_control.csv", index=False
    )

    counts = {
        "tickers_checked": int(len(audit)),
        "primary_history_available": int(audit["primary_sessions"].gt(0).sum()),
        "possible_share_class_alias": int(audit["alias_sessions"].gt(0).sum()),
        "no_history_for_tested_symbols": int(
            audit["status"].eq("no_history_for_tested_yahoo_symbols").sum()
        ),
        "provider_errors": int(
            audit["status"].eq("provider_errors_no_conclusion").sum()
        ),
        "control_sessions": int(control["sessions"]),
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
    parser.add_argument("--control-ticker", default=DEFAULT_CONTROL_TICKER)
    parser.add_argument("--control-start", default=DEFAULT_CONTROL_START)
    parser.add_argument("--control-end", default=DEFAULT_CONTROL_END)
    args = parser.parse_args()
    try:
        audit_missing_history(
            args.tickers_csv,
            args.output_dir,
            retries=args.retries,
            delay_seconds=args.delay_seconds,
            max_tickers=args.max_tickers,
            control_ticker=args.control_ticker,
            control_start=args.control_start,
            control_end=args.control_end,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
