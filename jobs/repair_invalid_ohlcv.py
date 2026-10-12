"""Build audited OHLCV repair previews and apply only explicitly approved snapshots.

The default workflow is read-only:
    python -m jobs.repair_invalid_ohlcv --input-csv invalid_ohlcv.csv
    python -m jobs.repair_invalid_ohlcv --input-csv invalid_ohlcv.csv \
        --mode intraday-high-low --report-csv intraday_preview.csv

After reviewing the CSV, set approved=True only on rows that have been approved:
    python -m jobs.repair_invalid_ohlcv --approved-report intraday_preview.csv --apply

Applying a report does not fetch a market-data provider. It checks the original
database snapshot for concurrent changes and writes only the approved fields.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv(".env.secrets")

PRICE_COLUMNS = ("open", "high", "low", "close")
VOLUME_COLUMN = "volume"
SNAPSHOT_COLUMNS = (*PRICE_COLUMNS, "adj_close", VOLUME_COLUMN)
INPUT_REQUIRED_COLUMNS = {"ticker", "price_date", *SNAPSHOT_COLUMNS}
MODES = {"provider-daily", "intraday-high-low"}
INTRADAY_MIN_BARS = 70
INTRADAY_MIN_SPAN = timedelta(hours=6)

REPORT_COLUMNS = [
    "ticker",
    "price_date",
    "reasons",
    "repair_mode",
    "repair_status",
    "repair_error",
    "approved",
    "repair_fields",
    *SNAPSHOT_COLUMNS,
    *[f"proposed_{column}" for column in SNAPSHOT_COLUMNS],
    "intraday_bars",
    "intraday_first_bar",
    "intraday_last_bar",
    "intraday_volume",
]


def fetch_adjusted_history(
    ticker: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch provider daily bars with a consistent adjusted OHLC basis."""
    return yf.Ticker(ticker).history(
        start=start,
        end=end,
        auto_adjust=True,
        repair=False,
    )


def fetch_intraday_history(
    ticker: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch regular-session five-minute bars for one session; never write data."""
    return yf.Ticker(ticker).history(
        start=start,
        end=end,
        interval="5m",
        auto_adjust=False,
        prepost=False,
        repair=False,
    )


def _normalize_columns(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    renamed = {
        column: str(column).strip().lower().replace(" ", "_")
        for column in frame.columns
    }
    return frame.rename(columns=renamed)


def normalize_provider_history(history: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize daily yfinance history to an index of YYYY-MM-DD strings."""
    columns = ["open", "high", "low", "close", "volume"]
    if history is None or history.empty:
        return pd.DataFrame(columns=columns)

    frame = _normalize_columns(history)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(
            "provider history missing required columns: "
            + ", ".join(sorted(missing))
        )

    frame["price_date"] = pd.to_datetime(frame.index).strftime("%Y-%m-%d")
    frame = frame.reset_index(drop=True)
    frame = frame.drop_duplicates("price_date", keep="last").set_index("price_date")
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # auto_adjust=True means provider Close is adjusted on the same basis as OHLC.
    frame["adj_close"] = frame["close"]
    return frame


def normalize_intraday_history(history: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize intraday OHLCV while preserving its timezone-aware timestamps."""
    columns = ["open", "high", "low", "close", "volume"]
    if history is None or history.empty:
        return pd.DataFrame(columns=columns)

    frame = _normalize_columns(history)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(
            "intraday response missing required columns: "
            + ", ".join(sorted(missing))
        )

    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if frame.index.hasnans:
        raise ValueError("intraday response contains invalid timestamps")
    if frame.index.has_duplicates:
        raise ValueError("intraday response contains duplicate timestamps")
    frame = frame.sort_index()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[columns]


def _row_quality_reasons(row: pd.Series) -> list[str]:
    """Validate a stored row and return stable, row-level OHLCV reason codes."""
    numeric: dict[str, float] = {}
    for column in (*PRICE_COLUMNS, VOLUME_COLUMN):
        try:
            numeric[column] = float(row[column])
        except (KeyError, TypeError, ValueError):
            return [f"missing_or_non_numeric_{column}"]

    if not all(np.isfinite(numeric[column]) for column in PRICE_COLUMNS):
        return ["non_finite_ohlc_values"]
    if any(numeric[column] <= 0 for column in PRICE_COLUMNS):
        return ["non_positive_ohlc_values"]
    if not np.isfinite(numeric[VOLUME_COLUMN]):
        return ["non_finite_volume"]
    if numeric[VOLUME_COLUMN] < 0:
        return ["negative_volume"]

    reasons: list[str] = []
    if numeric["high"] < max(numeric["open"], numeric["low"], numeric["close"]):
        reasons.append("high_below_ohlc_max")
    if numeric["low"] > min(numeric["open"], numeric["high"], numeric["close"]):
        reasons.append("low_above_ohlc_min")

    try:
        adj_close = float(row["adj_close"])
    except (KeyError, TypeError, ValueError):
        adj_close = np.nan
    if not np.isfinite(adj_close) or adj_close <= 0:
        reasons.append("invalid_adj_close")
    return reasons


def source_bar_invalid_reasons(bar: pd.Series) -> list[str]:
    """Validate a provider daily replacement candidate."""
    return _row_quality_reasons(
        pd.Series(
            {
                "open": bar.get("open", np.nan),
                "high": bar.get("high", np.nan),
                "low": bar.get("low", np.nan),
                "close": bar.get("close", np.nan),
                "adj_close": bar.get("adj_close", bar.get("close", np.nan)),
                "volume": bar.get("volume", np.nan),
            }
        )
    )


def _normalize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    missing = INPUT_REQUIRED_COLUMNS - set(candidates.columns)
    if missing:
        raise ValueError(
            "input audit CSV missing required columns: " + ", ".join(sorted(missing))
        )

    frame = candidates.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["price_date"] = pd.to_datetime(
        frame["price_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    if frame["ticker"].eq("").any() or frame["price_date"].isna().any():
        raise ValueError("input audit CSV contains blank ticker or invalid price_date")

    duplicates = frame.duplicated(["ticker", "price_date"], keep=False)
    if duplicates.any():
        duplicate_keys = frame.loc[duplicates, ["ticker", "price_date"]].to_dict("records")
        raise ValueError(f"input audit CSV has duplicate ticker/date rows: {duplicate_keys[:5]}")
    return frame


def _new_report_row(original: pd.Series, ticker: str, date: str, mode: str) -> dict:
    report = {
        "ticker": ticker,
        "price_date": date,
        "reasons": original.get("reasons", ""),
        "repair_mode": mode,
        "repair_status": "",
        "repair_error": "",
        "approved": False,
        "repair_fields": "",
        "intraday_bars": np.nan,
        "intraday_first_bar": "",
        "intraday_last_bar": "",
        "intraday_volume": np.nan,
    }
    for column in SNAPSHOT_COLUMNS:
        report[column] = original.get(column, np.nan)
        report[f"proposed_{column}"] = original.get(column, np.nan)
    return report


def _set_error(report: dict, status: str, message: str = "") -> dict:
    report["repair_status"] = status
    report["repair_error"] = message
    return report


def _intraday_validation_reasons(
    bars: pd.DataFrame,
    session_date: str,
) -> list[str]:
    """Fail closed unless intraday history has coherent bars and full-day coverage."""
    if bars is None or bars.empty:
        return ["intraday_data_missing"]

    try:
        frame = normalize_intraday_history(bars)
    except Exception as exc:
        return [f"invalid_intraday_response:{exc}"]

    expected_date = pd.Timestamp(session_date).date()
    if any(timestamp.date() != expected_date for timestamp in frame.index):
        return ["intraday_contains_other_session_dates"]
    if len(frame) < INTRADAY_MIN_BARS:
        return [f"insufficient_intraday_bars:{len(frame)}<{INTRADAY_MIN_BARS}"]
    if frame.index[-1] - frame.index[0] < INTRADAY_MIN_SPAN:
        return ["insufficient_intraday_session_span"]
    if frame.index.tz is None:
        return ["intraday_timezone_missing"]

    for column in (*PRICE_COLUMNS, VOLUME_COLUMN):
        values = frame[column].to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all():
            return [f"non_finite_intraday_{column}"]
        if column in PRICE_COLUMNS and (values <= 0).any():
            return ["non_positive_intraday_ohlc"]
        if column == VOLUME_COLUMN and (values < 0).any():
            return ["negative_intraday_volume"]

    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    if (highs < np.maximum.reduce([opens, lows, closes])).any():
        return ["invalid_intraday_high_low_geometry"]
    if (lows > np.minimum.reduce([opens, highs, closes])).any():
        return ["invalid_intraday_low_high_geometry"]
    return []


def repair_candidates(
    candidates: pd.DataFrame,
    *,
    mode: str = "provider-daily",
    fetcher: Callable[[str, datetime, datetime], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build a read-only repair proposal from audit rows and fresh provider data.

    provider-daily proposes a whole, valid replacement bar.
    intraday-high-low proposes changes to high/low only, preserving all other
    stored columns and requiring at least 70 valid five-minute bars spanning six
    hours. No database client or mutation is accepted by this function.
    """
    if mode not in MODES:
        raise ValueError(f"unsupported repair mode {mode!r}; choose from {sorted(MODES)}")

    frame = _normalize_candidates(candidates)
    if fetcher is None:
        fetcher = (
            fetch_intraday_history if mode == "intraday-high-low"
            else fetch_adjusted_history
        )
    reports: list[dict] = []

    for ticker, group in frame.groupby("ticker", sort=True):
        if mode == "provider-daily":
            dates = pd.to_datetime(group["price_date"])
            start = dates.min().to_pydatetime()
            end = dates.max().to_pydatetime() + timedelta(days=1)
            try:
                provider = normalize_provider_history(fetcher(ticker, start, end))
                fetch_error = None
            except Exception as exc:
                provider = pd.DataFrame()
                fetch_error = str(exc)
        else:
            provider = None
            fetch_error = None

        for _, original in group.iterrows():
            date = original["price_date"]
            report = _new_report_row(original, ticker, date, mode)

            original_reasons = _row_quality_reasons(original)
            try:
                existing_volume = float(original["volume"])
            except (TypeError, ValueError, KeyError):
                existing_volume = np.nan

            if np.isfinite(existing_volume) and existing_volume == 0:
                reports.append(_set_error(
                    report, "manual_review_existing_zero_volume",
                    "Stored volume is zero; do not auto-repair this row.",
                ))
                continue

            if mode == "intraday-high-low":
                allowed_reasons = {"high_below_ohlc_max", "low_above_ohlc_min"}
                if not original_reasons or not set(original_reasons).issubset(allowed_reasons):
                    reports.append(_set_error(
                        report, "not_eligible_intraday_high_low",
                        "Only rows whose sole defects are high/low geometry can use this mode; "
                        f"found reasons: {original_reasons}",
                    ))
                    continue
                session_start = datetime.fromisoformat(date)
                session_end = session_start + timedelta(days=1)
                try:
                    intraday = fetcher(ticker, session_start, session_end)
                except Exception as exc:
                    reports.append(_set_error(report, "intraday_fetch_error", str(exc)))
                    continue

                validation_reasons = _intraday_validation_reasons(intraday, date)
                if validation_reasons:
                    reports.append(_set_error(
                        report, "intraday_data_rejected", ";".join(validation_reasons)
                    ))
                    continue

                normalized = normalize_intraday_history(intraday)
                try:
                    open_price = float(original["open"])
                    close_price = float(original["close"])
                    original_adj_close = float(original["adj_close"])
                    daily_volume = int(original["volume"])
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    reports.append(_set_error(
                        report, "invalid_original_snapshot", str(exc)
                    ))
                    continue

                candidate_high = max(float(normalized["high"].max()), open_price, close_price)
                candidate_low = min(float(normalized["low"].min()), open_price, close_price)
                final_bar = {
                    "open": open_price,
                    "high": candidate_high,
                    "low": candidate_low,
                    "close": close_price,
                    "adj_close": original_adj_close,
                    "volume": daily_volume,
                }
                final_reasons = _row_quality_reasons(pd.Series(final_bar))
                if final_reasons:
                    reports.append(_set_error(
                        report, "candidate_rejected", ";".join(final_reasons)
                    ))
                    continue

                for column, value in final_bar.items():
                    report[f"proposed_{column}"] = value
                report["repair_fields"] = "high,low"
                report["repair_status"] = "would_update_intraday_high_low"
                report["intraday_bars"] = len(normalized)
                report["intraday_first_bar"] = normalized.index[0].isoformat()
                report["intraday_last_bar"] = normalized.index[-1].isoformat()
                report["intraday_volume"] = int(normalized["volume"].sum())
                report["repair_error"] = (
                    "Candidate uses intraday extrema; daily open, close, adj_close "
                    "and volume are preserved. Independent source verification is still advised."
                )
                reports.append(report)
                continue

            if fetch_error is not None:
                reports.append(_set_error(report, "provider_fetch_error", fetch_error))
                continue
            if date not in provider.index:
                reports.append(_set_error(report, "provider_bar_not_found"))
                continue

            source_bar = provider.loc[date]
            if isinstance(source_bar, pd.DataFrame):
                source_bar = source_bar.iloc[-1]
            invalid_reasons = source_bar_invalid_reasons(source_bar)
            if invalid_reasons:
                reports.append(_set_error(
                    report, "provider_bar_invalid", ",".join(invalid_reasons)
                ))
                continue

            proposed = {
                "open": float(source_bar["open"]),
                "high": float(source_bar["high"]),
                "low": float(source_bar["low"]),
                "close": float(source_bar["close"]),
                "adj_close": float(source_bar["adj_close"]),
                "volume": int(source_bar["volume"]),
            }
            if proposed["volume"] == 0:
                reports.append(_set_error(
                    report, "manual_review_zero_volume",
                    "Provider replacement has zero volume.",
                ))
                continue
            for column, value in proposed.items():
                report[f"proposed_{column}"] = value
            report["repair_fields"] = ",".join(SNAPSHOT_COLUMNS)
            report["repair_status"] = "would_update"

            # Existing non-geometry defects are not silently masked by a
            # valid replacement; make the replacement and original clear in report.
            report["repair_error"] = (
                f"Stored-row quality reasons before replacement: {original_reasons}"
            )
            reports.append(report)

    return pd.DataFrame(reports, columns=REPORT_COLUMNS)


def _approval_is_true(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "approved", "y"}


def _snapshot_matches(current: dict, approved_row: pd.Series) -> bool:
    """Ensure the DB row still matches every field captured in the preview."""
    for column in SNAPSHOT_COLUMNS:
        expected = approved_row.get(column, np.nan)
        actual = current.get(column)
        try:
            expected_num = float(expected)
        except (TypeError, ValueError):
            expected_num = np.nan
        try:
            actual_num = float(actual)
        except (TypeError, ValueError):
            actual_num = np.nan

        if np.isnan(expected_num) and np.isnan(actual_num):
            continue
        if not np.isfinite(expected_num) or not np.isfinite(actual_num):
            return False
        if column == "volume":
            if int(expected_num) != int(actual_num):
                return False
        elif not np.isclose(expected_num, actual_num, rtol=1e-9, atol=1e-6):
            return False
    return True


def apply_approved_report(approved_report: pd.DataFrame, client) -> pd.DataFrame:
    """Apply only approved proposals from an earlier CSV, without provider fetches.

    Optimistic snapshot checking prevents clobbering rows changed since preview.
    Returns a result report; it never deletes or inserts a price row.
    """
    required = {
        "ticker", "price_date", "repair_mode", "repair_status", "approved",
        "repair_fields", *SNAPSHOT_COLUMNS,
        *[f"proposed_{column}" for column in SNAPSHOT_COLUMNS],
    }
    missing = required - set(approved_report.columns)
    if missing:
        raise ValueError(
            "approved report missing required columns: " + ", ".join(sorted(missing))
        )

    result = approved_report.copy().reset_index(drop=True)
    eligible = result.apply(lambda row: _approval_is_true(row["approved"]), axis=1)
    if not eligible.any():
        raise ValueError(
            "approved report has no rows with approved=True; review the CSV and approve rows explicitly"
        )

    for idx, row in result.iterrows():
        if not eligible.loc[idx]:
            result.at[idx, "repair_status"] = "not_approved"
            continue

        ticker = str(row["ticker"]).strip().upper()
        date = pd.to_datetime(row["price_date"], errors="raise").strftime("%Y-%m-%d")
        mode = str(row["repair_mode"])
        status = str(row["repair_status"])
        fields_text = str(row["repair_fields"])
        fields = [field.strip() for field in fields_text.split(",") if field.strip()]

        if mode == "intraday-high-low":
            if status != "would_update_intraday_high_low" or set(fields) != {"high", "low"}:
                result.at[idx, "repair_status"] = "approval_rejected"
                result.at[idx, "repair_error"] = "Intraday mode can approve only high and low from a valid preview."
                continue
        elif mode == "provider-daily":
            if status != "would_update" or set(fields) != set(SNAPSHOT_COLUMNS):
                result.at[idx, "repair_status"] = "approval_rejected"
                result.at[idx, "repair_error"] = "Daily provider mode requires a complete valid preview."
                continue
        else:
            result.at[idx, "repair_status"] = "approval_rejected"
            result.at[idx, "repair_error"] = f"Unsupported repair mode: {mode}"
            continue

        response = (
            client.table("stock_prices")
            .select("ticker,price_date,open,high,low,close,adj_close,volume")
            .eq("ticker", ticker)
            .eq("price_date", date)
            .limit(2)
            .execute()
        )
        rows = getattr(response, "data", None) or []
        if len(rows) != 1:
            result.at[idx, "repair_status"] = "current_row_missing_or_ambiguous"
            result.at[idx, "repair_error"] = f"Expected one current row, found {len(rows)}."
            continue
        current = rows[0]
        if not _snapshot_matches(current, row):
            result.at[idx, "repair_status"] = "stale_preview_snapshot"
            result.at[idx, "repair_error"] = "Database values changed after preview; regenerate and review the proposal."
            continue

        proposed_values: dict[str, float | int] = {}
        invalid_value = False
        for field in fields:
            value = row.get(f"proposed_{field}")
            try:
                number = float(value)
            except (TypeError, ValueError):
                invalid_value = True
                break
            if not np.isfinite(number):
                invalid_value = True
                break
            proposed_values[field] = int(number) if field == "volume" else number
        if invalid_value:
            result.at[idx, "repair_status"] = "approved_values_invalid"
            result.at[idx, "repair_error"] = "One or more approved values are missing or non-finite."
            continue

        final_values = {column: current.get(column) for column in SNAPSHOT_COLUMNS}
        final_values.update(proposed_values)
        if _row_quality_reasons(pd.Series(final_values)):
            result.at[idx, "repair_status"] = "approved_values_invalid"
            result.at[idx, "repair_error"] = "Approved values fail OHLCV validation."
            continue

        try:
            # Use the captured snapshot as conditional update filters too. This
            # closes the race between the preceding SELECT and the UPDATE.
            update_query = (
                client.table("stock_prices")
                .update(proposed_values)
                .eq("ticker", ticker)
                .eq("price_date", date)
            )
            for column in SNAPSHOT_COLUMNS:
                snapshot_value = current.get(column)
                if snapshot_value is None:
                    update_query = update_query.is_(column, "null")
                else:
                    update_query = update_query.eq(column, snapshot_value)
            update_response = (
                update_query
                .select("ticker,price_date")
                .execute()
            )
            updated = getattr(update_response, "data", None) or []
            if len(updated) != 1:
                result.at[idx, "repair_status"] = "update_not_confirmed"
                result.at[idx, "repair_error"] = f"Expected one updated row, found {len(updated)}."
            else:
                result.at[idx, "repair_status"] = "updated"
                result.at[idx, "repair_error"] = ""
        except Exception as exc:
            result.at[idx, "repair_status"] = "update_error"
            result.at[idx, "repair_error"] = str(exc)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview OHLCV repairs; apply only explicitly approved preview snapshots"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-csv",
        help="Audit export from scripts/sql/audit_ohlcv_invalid_rows.sql (preview only)",
    )
    source.add_argument(
        "--approved-report",
        help="Previously reviewed preview CSV with approved=True on intended rows",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODES),
        default="provider-daily",
        help="provider-daily replaces a complete bar; intraday-high-low preserves every field except high/low",
    )
    parser.add_argument(
        "--date",
        help="Limit the audit input to one session date (YYYY-MM-DD). Required for intraday-high-low.",
    )
    parser.add_argument(
        "--tickers",
        help="Optional comma-separated ticker allowlist, e.g. BRO,CE,COF.",
    )
    parser.add_argument(
        "--report-csv",
        default="ohlcv_repair_preview.csv",
        help="Path for preview or apply-result CSV",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply approved report values exactly as reviewed; does not fetch market data",
    )
    args = parser.parse_args()

    report_path = Path(args.report_csv)
    if args.approved_report:
        if not args.apply:
            parser.error("--approved-report requires --apply")
        approved_path = Path(args.approved_report)
        if not approved_path.is_file():
            parser.error(f"approved report does not exist: {approved_path}")
        if approved_path.resolve() == report_path.resolve():
            parser.error("--report-csv must differ from --approved-report to preserve the approval snapshot")
        from jobs.common import get_supabase_client

        reviewed = pd.read_csv(approved_path)
        result = apply_approved_report(reviewed, get_supabase_client())
        result.to_csv(report_path, index=False)
        print(f"Approval rows processed: {len(result)}")
        print(f"Result report written: {report_path}")
        print(result["repair_status"].value_counts(dropna=False).to_string())
        return

    if args.apply:
        parser.error("--apply is not allowed with --input-csv; first create and review a preview, then use --approved-report")

    input_path = Path(args.input_csv)
    if not input_path.is_file():
        parser.error(f"input CSV does not exist: {input_path}")

    candidates = pd.read_csv(input_path)
    if args.mode == "intraday-high-low" and not args.date:
        parser.error("--mode intraday-high-low requires --date to avoid scanning historical audit rows")
    if args.date:
        requested_date = pd.to_datetime(args.date, errors="coerce")
        if pd.isna(requested_date):
            parser.error("--date must be a valid date in YYYY-MM-DD format")
        requested_date = requested_date.strftime("%Y-%m-%d")
        candidates = candidates.loc[
            pd.to_datetime(candidates["price_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            == requested_date
        ]
    if args.tickers:
        ticker_allowlist = {
            ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()
        }
        if not ticker_allowlist:
            parser.error("--tickers must contain at least one ticker")
        candidates = candidates.loc[
            candidates["ticker"].astype(str).str.strip().str.upper().isin(ticker_allowlist)
        ]
    if candidates.empty:
        parser.error("no audit rows matched --date/--tickers filters")

    result = repair_candidates(candidates, mode=args.mode)
    result.to_csv(report_path, index=False)
    print(f"Audit rows processed: {len(result)}")
    print(f"Preview written: {report_path}")
    print(result["repair_status"].value_counts(dropna=False).to_string())
    print("DRY RUN: Supabase was not modified. Review the CSV and set approved=True only for rows explicitly approved.")


if __name__ == "__main__":
    main()
