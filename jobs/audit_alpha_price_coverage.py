"""Audit and prepare repairs for missing adjusted OHLCV rows used by Alpha backtests.

The job is deliberately read-only. It compares Supabase rows against Yahoo Finance
sessions during the required OOS/backtest calendar, then writes audit CSVs and an
idempotent SQL repair file. It never writes to Supabase directly.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from jobs.common import get_supabase_client
from jobs.export_alpha_adjusted_prices import _select_backtest_calendar


OHLCV_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")
SUMMARY_COLUMNS = (
    "ticker",
    "yahoo_sessions",
    "supabase_sessions",
    "missing_sessions",
    "repair_candidates",
    "calibration_observations",
    "scale_factor",
    "p95_relative_dispersion",
    "status",
    "details",
)
GAP_COLUMNS = (
    "ticker",
    "date",
    "classification",
    "yahoo_adj_close",
    "scale_factor",
    "p95_relative_dispersion",
    "details",
)


def _read_predictions(path: Path) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str]]:
    rows = pd.read_csv(path)
    required = {"date", "ticker"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"predictions CSV is missing columns: {sorted(missing)}")
    if rows.empty:
        raise ValueError("predictions CSV is empty")
    rows["date"] = pd.to_datetime(rows["date"], errors="raise").dt.normalize()
    if rows[["date", "ticker"]].isna().any().any():
        raise ValueError("predictions CSV contains empty date/ticker values")
    rows["ticker"] = rows["ticker"].astype(str).str.strip().str.upper()
    if (rows["ticker"] == "").any():
        raise ValueError("predictions CSV contains empty ticker values")
    if rows.duplicated(["date", "ticker"]).any():
        raise ValueError("predictions CSV contains duplicate date/ticker rows")
    signal_dates = pd.DatetimeIndex(sorted(rows["date"].unique()))
    tickers = sorted(set(rows["ticker"]))
    return rows, signal_dates, tickers


def _normalize_yahoo_history(history: pd.DataFrame) -> pd.DataFrame:
    """Convert Yahoo raw OHLC plus Adj Close into adjusted OHLCV fields."""
    columns = list(OHLCV_COLUMNS)
    if history is None or history.empty:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([]))

    required = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Yahoo history is missing columns: {sorted(missing)}")

    frame = history.copy()
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        # Preserve Yahoo's local exchange session date instead of shifting it to UTC.
        index = index.tz_localize(None)
    frame.index = index.normalize()
    frame = frame.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]

    numeric = {}
    for source, target in (
        ("Open", "open"),
        ("High", "high"),
        ("Low", "low"),
        ("Close", "close"),
        ("Adj Close", "adj_close"),
        ("Volume", "volume"),
    ):
        numeric[target] = pd.to_numeric(frame[source], errors="coerce")

    raw_close = numeric["close"]
    adjusted_close = numeric["adj_close"]
    factor = adjusted_close / raw_close.where(raw_close > 0)
    result = pd.DataFrame(index=frame.index)
    for column in ("open", "high", "low", "close"):
        result[column] = numeric[column] * factor
    result["adj_close"] = adjusted_close
    result["volume"] = numeric["volume"]

    prices = result[["open", "high", "low", "close", "adj_close"]].to_numpy(dtype=float)
    volumes = result["volume"].to_numpy(dtype=float)
    valid = (
        np.isfinite(prices).all(axis=1)
        & (prices > 0).all(axis=1)
        & np.isfinite(volumes)
        & (volumes >= 0)
    )
    result = result.loc[valid, columns].copy()
    result["volume"] = result["volume"].astype("int64")
    return result.sort_index()


def _fetch_yahoo_history(
    ticker: str,
    start: str,
    end: str,
    *,
    retries: int = 3,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            history = yf.Ticker(ticker).history(
                start=start,
                end=end,
                auto_adjust=False,
                actions=False,
            )
            if history.empty:
                raise ValueError(f"Yahoo returned no price history for {ticker}")
            return _normalize_yahoo_history(history)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Yahoo history failed for {ticker}: {last_error}")


def _fetch_supabase_adjusted(
    client,
    ticker: str,
    start: str,
) -> pd.Series:
    """Read adjusted closes with pagination; an absent ticker returns an empty series."""
    records: list[dict] = []
    offset = 0
    page_size = 1000
    while True:
        response = (
            client.table("stock_prices")
            .select("price_date,adj_close")
            .eq("ticker", ticker)
            .gte("price_date", start)
            .order("price_date")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = response.data or []
        if not batch:
            break
        records.extend(batch)
        offset += len(batch)
        if len(batch) < page_size:
            break
    if not records:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]), name=ticker)

    frame = pd.DataFrame(records)
    dates = pd.to_datetime(frame["price_date"], errors="raise")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    frame["date"] = dates.dt.normalize()
    frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["adj_close"]) & (frame["adj_close"] > 0)]
    if frame["date"].duplicated().any():
        raise ValueError(f"Supabase has duplicate price dates for {ticker}")
    return pd.Series(
        frame["adj_close"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["date"]),
        name=ticker,
    ).sort_index()


def _calibrate_price_scale(
    stored_adj_close: pd.Series,
    yahoo_adj_close: pd.Series,
    *,
    min_overlap: int = 5,
    max_dispersion: float = 0.005,
) -> dict[str, float | int | str | None]:
    """Infer a stable Yahoo-to-Supabase scale from overlapping adjusted closes.

    A stable constant scale is allowed; a time-varying mismatch is not. In the
    latter case candidate rows are reported but deliberately excluded from SQL.
    """
    overlap_dates = stored_adj_close.index.intersection(yahoo_adj_close.index)
    stored = pd.to_numeric(stored_adj_close.reindex(overlap_dates), errors="coerce")
    source = pd.to_numeric(yahoo_adj_close.reindex(overlap_dates), errors="coerce")
    valid = (
        np.isfinite(stored.to_numpy(dtype=float))
        & np.isfinite(source.to_numpy(dtype=float))
        & (stored.to_numpy(dtype=float) > 0)
        & (source.to_numpy(dtype=float) > 0)
    )
    ratios = (stored.loc[valid] / source.loc[valid]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratios) < min_overlap:
        return {
            "status": "insufficient_overlap",
            "observations": int(len(ratios)),
            "scale_factor": None,
            "p95_relative_dispersion": None,
        }

    scale = float(ratios.median())
    if not np.isfinite(scale) or scale <= 0:
        return {
            "status": "invalid_scale",
            "observations": int(len(ratios)),
            "scale_factor": None,
            "p95_relative_dispersion": None,
        }
    dispersion = float((ratios / scale - 1.0).abs().quantile(0.95))
    status = "calibrated" if dispersion <= max_dispersion else "inconsistent_price_basis"
    return {
        "status": status,
        "observations": int(len(ratios)),
        "scale_factor": scale,
        "p95_relative_dispersion": dispersion,
    }


def _find_missing_sessions(
    yahoo_history: pd.DataFrame,
    stored_adj_close: pd.Series,
    calendar: pd.DatetimeIndex,
) -> list[pd.Timestamp]:
    """Return sessions with a Yahoo bar but no valid Supabase adjusted close."""
    source = yahoo_history.reindex(calendar)
    existing_dates = stored_adj_close.index.intersection(calendar)
    expected_dates = source.index[source["adj_close"].notna()]
    return [pd.Timestamp(date) for date in expected_dates if date not in existing_dates]


def _sql_number(value: float, decimals: int = 8) -> str:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("Cannot render a non-finite SQL numeric value")
    return f"{number:.{decimals}f}"


def _render_repair_sql(rows: list[dict], batch_size: int = 200) -> str:
    """Render one transaction of idempotent INSERTs for reviewed candidates."""
    if not rows:
        return (
            "-- No automatically repairable price gaps were found.\n"
            "-- Review reports/alpha_price_coverage_audit before changing source data.\n"
        )

    statements = [
        "-- Generated by jobs.audit_alpha_price_coverage; review before execution.",
        "-- Only absent (ticker, price_date) keys are inserted; existing rows are not overwritten.",
        "BEGIN;",
    ]
    columns = (
        "price_date", "ticker", '"open"', '"high"', '"low"',
        '"close"', "adj_close", "volume",
    )
    cte_columns = "price_date, ticker, " + ', '.join(columns[2:])
    insert_columns = ", ".join(columns)
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        values = []
        for row in batch:
            ticker = str(row["ticker"]).replace("'", "''")
            volume = int(row["volume"])
            values.append(
                "("
                f"DATE '{pd.Timestamp(row['price_date']).date().isoformat()}', "
                f"'{ticker}', "
                f"{_sql_number(row['open'])}, "
                f"{_sql_number(row['high'])}, "
                f"{_sql_number(row['low'])}, "
                f"{_sql_number(row['close'])}, "
                f"{_sql_number(row['adj_close'])}, "
                f"{volume}"
                ")"
            )
        statements.extend(
            [
                "",
                f"WITH proposed ({cte_columns}) AS (",
                "    VALUES",
                "        " + ",\n        ".join(values),
                ")",
                f"INSERT INTO public.stock_prices ({insert_columns})",
                "SELECT p.price_date, p.ticker, p.\"open\", p.\"high\", p.\"low\",",
                "       p.\"close\", p.adj_close, p.volume",
                "FROM proposed AS p",
                "WHERE NOT EXISTS (",
                "    SELECT 1 FROM public.stock_prices AS existing",
                "    WHERE existing.price_date = p.price_date",
                "      AND existing.ticker = p.ticker",
                ");",
            ]
        )
    statements.extend(["", "COMMIT;", ""])
    return "\n".join(statements)


def audit_price_coverage(
    predictions_csv: Path,
    output_dir: Path,
    *,
    benchmark_ticker: str = "SPY",
    momentum_lookback: int = 60,
    min_overlap: int = 5,
    max_scale_dispersion: float = 0.005,
) -> dict[str, int]:
    if momentum_lookback <= 0:
        raise ValueError("momentum_lookback must be > 0")
    if min_overlap < 1:
        raise ValueError("min_overlap must be >= 1")
    if not np.isfinite(max_scale_dispersion) or max_scale_dispersion < 0:
        raise ValueError("max_scale_dispersion must be finite and >= 0")

    _, signal_dates, prediction_tickers = _read_predictions(predictions_csv)
    benchmark_ticker = benchmark_ticker.strip().upper()
    tickers = sorted(set(prediction_tickers) | {benchmark_ticker})
    # Fetch ample calendar context, then keep exactly the window required by
    # the backtest (lookback sessions before the first signal and two after the last).
    start = (signal_dates[0] - pd.Timedelta(days=max(365, momentum_lookback * 5))).strftime("%Y-%m-%d")
    end = (signal_dates[-1] + pd.Timedelta(days=21)).strftime("%Y-%m-%d")
    benchmark_history = _fetch_yahoo_history(benchmark_ticker, start, end)
    calendar = _select_backtest_calendar(
        benchmark_history.index,
        signal_dates,
        momentum_lookback,
    )
    benchmark_calendar_history = benchmark_history.reindex(calendar)

    client = get_supabase_client()
    summary_rows: list[dict] = []
    gap_rows: list[dict] = []
    repair_rows: list[dict] = []
    database_errors = 0

    for position, ticker in enumerate(tickers, start=1):
        try:
            yahoo_history = (
                benchmark_calendar_history.copy()
                if ticker == benchmark_ticker
                else _fetch_yahoo_history(ticker, start, end).reindex(calendar)
            )
        except Exception as exc:
            database_errors += 1
            summary_rows.append({
                "ticker": ticker,
                "yahoo_sessions": 0,
                "supabase_sessions": 0,
                "missing_sessions": 0,
                "repair_candidates": 0,
                "calibration_observations": 0,
                "scale_factor": None,
                "p95_relative_dispersion": None,
                "status": "yahoo_provider_error",
                "details": str(exc),
            })
            print(f"[{position}/{len(tickers)}] {ticker}: Yahoo error: {exc}")
            continue

        try:
            stored = _fetch_supabase_adjusted(client, ticker, start)
        except Exception as exc:
            database_errors += 1
            summary_rows.append({
                "ticker": ticker,
                "yahoo_sessions": int(yahoo_history["adj_close"].notna().sum()),
                "supabase_sessions": 0,
                "missing_sessions": 0,
                "repair_candidates": 0,
                "calibration_observations": 0,
                "scale_factor": None,
                "p95_relative_dispersion": None,
                "status": "supabase_read_error",
                "details": str(exc),
            })
            print(f"[{position}/{len(tickers)}] {ticker}: Supabase read error: {exc}")
            continue

        stored_window = stored.loc[stored.index.isin(calendar)]
        yahoo_sessions = int(yahoo_history["adj_close"].notna().sum())
        supabase_sessions = int(len(stored_window))
        if stored_window.empty:
            summary_rows.append({
                "ticker": ticker,
                "yahoo_sessions": yahoo_sessions,
                "supabase_sessions": 0,
                "missing_sessions": yahoo_sessions,
                "repair_candidates": 0,
                "calibration_observations": 0,
                "scale_factor": None,
                "p95_relative_dispersion": None,
                "status": "no_supabase_history",
                "details": "No stored adjusted close in the audit window; no rows auto-approved.",
            })
            for missing_date in yahoo_history.index[yahoo_history["adj_close"].notna()]:
                gap_rows.append({
                    "ticker": ticker,
                    "date": missing_date.date().isoformat(),
                    "classification": "requires_review_no_supabase_history",
                    "yahoo_adj_close": float(yahoo_history.loc[missing_date, "adj_close"]),
                    "scale_factor": None,
                    "p95_relative_dispersion": None,
                    "details": "No overlapping stored history to calibrate price basis.",
                })
            print(f"[{position}/{len(tickers)}] {ticker}: no stored history; manual review")
            continue

        calibration = _calibrate_price_scale(
            stored_window,
            yahoo_history["adj_close"],
            min_overlap=min_overlap,
            max_dispersion=max_scale_dispersion,
        )
        missing_dates = _find_missing_sessions(yahoo_history, stored_window, calendar)
        scale_factor = calibration["scale_factor"]
        dispersion = calibration["p95_relative_dispersion"]
        calibrated = calibration["status"] == "calibrated"
        candidate_count = 0

        for missing_date in missing_dates:
            source_row = yahoo_history.loc[missing_date]
            if calibrated:
                row = {
                    "price_date": missing_date.date().isoformat(),
                    "ticker": ticker,
                    "open": float(source_row["open"]) * float(scale_factor),
                    "high": float(source_row["high"]) * float(scale_factor),
                    "low": float(source_row["low"]) * float(scale_factor),
                    "close": float(source_row["close"]) * float(scale_factor),
                    "adj_close": float(source_row["adj_close"]) * float(scale_factor),
                    "volume": int(source_row["volume"]),
                }
                repair_rows.append(row)
                candidate_count += 1
                classification = "repair_candidate"
                details = "Yahoo adjusted OHLCV calibrated against overlapping Supabase adjusted closes."
            else:
                classification = f"requires_review_{calibration['status']}"
                details = (
                    "Not included in repair SQL because the price basis could not be "
                    "calibrated with sufficient stable overlap."
                )
            gap_rows.append({
                "ticker": ticker,
                "date": missing_date.date().isoformat(),
                "classification": classification,
                "yahoo_adj_close": float(source_row["adj_close"]),
                "scale_factor": scale_factor,
                "p95_relative_dispersion": dispersion,
                "details": details,
            })

        if not missing_dates:
            status = "ok" if calibrated else calibration["status"]
            details = "No Yahoo trading sessions were missing from Supabase in the audit window."
        elif calibrated:
            status = "repairable_gaps"
            details = "All identified gaps passed the adjusted-price basis consistency check."
        else:
            status = calibration["status"]
            details = "Price gaps found, but not approved for automatic repair."
        summary_rows.append({
            "ticker": ticker,
            "yahoo_sessions": yahoo_sessions,
            "supabase_sessions": supabase_sessions,
            "missing_sessions": len(missing_dates),
            "repair_candidates": candidate_count,
            "calibration_observations": calibration["observations"],
            "scale_factor": scale_factor,
            "p95_relative_dispersion": dispersion,
            "status": status,
            "details": details,
        })
        print(
            f"[{position}/{len(tickers)}] {ticker}: "
            f"{len(missing_dates)} missing, {candidate_count} repair candidate(s), "
            f"status={status}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    gaps = pd.DataFrame(gap_rows, columns=GAP_COLUMNS)
    candidates = pd.DataFrame(repair_rows, columns=("price_date", "ticker", *OHLCV_COLUMNS))
    summary.to_csv(output_dir / "price_coverage_summary.csv", index=False)
    gaps.to_csv(output_dir / "price_coverage_gaps.csv", index=False)
    candidates.to_csv(output_dir / "price_repair_candidates.csv", index=False)
    (output_dir / "price_repair.sql").write_text(
        _render_repair_sql(repair_rows),
        encoding="utf-8",
    )

    counts = {
        "tickers": len(tickers),
        "gap_rows": len(gap_rows),
        "repair_candidates": len(repair_rows),
        "review_gaps": sum(
            row["classification"] != "repair_candidate" for row in gap_rows
        ),
        "provider_or_database_errors": database_errors,
    }
    print("\n=== Alpha price coverage audit ===")
    print(f"Tickers checked: {counts['tickers']}")
    print(f"Gap rows found: {counts['gap_rows']}")
    print(f"Automatically repairable rows: {counts['repair_candidates']}")
    print(f"Rows requiring review: {counts['review_gaps']}")
    print(f"Provider/database errors: {counts['provider_or_database_errors']}")
    print(f"Reports: {output_dir.resolve()}")
    print("No Supabase rows were changed. Review price_repair.sql before execution.")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit missing historical price sessions and generate a reviewed SQL repair file"
    )
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/alpha_price_coverage_audit"),
    )
    parser.add_argument("--benchmark-ticker", default="SPY")
    parser.add_argument("--momentum-lookback", type=int, default=60)
    parser.add_argument("--min-overlap", type=int, default=5)
    parser.add_argument("--max-scale-dispersion", type=float, default=0.005)
    args = parser.parse_args()
    try:
        audit_price_coverage(
            args.predictions_csv,
            args.output_dir,
            benchmark_ticker=args.benchmark_ticker,
            momentum_lookback=args.momentum_lookback,
            min_overlap=args.min_overlap,
            max_scale_dispersion=args.max_scale_dispersion,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
