"""Export adjusted prices from Supabase for the Alpha portfolio research backtest."""
from __future__ import annotations

import argparse
import ssl
import time
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import truststore
import yfinance as yf
from dotenv import load_dotenv

from data.excluded_tickers import filter_excluded_ticker_rows, is_excluded_ticker
from jobs.common import get_supabase_client

load_dotenv(".env.secrets")

FX = {
    "GBP": ("GBPUSD=X", False),
    "CHF": ("USDCHF=X", True),
    "SEK": ("USDSEK=X", True),
    "DKK": ("USDDKK=X", True),
    "NOK": ("USDNOK=X", True),
    "PLN": ("USDPLN=X", True),
    "CZK": ("USDCZK=X", True),
}
SUFFIXES = {
    ".AS": ("EUR", 1.0), ".DE": ("EUR", 1.0), ".PA": ("EUR", 1.0),
    ".BR": ("EUR", 1.0), ".MC": ("EUR", 1.0), ".MI": ("EUR", 1.0),
    ".VI": ("EUR", 1.0), ".HE": ("EUR", 1.0), ".LS": ("EUR", 1.0),
    ".IE": ("EUR", 1.0), ".L": ("GBP", 0.01), ".SW": ("CHF", 1.0),
    ".SE": ("SEK", 1.0), ".OL": ("NOK", 1.0), ".PW": ("PLN", 1.0),
    ".PR": ("CZK", 1.0), ".DK": ("DKK", 1.0), ".CO": ("DKK", 1.0),
}


def _currency_scale(ticker: str) -> tuple[str, float]:
    ticker = ticker.strip().upper()
    for suffix, pair in SUFFIXES.items():
        if ticker.endswith(suffix):
            return pair
    return "USD", 1.0


def _fetch_adjusted(client, ticker: str, start: str) -> pd.Series:
    records = []
    offset = 0
    while True:
        result = (
            client.table("stock_prices")
            .select("price_date,adj_close")
            .eq("ticker", ticker)
            .gte("price_date", start)
            .order("price_date")
            .range(offset, offset + 999)
            .execute()
        )
        batch = result.data or []
        if not batch:
            break
        records.extend(batch)
        offset += len(batch)
        if len(batch) < 1000:
            break
    if not records:
        raise ValueError(f"No adj_close history found for {ticker} since {start}")
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame["price_date"], utc=True).dt.tz_convert(None).dt.normalize()
    frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["adj_close"]) & (frame["adj_close"] > 0)]
    if frame.empty:
        raise ValueError(f"No valid adj_close values found for {ticker}")
    if frame["date"].duplicated().any():
        raise ValueError(f"Duplicate adjusted-price dates found for {ticker}")
    return pd.Series(frame["adj_close"].to_numpy(), index=pd.DatetimeIndex(frame["date"]), name=ticker).sort_index()


def _fetch_eur_usd_reference_chunk(start: str, end: str) -> pd.Series:
    """Fetch daily USD-per-EUR reference rates from the ECB Data Portal.

    A dedicated truststore SSLContext uses the operating system certificate
    store where supported (including Windows) without disabling TLS checks or
    modifying SSL settings globally.
    """
    query = urlencode(
        {
            "startPeriod": start,
            "endPeriod": end,
            "format": "csvdata",
        }
    )
    url = (
        "https://data-api.ecb.europa.eu/service/data/EXR/"
        f"D.USD.EUR.SP00.A?{query}"
    )
    request = Request(
        url,
        headers={
            "Accept": "text/csv",
            "User-Agent": "Algorithmic-Trading/1.0",
        },
    )
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    # Retry transient read timeouts for this bounded date chunk. Certificate and
    # other non-timeout failures are surfaced immediately rather than retried.
    for attempt in range(3):
        try:
            with urlopen(request, context=context, timeout=30) as response:
                payload = response.read()
            break
        except (OSError, TimeoutError) as exc:
            is_timeout = isinstance(exc, TimeoutError) or "timed out" in str(exc).lower()
            if is_timeout and attempt < 2:
                time.sleep(attempt + 1)
                continue
            raise RuntimeError(
                f"Could not download ECB EUR/USD reference rates for {start} to {end}: {exc}"
            ) from exc

    try:
        frame = pd.read_csv(BytesIO(payload))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"ECB returned an unreadable EUR/USD CSV response: {exc}") from exc

    required = {"TIME_PERIOD", "OBS_VALUE"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"ECB EUR/USD response is missing required columns: {sorted(missing)}"
        )

    # Validate the returned series identity when the API supplies its dimensions.
    expected_dimensions = {
        "FREQ": "D",
        "CURRENCY": "USD",
        "CURRENCY_DENOM": "EUR",
        "EXR_TYPE": "SP00",
        "EXR_SUFFIX": "A",
    }
    for column, expected in expected_dimensions.items():
        if column in frame.columns:
            observed = set(frame[column].dropna().astype(str).unique())
            if observed and observed != {expected}:
                raise ValueError(
                    f"ECB EUR/USD response has unexpected {column} values: "
                    f"{sorted(observed)}"
                )

    dates = pd.to_datetime(frame["TIME_PERIOD"], errors="coerce").dt.normalize()
    values = pd.to_numeric(frame["OBS_VALUE"], errors="coerce")
    valid = dates.notna() & np.isfinite(values) & (values > 0)
    observations = pd.DataFrame(
        {"date": dates.loc[valid], "rate": values.loc[valid]}
    )
    observations = observations.loc[
        observations["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    ]
    if observations.empty:
        raise RuntimeError(
            f"ECB returned no valid EUR/USD reference rates for {start} to {end}"
        )
    if observations["date"].duplicated().any():
        raise ValueError("ECB EUR/USD response contains duplicate observation dates")

    return pd.Series(
        observations["rate"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(observations["date"]),
        name="EUR",
    ).sort_index()



def _fetch_eur_usd_reference(start: str, end: str) -> pd.Series:
    """Fetch ECB EUR/USD reference rates in calendar-year chunks.

    The ECB endpoint can time out on multi-decade requests. Annual chunks keep
    each response bounded; only transient read timeouts are retried by the
    chunk downloader. No TLS verification is disabled and no observations are
    synthesized or forward-filled here.
    """
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if start_date > end_date:
        raise ValueError(f"ECB EUR/USD start date {start} is after end date {end}")

    chunks: list[pd.Series] = []
    chunk_start = start_date
    while chunk_start <= end_date:
        year_end = pd.Timestamp(year=chunk_start.year, month=12, day=31)
        chunk_end = min(end_date, year_end)
        chunks.append(
            _fetch_eur_usd_reference_chunk(
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
        )
        chunk_start = chunk_end + timedelta(days=1)

    rates = pd.concat(chunks).sort_index()
    if rates.index.has_duplicates:
        raise ValueError("ECB EUR/USD series contains duplicate observation dates across chunks")
    rates = rates.loc[rates.index.to_series().between(start_date, end_date).to_numpy()]
    if rates.empty:
        raise RuntimeError(
            f"ECB returned no valid EUR/USD reference rates for {start} to {end}"
        )
    rates.name = "EUR"
    return rates

def _fx_usd_per_unit(currency: str, start: str, end: str) -> pd.Series:
    """Return USD per unit of currency using the ECB for EUR and Yahoo otherwise."""
    if currency == "EUR":
        return _fetch_eur_usd_reference(start, end)
    if currency not in FX:
        raise ValueError(f"Unsupported FX currency {currency}")
    symbol, inverse = FX[currency]
    history = yf.Ticker(symbol).history(
        start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
        auto_adjust=False,
    )
    if history.empty or "Close" not in history:
        raise ValueError(f"No FX history returned for {currency} ({symbol})")
    # Preserve the provider's daily session label instead of converting the
    # timestamp to UTC first (which can move non-UTC bars to the prior date).
    index = pd.DatetimeIndex(history.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    values = pd.to_numeric(history["Close"], errors="coerce").to_numpy(dtype=float)
    series = pd.Series(values, index=index).replace([np.inf, -np.inf], np.nan).dropna()
    series = series.loc[series > 0].sort_index()
    series = series[~series.index.duplicated(keep="last")]
    if inverse:
        series = 1.0 / series
    return series

def _align_fx_to_calendar(
    fx_series: pd.Series,
    calendar: pd.DatetimeIndex,
    *,
    currency: str,
    max_stale_sessions: int = 5,
) -> pd.Series:
    """Align FX quotes without backfilling or extrapolating future rates.

    Leading gaps and internal gaps longer than the bounded forward-fill limit
    remain NaN. This permits prices for unaffected tickers/dates to be exported;
    the portfolio backtest will reject a missing next-session return if a held
    asset is affected by one of these gaps.
    """
    if max_stale_sessions < 0:
        raise ValueError("max_stale_sessions must be >= 0")
    index = pd.DatetimeIndex(fx_series.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    normalized = fx_series.copy()
    normalized.index = index.normalize()
    normalized = normalized.sort_index()
    aligned = normalized.reindex(calendar).ffill(limit=max_stale_sessions)
    valid_positions = np.flatnonzero(aligned.notna().to_numpy())
    if len(valid_positions) == 0:
        raise ValueError(f"No FX coverage for {currency} in the requested date range")
    return aligned


def _select_backtest_calendar(
    calendar: pd.DatetimeIndex,
    signal_dates: pd.DatetimeIndex,
    momentum_lookback: int,
) -> pd.DatetimeIndex:
    """Limit FX coverage checks to dates required by the portfolio backtest.

    Retain momentum_lookback benchmark sessions before the first OOS signal and
    two benchmark sessions after the last signal. Missing FX data months after
    the OOS window should not invalidate a backtest that never uses those dates.
    """
    calendar = pd.DatetimeIndex(calendar).sort_values()
    signal_dates = pd.DatetimeIndex(signal_dates).sort_values()
    missing = signal_dates[~signal_dates.isin(calendar)]
    if len(missing):
        raise ValueError(
            "Benchmark price calendar misses OOS dates: "
            + ", ".join(str(value.date()) for value in missing[:5])
        )
    first_pos = int(calendar.get_loc(signal_dates[0]))
    last_pos = int(calendar.get_loc(signal_dates[-1]))
    if first_pos < momentum_lookback:
        raise ValueError(
            "Not enough benchmark history before first OOS date for momentum"
        )
    if last_pos + 2 >= len(calendar):
        raise ValueError(
            "Price history must include two benchmark sessions after last OOS date"
        )
    return calendar[first_pos - momentum_lookback : last_pos + 3]


def export_prices(predictions_csv: Path, output_csv: Path, *, benchmark: str = "SPY",
                  momentum_lookback: int = 60, foreign_stale_sessions: int = 3) -> int:
    if momentum_lookback <= 0:
        raise ValueError("momentum_lookback must be > 0")
    if foreign_stale_sessions < 0:
        raise ValueError("foreign_stale_sessions must be >= 0")
    predictions = pd.read_csv(predictions_csv)
    if not {"date", "ticker"}.issubset(predictions.columns):
        raise ValueError("predictions CSV must contain date,ticker columns")
    if predictions.empty:
        raise ValueError("predictions CSV is empty")
    predictions["date"] = pd.to_datetime(predictions["date"], errors="raise").dt.normalize()
    predictions["ticker"] = predictions["ticker"].astype(str).str.strip().str.upper()
    if predictions[["date", "ticker"]].isna().any().any():
        raise ValueError("predictions CSV contains empty date/ticker values")
    predictions, excluded_tickers = filter_excluded_ticker_rows(
        predictions, include_historical_retired=True
    )
    if excluded_tickers:
        print(
            "Excluding retired/invalid tickers from price export: "
            + ", ".join(excluded_tickers)
        )
    if predictions.empty:
        raise ValueError("predictions contain no eligible tickers after exclusions")
    if is_excluded_ticker(benchmark):
        raise ValueError(f"benchmark ticker {benchmark!r} is excluded")
    if predictions.duplicated(["date", "ticker"]).any():
        raise ValueError("predictions CSV contains duplicate date/ticker keys")

    benchmark = benchmark.strip().upper()
    tickers = sorted(set(predictions["ticker"]) | {benchmark})
    first = predictions["date"].min()
    last = predictions["date"].max()
    start = (first - pd.Timedelta(days=max(300, momentum_lookback * 4))).strftime("%Y-%m-%d")
    client = get_supabase_client()
    native = {ticker: _fetch_adjusted(client, ticker, start) for ticker in tickers}
    calendar = native[benchmark].index.sort_values()
    signal_dates = pd.DatetimeIndex(sorted(predictions["date"].unique()))
    calendar = _select_backtest_calendar(calendar, signal_dates, momentum_lookback)

    currencies = sorted({_currency_scale(t)[0] for t in tickers} - {"USD"})
    fx = {
        currency: _fx_usd_per_unit(currency, start, calendar[-1].strftime("%Y-%m-%d"))
        for currency in currencies
    }
    panel = {}
    for ticker in tickers:
        currency, unit_scale = _currency_scale(ticker)
        values = native[ticker].reindex(calendar)
        if currency != "USD":
            values = values.ffill(limit=foreign_stale_sessions)
            fx_values = _align_fx_to_calendar(
                fx[currency],
                calendar,
                currency=currency,
                max_stale_sessions=5,
            )
            # Keep pre-first-quote prices missing; do not use a future FX rate
            # to invent historical USD values. This makes foreign tickers
            # temporarily ineligible at the start of the benchmark calendar.
            values = values * unit_scale * fx_values
        else:
            values = values * unit_scale
        panel[ticker] = values.where(np.isfinite(values) & (values > 0))

    prices = pd.DataFrame(panel, index=calendar)
    if prices[benchmark].isna().any():
        raise ValueError("Benchmark price series has missing values")
    missing_by_ticker = prices.isna().sum()
    affected = missing_by_ticker[missing_by_ticker > 0].sort_values(ascending=False)
    if not affected.empty:
        examples = ", ".join(
            f"{ticker}={int(count)} sessions"
            for ticker, count in affected.head(12).items()
        )
        print(
            "WARNING: some adjusted prices are missing after bounded local-price/FX "
            f"alignment ({examples}). The CSV keeps these observations absent rather "
            "than using future FX quotes. The backtest will fail if a selected holding "
            "has no valid next-session return."
        )
    output = (
        prices.rename_axis(index="date", columns="ticker")
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="adjusted_close_base")
        .dropna(subset=["adjusted_close_base"])
    )
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    return len(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one-base-currency adjusted prices for Alpha backtest")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("reports/adjusted_close_prices_base_currency.csv"))
    parser.add_argument("--benchmark-ticker", default="SPY")
    parser.add_argument("--momentum-lookback", type=int, default=60)
    parser.add_argument("--foreign-stale-sessions", type=int, default=3)
    args = parser.parse_args()
    try:
        count = export_prices(args.predictions_csv, args.output_csv,
                              benchmark=args.benchmark_ticker,
                              momentum_lookback=args.momentum_lookback,
                              foreign_stale_sessions=args.foreign_stale_sessions)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Exported {count:,} ticker-date prices to {args.output_csv.resolve()}")
    print("Prices are converted to USD. Supabase is read-only.")


if __name__ == "__main__":
    main()
