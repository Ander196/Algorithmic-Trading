from __future__ import annotations

import pandas as pd
import pytest

from jobs.export_alpha_adjusted_prices import _currency_scale, _fetch_adjusted


@pytest.mark.parametrize(
    "ticker,currency,scale",
    [
        ("ASML.AS", "EUR", 1.0),
        ("SAP.DE", "EUR", 1.0),
        ("SHEL.L", "GBP", 0.01),
        ("NESN.SW", "CHF", 1.0),
        ("ERIC-B.SE", "SEK", 1.0),
        ("AAPL", "USD", 1.0),
        ("SPY", "USD", 1.0),
    ],
)
def test_currency_scale_maps_supported_listing_suffixes(
    ticker: str, currency: str, scale: float
) -> None:
    assert _currency_scale(ticker) == (currency, scale)


class _FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def select(self, *_args):
        return self

    def eq(self, *_args):
        return self

    def gte(self, *_args):
        return self

    def order(self, *_args):
        return self

    def range(self, start, end):
        self.rows = self.rows[start : end + 1]
        return self

    def execute(self):
        return type("Response", (), {"data": self.rows})()


class _FakeClient:
    def __init__(self, rows):
        self.rows = rows

    def table(self, _name):
        return _FakeQuery(self.rows)


def test_fetch_adjusted_reads_adj_close_and_returns_chronological_series() -> None:
    client = _FakeClient([
        {"price_date": "2024-01-03", "adj_close": 12.0},
        {"price_date": "2024-01-02", "adj_close": 11.0},
    ])

    series = _fetch_adjusted(client, "AAA", "2024-01-01")

    assert series.index.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert series.tolist() == [11.0, 12.0]


def test_fetch_adjusted_fails_clearly_when_history_is_missing() -> None:
    with pytest.raises(ValueError, match="No adj_close history"):
        _fetch_adjusted(_FakeClient([]), "MISSING", "2024-01-01")


from jobs.export_alpha_adjusted_prices import _align_fx_to_calendar


def test_fx_alignment_allows_leading_missing_values_without_lookahead() -> None:
    calendar = pd.bdate_range("2003-12-29", periods=5)
    quotes = pd.Series(
        [1.1, 1.12, 1.11],
        index=pd.DatetimeIndex([calendar[2], calendar[3], calendar[4]]),
    )

    aligned = _align_fx_to_calendar(quotes, calendar, currency="EUR")

    assert aligned.iloc[:2].isna().all()
    assert aligned.iloc[2:].notna().all()
    assert aligned.iloc[0:2].isna().all()


def test_fx_alignment_rejects_internal_gap_longer_than_stale_limit() -> None:
    calendar = pd.bdate_range("2024-01-01", periods=10)
    quotes = pd.Series(
        [1.1, 1.11],
        index=pd.DatetimeIndex([calendar[0], calendar[8]]),
    )

    with pytest.raises(ValueError, match="Missing FX coverage"):
        _align_fx_to_calendar(
            quotes,
            calendar,
            currency="EUR",
            max_stale_sessions=2,
        )


def test_fx_alignment_fails_when_currency_has_no_quotes() -> None:
    calendar = pd.bdate_range("2024-01-01", periods=5)
    quotes = pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    with pytest.raises(ValueError, match="No FX coverage"):
        _align_fx_to_calendar(quotes, calendar, currency="EUR")


from jobs.export_alpha_adjusted_prices import _select_backtest_calendar


def test_fx_calendar_is_limited_to_pre_oos_lookback_and_required_forward_sessions() -> None:
    calendar = pd.bdate_range("2020-01-01", periods=200)
    signal_dates = pd.DatetimeIndex([calendar[100], calendar[110], calendar[120]])

    result = _select_backtest_calendar(calendar, signal_dates, momentum_lookback=60)

    assert result[0] == calendar[40]
    assert result[-1] == calendar[122]
    assert len(result) == 83
    assert signal_dates.isin(result).all()


def test_fx_calendar_selection_rejects_missing_oos_benchmark_date() -> None:
    calendar = pd.bdate_range("2020-01-01", periods=200)
    signal_dates = pd.DatetimeIndex([calendar[100], pd.Timestamp("2020-08-30")])

    with pytest.raises(ValueError, match="Benchmark price calendar misses OOS dates"):
        _select_backtest_calendar(calendar, signal_dates, momentum_lookback=60)


def test_fx_calendar_selection_rejects_short_pre_oos_history() -> None:
    calendar = pd.bdate_range("2020-01-01", periods=100)
    signal_dates = pd.DatetimeIndex([calendar[30], calendar[40]])

    with pytest.raises(ValueError, match="Not enough benchmark history"):
        _select_backtest_calendar(calendar, signal_dates, momentum_lookback=60)
