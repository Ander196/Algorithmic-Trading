from __future__ import annotations

from io import BytesIO
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from jobs import export_alpha_adjusted_prices as export_job
from jobs.export_alpha_adjusted_prices import (
    _align_fx_to_calendar,
    _currency_scale,
    _fetch_adjusted,
    _fetch_eur_usd_reference,
    _fx_usd_per_unit,
    _select_backtest_calendar,
)

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

def test_fx_alignment_preserves_long_internal_gap_as_missing_without_lookahead() -> None:
    calendar = pd.bdate_range("2024-01-01", periods=10)
    quotes = pd.Series(
        [1.1, 1.11],
        index=pd.DatetimeIndex([calendar[0], calendar[8]]),
    )

    aligned = _align_fx_to_calendar(
        quotes,
        calendar,
        currency="EUR",
        max_stale_sessions=2,
    )

    assert aligned.iloc[:3].notna().all()
    assert aligned.iloc[3:8].isna().all()
    assert aligned.iloc[8:].notna().all()

def test_fx_alignment_fails_when_currency_has_no_quotes() -> None:
    calendar = pd.bdate_range("2024-01-01", periods=5)
    quotes = pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    with pytest.raises(ValueError, match="No FX coverage"):
        _align_fx_to_calendar(quotes, calendar, currency="EUR")

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

def _ecb_csv(
    rows: str,
    *,
    currency: str = "USD",
    denominator: str = "EUR",
    frequency: str = "D",
) -> bytes:
    return (
        "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE\n"
        + "".join(
            f"EXR.D.USD.EUR.SP00.A,{frequency},{currency},{denominator},"
            f"SP00,A,{row}\n"
            for row in rows.splitlines()
            if row.strip()
        )
    ).encode("utf-8")

def test_eur_fx_uses_ecb_csv_and_system_truststore(monkeypatch) -> None:
    payload = _ecb_csv(
        "2008-08-13,1.4903\n2008-08-14,1.4907"
    )
    requests = []
    marker_context = object()

    class FakeResponse(BytesIO):
        pass

    def fake_urlopen(request, *, context, timeout):
        requests.append((request, context, timeout))
        return FakeResponse(payload)

    monkeypatch.setattr(export_job, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        export_job.truststore,
        "SSLContext",
        lambda protocol: marker_context,
    )

    result = _fetch_eur_usd_reference("2008-08-01", "2008-08-31")

    assert result.index.tolist() == [
        pd.Timestamp("2008-08-13"),
        pd.Timestamp("2008-08-14"),
    ]
    assert result.tolist() == pytest.approx([1.4903, 1.4907])
    request, context, timeout = requests[0]
    query = parse_qs(urlparse(request.full_url).query)
    assert urlparse(request.full_url).path.endswith(
        "/service/data/EXR/D.USD.EUR.SP00.A"
    )
    assert query == {
        "startPeriod": ["2008-08-01"],
        "endPeriod": ["2008-08-31"],
        "format": ["csvdata"],
    }
    assert context is marker_context
    assert timeout == 30
    assert request.get_header("Accept") == "text/csv"

def test_eur_fx_rejects_unexpected_currency_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(
        export_job,
        "urlopen",
        lambda *_args, **_kwargs: BytesIO(
            _ecb_csv("2008-08-13,1.4903", currency="GBP")
        ),
    )

    with pytest.raises(ValueError, match="unexpected CURRENCY"):
        _fetch_eur_usd_reference("2008-08-01", "2008-08-31")

def test_eur_fx_rejects_response_without_required_columns(monkeypatch) -> None:
    monkeypatch.setattr(
        export_job,
        "urlopen",
        lambda *_args, **_kwargs: BytesIO(b"wrong_column\nfoo\n"),
    )

    with pytest.raises(ValueError, match="missing required columns"):
        _fetch_eur_usd_reference("2008-08-01", "2008-08-31")

def test_eur_fx_rejects_response_with_no_valid_observations(monkeypatch) -> None:
    monkeypatch.setattr(
        export_job,
        "urlopen",
        lambda *_args, **_kwargs: BytesIO(
            _ecb_csv("2008-08-13,not-a-number")
        ),
    )

    with pytest.raises(RuntimeError, match="no valid EUR/USD reference rates"):
        _fetch_eur_usd_reference("2008-08-01", "2008-08-31")

def test_eur_fx_wraps_network_errors_without_disabling_tls(monkeypatch) -> None:
    def fail_urlopen(*_args, **_kwargs):
        raise OSError("certificate verify failed")

    monkeypatch.setattr(export_job, "urlopen", fail_urlopen)

    with pytest.raises(RuntimeError, match="Could not download ECB EUR/USD"):
        _fetch_eur_usd_reference("2008-08-01", "2008-08-31")

def test_eur_fx_routes_to_ecb_without_using_yahoo(monkeypatch) -> None:
    expected = pd.Series(
        [1.47],
        index=pd.DatetimeIndex([pd.Timestamp("2008-08-18")]),
        name="EUR",
    )
    monkeypatch.setattr(
        export_job,
        "_fetch_eur_usd_reference",
        lambda start, end: expected,
    )
    monkeypatch.setattr(
        export_job.yf,
        "Ticker",
        lambda *_args, **_kwargs: pytest.fail("EUR must not use Yahoo FX"),
    )

    actual = _fx_usd_per_unit("EUR", "2008-08-01", "2008-08-31")

    pd.testing.assert_series_equal(actual, expected)

def test_non_eur_fx_keeps_using_yahoo(monkeypatch) -> None:
    history = pd.DataFrame(
        {"Close": [1.25]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")]),
    )

    class FakeTicker:
        def __init__(self, symbol):
            assert symbol == "GBPUSD=X"

        def history(self, **kwargs):
            assert kwargs["start"] == "2024-01-01"
            assert kwargs["auto_adjust"] is False
            return history

    monkeypatch.setattr(export_job.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(
        export_job,
        "_fetch_eur_usd_reference",
        lambda *_args, **_kwargs: pytest.fail("Non-EUR currency must not use ECB"),
    )

    actual = _fx_usd_per_unit("GBP", "2024-01-01", "2024-01-03")

    assert actual.index.tolist() == [pd.Timestamp("2024-01-02")]
    assert actual.tolist() == [1.25]
