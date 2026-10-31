from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from data import risk_history
from jobs import daily_allocation


def _provider_frame(index):
    count = len(index)
    return pd.DataFrame(
        {
            "Open": [10.0 + i for i in range(count)],
            "High": [11.0 + i for i in range(count)],
            "Low": [9.0 + i for i in range(count)],
            "Close": [10.5 + i for i in range(count)],
            "Volume": [1000 + i for i in range(count)],
        },
        index=index,
    )


def test_fetch_pre_membership_history_is_adjusted_and_excludes_entry_date(monkeypatch):
    calls = []

    class FakeTicker:
        def history(self, **kwargs):
            calls.append(kwargs)
            return _provider_frame(
                pd.to_datetime(
                    ["2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09"]
                )
            )

    seen_symbols = []
    monkeypatch.setattr(
        risk_history.yf,
        "Ticker",
        lambda symbol: (seen_symbols.append(symbol) or FakeTicker()),
    )

    result = risk_history.fetch_pre_membership_price_history(
        "BRK.B",
        date(2026, 10, 8),
        lookback_days=180,
        retries=1,
    )

    assert seen_symbols == ["BRK-B"]
    assert calls[0]["start"] == pd.Timestamp("2026-04-11", tz="UTC").to_pydatetime()
    assert calls[0]["end"] == pd.Timestamp("2026-10-08", tz="UTC").to_pydatetime()
    assert calls[0]["auto_adjust"] is True
    assert calls[0]["actions"] is False
    assert result["price_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-10-06",
        "2026-10-07",
    ]


def test_fetch_pre_membership_history_rejects_empty_provider_response(monkeypatch):
    class FakeTicker:
        def history(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(risk_history.yf, "Ticker", lambda _symbol: FakeTicker())

    with pytest.raises(RuntimeError, match="returned no pre-membership history"):
        risk_history.fetch_pre_membership_price_history(
            "SKYD",
            date(2026, 10, 8),
            retries=1,
        )


def test_fetch_pre_membership_history_rejects_invalid_ohlcv(monkeypatch):
    class FakeTicker:
        def history(self, **kwargs):
            frame = _provider_frame(pd.to_datetime(["2026-10-07"]))
            frame.loc[frame.index[0], "Close"] = -1
            return frame

    monkeypatch.setattr(risk_history.yf, "Ticker", lambda _symbol: FakeTicker())

    with pytest.raises(RuntimeError, match="no valid pre-membership OHLCV rows"):
        risk_history.fetch_pre_membership_price_history(
            "SKYD",
            date(2026, 10, 8),
            retries=1,
        )


def _history_frame(dates, close_start=10.0):
    return pd.DataFrame(
        {
            "price_date": pd.to_datetime(dates, utc=True),
            "open": [close_start + i for i in range(len(dates))],
            "high": [close_start + i + 1 for i in range(len(dates))],
            "low": [close_start + i - 1 for i in range(len(dates))],
            "close": [close_start + i for i in range(len(dates))],
            "volume": [1000] * len(dates),
        }
    )


def test_augment_stock_history_uses_transient_pre_membership_bars(monkeypatch):
    membership_start = date(2026, 10, 8)
    all_dates = pd.bdate_range("2026-06-01", "2026-10-30", tz="UTC")
    prior_dates = all_dates[all_dates < pd.Timestamp(membership_start, tz="UTC")]
    current_dates = all_dates[all_dates >= pd.Timestamp(membership_start, tz="UTC")]
    auxiliary_dates = prior_dates[-50:]

    market_history = _history_frame(all_dates)
    current_history = _history_frame(current_dates, close_start=200.0)
    # Include the entry date in the mocked provider data to exercise the caller's
    # second boundary check; the current in-membership close must win the merge.
    provider_dates = auxiliary_dates.append(
        pd.DatetimeIndex([pd.Timestamp(membership_start, tz="UTC")])
    )
    auxiliary_history = _history_frame(provider_dates, close_start=10.0)
    calls = []

    monkeypatch.setattr(
        daily_allocation,
        "get_sp500_price_periods",
        lambda _ticker: [
            SimpleNamespace(start_date=membership_start, end_date_exclusive=None)
        ],
    )

    def fake_fetch(ticker, start, *, lookback_days):
        calls.append((ticker, start, lookback_days))
        return auxiliary_history

    monkeypatch.setattr(
        daily_allocation,
        "fetch_pre_membership_price_history",
        fake_fetch,
    )

    combined, extra_bars = daily_allocation._augment_stock_history_for_beta(
        "SKYD",
        market_history,
        current_history,
        minimum_sessions=64,
    )
    joined = daily_allocation._join_stock_market_history(
        market_history,
        combined,
    )

    assert calls == [("SKYD", membership_start, 180)]
    assert extra_bars == 50
    assert len(joined) == len(current_dates) + 50
    assert combined["price_date"].is_unique
    entry_row = combined.loc[
        combined["price_date"] == pd.Timestamp(membership_start, tz="UTC")
    ].iloc[0]
    assert entry_row["close"] == current_history.iloc[0]["close"]


def test_augment_stock_history_skips_auxiliary_fetch_when_window_is_sufficient(monkeypatch):
    dates = pd.bdate_range("2026-01-01", periods=90, tz="UTC")
    stock_history = _history_frame(dates)
    market_history = _history_frame(dates)

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("auxiliary source must not be called")

    monkeypatch.setattr(
        daily_allocation,
        "get_sp500_price_periods",
        lambda _ticker: [],
    )
    monkeypatch.setattr(
        daily_allocation,
        "fetch_pre_membership_price_history",
        unexpected_fetch,
    )

    result, extra_bars = daily_allocation._augment_stock_history_for_beta(
        "AAPL",
        market_history,
        stock_history,
        minimum_sessions=64,
    )

    assert result.equals(stock_history)
    assert extra_bars == 0


def test_augment_stock_history_does_not_fetch_without_matching_membership_period(monkeypatch):
    dates = pd.bdate_range("2026-10-08", periods=17, tz="UTC")
    stock_history = _history_frame(dates)
    market_history = _history_frame(pd.bdate_range("2026-06-01", periods=110, tz="UTC"))

    monkeypatch.setattr(
        daily_allocation,
        "get_sp500_price_periods",
        lambda _ticker: [],
    )

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("no current membership interval means no fetch")

    monkeypatch.setattr(
        daily_allocation,
        "fetch_pre_membership_price_history",
        unexpected_fetch,
    )

    result, extra_bars = daily_allocation._augment_stock_history_for_beta(
        "UNKNOWN",
        market_history,
        stock_history,
        minimum_sessions=64,
    )

    assert result.equals(stock_history)
    assert extra_bars == 0
