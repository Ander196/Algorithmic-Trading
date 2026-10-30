from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from data import data_loader_historical as loader


def test_process_ticker_backfills_from_2000_inside_membership_window(monkeypatch):
    calls = []

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 27, tzinfo=tz)

    monkeypatch.setattr(loader, "datetime", FrozenDateTime)
    monkeypatch.setattr(
        loader,
        "getEarliestPriceDate",
        lambda client, ticker, start_date=None, end_date_exclusive=None:
            datetime(2021, 1, 1, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        loader,
        "getLatestPriceDate",
        lambda client, ticker, start_date=None, end_date_exclusive=None:
            datetime(2026, 9, 25, tzinfo=timezone.utc),
    )

    def fake_fetch(ticker, startDate, endDate, retries=2):
        calls.append((ticker, startDate, endDate))
        return None

    monkeypatch.setattr(loader, "fetchPriceData", fake_fetch)

    result = loader.processTicker(
        object(),
        "AAPL",
        years_of_history=None,
        max_days_stale=3,
    )

    assert result["skipped"] == 1
    assert len(calls) == 1
    assert calls[0][0] == "AAPL"
    assert calls[0][1] == loader.HISTORICAL_START_DATE
    assert calls[0][2] == datetime(2021, 1, 1, tzinfo=timezone.utc)


def test_fetch_price_data_starts_at_2000_when_start_is_none(monkeypatch):
    calls = []

    class FakeTicker:
        def history(self, **kwargs):
            calls.append(kwargs)
            index = pd.to_datetime(["2020-01-01"])
            return pd.DataFrame(
                {
                    "Open": [10.0],
                    "High": [11.0],
                    "Low": [9.0],
                    "Close": [10.5],
                    "Volume": [1000],
                    "Adj Close": [10.5],
                },
                index=index,
            )

    monkeypatch.setattr(loader.yf, "Ticker", lambda ticker: FakeTicker())

    result = loader.fetchPriceData(
        "AAPL",
        startDate=None,
        endDate=datetime(2026, 9, 28, tzinfo=timezone.utc),
    )

    assert result is not None
    assert not result.empty
    assert calls[0]["start"] == loader.HISTORICAL_START_DATE
    assert calls[0]["end"] == datetime(2026, 9, 28, tzinfo=timezone.utc)
    assert "period" not in calls[0]


def test_upload_filters_prices_outside_membership_periods():
    class FakeQuery:
        def insert(self, records):
            self.records = records
            return self

        def execute(self):
            captured.extend(self.records)
            return SimpleNamespace(data=[])

    class FakeClient:
        def table(self, _name):
            return FakeQuery()

    captured = []
    prices = pd.DataFrame({
        "ticker": ["SBNY"] * 4,
        "price_date": ["2021-12-20", "2023-03-10", "2023-03-14", "2024-01-02"],
        "open": [1.0, 2.0, 2.5, 3.0],
        "high": [1.1, 2.1, 2.6, 3.1],
        "low": [0.9, 1.9, 2.4, 2.9],
        "close": [1.0, 2.0, 2.5, 3.0],
        "adj_close": [1.0, 2.0, 2.5, 3.0],
        "volume": [100, 200, 250, 300],
    })

    assert loader.uploadToSupabase(FakeClient(), "SBNY", prices) == 2
    assert [row["price_date"] for row in captured] == [
        "2021-12-20",
        "2023-03-10",
    ]


def test_yfinance_share_class_symbol_is_aliased(monkeypatch):
    seen = []

    class FakeTicker:
        def history(self, **kwargs):
            return pd.DataFrame(
                {
                    "Open": [10.0],
                    "High": [11.0],
                    "Low": [9.0],
                    "Close": [10.5],
                    "Volume": [100],
                    "Adj Close": [10.5],
                },
                index=pd.to_datetime(["2019-01-02"]),
            )

    def fake_ticker(symbol):
        seen.append(symbol)
        return FakeTicker()

    monkeypatch.setattr(loader.yf, "Ticker", fake_ticker)
    loader.fetchPriceData(
        "BRK.B",
        startDate=datetime(2018, 1, 1, tzinfo=timezone.utc),
        endDate=datetime(2019, 2, 1, tzinfo=timezone.utc),
    )
    assert seen == ["BRK-B"]
