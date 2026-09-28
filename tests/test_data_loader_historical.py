from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from data import data_loader_historical as loader


def test_process_ticker_defaults_to_maximum_available_history(monkeypatch):
    calls = []

    class FakeClient:
        pass

    monkeypatch.setattr(
        loader,
        "getEarliestPriceDate",
        lambda client, ticker: datetime(2021, 1, 1, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        loader,
        "getLatestPriceDate",
        lambda client, ticker: datetime(2026, 9, 25, tzinfo=timezone.utc),
    )

    def fake_fetch(ticker, startDate, endDate, retries=2):
        calls.append((ticker, startDate, endDate))
        return None

    monkeypatch.setattr(loader, "fetchPriceData", fake_fetch)

    result = loader.processTicker(
        FakeClient(),
        "AAPL",
        years_of_history=None,
        max_days_stale=3,
    )

    assert result["skipped"] == 1
    assert len(calls) == 1
    assert calls[0][0] == "AAPL"
    assert calls[0][1] is None
    assert calls[0][2] == datetime(2021, 1, 1, tzinfo=timezone.utc)


def test_fetch_price_data_uses_period_max_when_start_is_none(monkeypatch):
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
    assert calls[0]["period"] == "max"
    assert "start" not in calls[0]
    assert "end" in calls[0]
