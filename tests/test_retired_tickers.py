from types import SimpleNamespace

import pandas as pd

from data import data_loader_historical as historical_loader
from data import ticker_loader
from data.retired_tickers import RETIRED_TICKERS


def test_fetch_retired_ticker_info_requires_yahoo_history(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, **kwargs):
            assert kwargs == {
                "period": "max",
                "auto_adjust": False,
                "actions": False,
            }
            if self.ticker == "ATVI":
                return pd.DataFrame()
            return pd.DataFrame(
                {"Close": [10.0]},
                index=pd.to_datetime(["2020-01-02"]),
            )

    monkeypatch.setattr(ticker_loader.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(ticker_loader.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        ticker_loader,
        "RETIRED_TICKERS",
        {"SPLK": RETIRED_TICKERS["SPLK"], "ATVI": RETIRED_TICKERS["ATVI"]},
    )

    result = ticker_loader.fetchRetiredTickerInfo()

    assert result["ticker"].tolist() == ["SPLK"]
    assert bool(result.iloc[0]["historical_only"]) is True


def test_fetch_all_tickers_skips_historical_only_symbols(monkeypatch):
    fetched = []

    def fake_fetch(ticker):
        fetched.append(ticker)
        return {
            "ticker": ticker,
            "name": ticker,
            "exchange": "NMS",
            "sector": "Test",
            "industry": "Test",
        }

    monkeypatch.setattr(ticker_loader, "fetchTickerInfo", fake_fetch)
    monkeypatch.setattr(ticker_loader.time, "sleep", lambda *_: None)

    result = ticker_loader.fetchAllTickers(
        ["AAPL", "SPLK", "TWTR", "WBD"],
        delay=0,
    )

    assert fetched == ["AAPL"]
    assert result["ticker"].tolist() == ["AAPL"]


def test_historical_loader_includes_registered_retired_symbols(monkeypatch):
    class FakeQuery:
        def select(self, *_args):
            return self

        def execute(self):
            return SimpleNamespace(data=[
                {"ticker": "AAPL", "is_active": True},
                {"ticker": "SPLK", "is_active": False},
                {"ticker": "WBD", "is_active": False},
                {"ticker": "OLD", "is_active": False},
            ])

    class FakeClient:
        def table(self, name):
            assert name == "stocks"
            return FakeQuery()

    monkeypatch.setattr(
        historical_loader,
        "RETIRED_TICKERS",
        {"SPLK": RETIRED_TICKERS["SPLK"]},
    )

    assert historical_loader.getActiveTickers(FakeClient()) == ["AAPL", "SPLK"]
