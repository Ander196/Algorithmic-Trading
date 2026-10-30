from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from data import data_loader_historical as historical_loader
from data import ticker_loader
from data.retired_tickers import RETIRED_TICKERS


def test_fetch_retired_ticker_info_requires_yahoo_history_and_respects_cutoff(monkeypatch):
    calls = []

    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, **kwargs):
            calls.append((self.ticker, kwargs))
            assert kwargs["start"] == "2000-01-01"
            assert kwargs["auto_adjust"] is False
            assert kwargs["actions"] is False
            assert kwargs["raise_errors"] is True
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
    assert calls[0][1]["end"].date().isoformat() == "2024-03-16"


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



def test_upload_keeps_retired_tickers_inactive(monkeypatch):
    class FakeQuery:
        def __init__(self, client):
            self.client = client

        def select(self, *_args):
            return self

        def execute(self):
            return SimpleNamespace(data=self.client.existing_rows)

        def upsert(self, records, on_conflict=None):
            self.client.upserted_records = records
            return self

    class FakeClient:
        def __init__(self):
            self.existing_rows = [{"ticker": "AAPL", "is_active": True}]
            self.upserted_records = []

        def table(self, name):
            assert name == "stocks"
            return FakeQuery(self)

    client = FakeClient()
    df = pd.DataFrame([
        {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "exchange": "NMS",
            "sector": "Technology",
            "industry": "Consumer Electronics",
        },
        {
            "ticker": "SPLK",
            **RETIRED_TICKERS["SPLK"],
            "historical_only": True,
        },
    ])

    ticker_loader.uploadToSupabase(df, client)

    records = {record["ticker"]: record for record in client.upserted_records}
    assert records["AAPL"]["is_active"] is True
    assert "created_at" in records["AAPL"]
    assert records["SPLK"]["is_active"] is False
    assert "created_at" not in records["SPLK"]



@pytest.mark.parametrize("module", [ticker_loader, historical_loader])
def test_supabase_clients_require_privileged_key(monkeypatch, module):
    captured = {}

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setenv("SUPABASE_ANON_KEY", "public-anon-key")

    with pytest.raises(ValueError, match="SUPABASE_SECRET_KEY"):
        module.getSupabaseClient()

    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_test_only")

    def fake_create_client(url, key):
        captured["url"] = url
        captured["key"] = key
        return "fake-client"

    monkeypatch.setattr(module.supabase, "create_client", fake_create_client)

    assert module.getSupabaseClient() == "fake-client"
    assert captured == {
        "url": "https://example.supabase.co",
        "key": "sb_secret_test_only",
    }


def test_process_retired_ticker_never_fetches_after_last_trade_date(monkeypatch):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 25, tzinfo=tz)

    calls = []

    monkeypatch.setattr(historical_loader, "datetime", FrozenDateTime)
    monkeypatch.setattr(
        historical_loader,
        "RETIRED_TICKERS",
        {"SPLK": {"last_trade_date": "2024-03-15"}},
    )
    monkeypatch.setattr(
        historical_loader,
        "getEarliestPriceDate",
        lambda client, ticker: None,
    )
    monkeypatch.setattr(
        historical_loader,
        "getLatestPriceDate",
        lambda client, ticker: None,
    )

    def fake_fetch(ticker, startDate, endDate, retries=2):
        calls.append((ticker, startDate, endDate))
        return None

    monkeypatch.setattr(historical_loader, "fetchPriceData", fake_fetch)

    result = historical_loader.processTicker(object(), "SPLK")

    assert result["error"] is None
    assert calls[0][0] == "SPLK"
    assert calls[0][1] == historical_loader.HISTORICAL_START_DATE
    assert calls[0][2] == datetime(2024, 3, 16, tzinfo=timezone.utc)
