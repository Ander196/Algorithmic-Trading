from types import SimpleNamespace

import pandas as pd
import pytest

from data import data_loader_historical as historical_loader
from data import ticker_loader
from data.sp500_universe import (
    SP500_BENCHMARK_TICKER,
    get_current_sp500_tickers,
    get_sp500_tickers,
)


def test_ticker_catalog_contains_every_historical_constituent_and_spy(monkeypatch):
    def fake_fetch_all(tickers, **_kwargs):
        return pd.DataFrame([
            {
                "ticker": ticker,
                "name": ticker,
                "exchange": "TEST",
                "sector": "Test",
                "industry": "Test",
            }
            for ticker in tickers
        ])

    monkeypatch.setattr(ticker_loader, "fetchAllTickers", fake_fetch_all)
    catalog = ticker_loader.buildTickerCatalog()

    expected = set(get_sp500_tickers()) | {SP500_BENCHMARK_TICKER}
    assert set(catalog["ticker"]) == expected
    assert len(catalog) == len(expected)
    assert len(expected) > 1_100

    current = set(get_current_sp500_tickers())
    actual_current = set(
        catalog.loc[~catalog["historical_only"], "ticker"]
    )
    assert actual_current == current | {SP500_BENCHMARK_TICKER}

    former = set(catalog.loc[catalog["historical_only"], "ticker"])
    assert {"APC", "SBNY", "QRVO", "WBD"}.issubset(former)
    assert "QQQ" not in expected
    assert "ULVR.L" not in expected


def test_fetch_all_tickers_keeps_symbols_even_when_yahoo_metadata_is_missing(monkeypatch):
    def fake_fetch(ticker):
        return {
            "ticker": ticker,
            "name": ticker,
            "exchange": "Unknown",
            "sector": "Unknown",
            "industry": "Unknown",
        }

    monkeypatch.setattr(ticker_loader, "fetchTickerInfo", fake_fetch)

    result = ticker_loader.fetchAllTickers(
        ["NOQUOTE", "AAPL"],
        delay=0,
        batchSize=1,
    )
    assert set(result["ticker"]) == {"NOQUOTE", "AAPL"}


def test_historical_loader_includes_full_snp_universe_and_spy():
    tickers = historical_loader.getActiveTickers(object())
    assert set(tickers) == set(get_sp500_tickers()) | {SP500_BENCHMARK_TICKER}


def test_upload_keeps_former_members_inactive_and_does_not_reset_created_at():
    class FakeQuery:
        def __init__(self, client):
            self.client = client
            self.ticker = None

        def select(self, *_args):
            return self

        def execute(self):
            return SimpleNamespace(data=self.client.existing_rows)

        def upsert(self, records, on_conflict=None):
            self.client.upserted_records.extend(records)
            return self

        def update(self, record):
            self.client.updated_records.append(record)
            return self

        def insert(self, records):
            self.client.inserted_records.extend(records)
            return self

        def eq(self, column, value):
            self.ticker = value
            return self

    class FakeClient:
        def __init__(self):
            self.existing_rows = [{"ticker": "AAPL", "is_active": True}]
            self.upserted_records = []
            self.updated_records = []
            self.inserted_records = []

        def table(self, _name):
            return FakeQuery(self)

    client = FakeClient()
    frame = pd.DataFrame([
        {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "exchange": "NMS",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "historical_only": False,
        },
        {
            "ticker": "SPLK",
            "name": "Splunk",
            "exchange": "Unknown",
            "sector": "Unknown",
            "industry": "Unknown",
            "historical_only": True,
        },
    ])

    ticker_loader.uploadToSupabase(frame, client)

    current = client.upserted_records[0]
    retired = client.inserted_records[0]
    assert current["ticker"] == "AAPL"
    assert current["is_active"] is True
    assert "created_at" in current
    assert retired["ticker"] == "SPLK"
    assert retired["is_active"] is False
    assert "created_at" not in retired


@pytest.mark.parametrize("module", [ticker_loader, historical_loader])
def test_supabase_clients_require_privileged_key(monkeypatch, module):
    captured = {}

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setenv("SUPABASE_ANON_KEY", "public-anon-key")

    with pytest.raises(ValueError, match="SUPABASE_KEY"):
        module.getSupabaseClient()

    monkeypatch.setenv("SUPABASE_KEY", "sb_test_only")

    def fake_create_client(url, key):
        captured["url"] = url
        captured["key"] = key
        return "fake-client"

    monkeypatch.setattr(module.supabase, "create_client", fake_create_client)

    assert module.getSupabaseClient() == "fake-client"
    assert captured == {
        "url": "https://example.supabase.co",
        "key": "sb_test_only",
    }
