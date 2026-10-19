from datetime import datetime, timezone

import pandas as pd
import pytest

from data import data_loader


class FakeResponse:
    data = []


class FakeTable:
    def __init__(self, failures):
        self.failures = failures
        self.calls = []

    def insert(self, payload):
        self.calls.append(payload)
        return self

    def execute(self):
        payload = self.calls[-1]
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            error = self.failures.get(record["price_date"])
            if error:
                raise error
        return FakeResponse()


class FakeClient:
    def __init__(self, failures):
        self.fake_table = FakeTable(failures)

    def table(self, name):
        assert name == "stock_prices"
        return self.fake_table


def sample_frame():
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "price_date": ["2026-10-15", "2026-10-16"],
            "open": [10.0, 11.0],
            "high": [12.0, 13.0],
            "low": [9.0, 10.0],
            "close": [11.0, 12.0],
            "adj_close": [11.0, 12.0],
            "volume": [100, 200],
        }
    )


def test_upload_counts_duplicate_rows_as_existing_not_as_inserted(monkeypatch):
    monkeypatch.setattr(data_loader, "datetime", datetime)
    client = FakeClient(
        {"2026-10-15": Exception('duplicate key value violates unique constraint "uq_stock_prices_ticker_date" (23505)')}
    )

    inserted = data_loader.uploadToSupabase(client, "AAA", sample_frame())

    assert inserted == 1


def test_upload_raises_when_non_duplicate_row_insert_fails():
    client = FakeClient({"2026-10-15": RuntimeError("database unavailable")})

    with pytest.raises(data_loader.PriceUploadError, match="failed to insert 1 row"):
        data_loader.uploadToSupabase(client, "AAA", sample_frame())


def test_duplicate_detection_uses_postgres_code():
    error = RuntimeError("unique violation")
    error.code = "23505"

    assert data_loader._is_duplicate_key_error(error)
    assert not data_loader._is_duplicate_key_error(RuntimeError("database unavailable"))


def test_fetch_price_data_rejects_dates_outside_requested_range(monkeypatch):
    history = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [12.0, 13.0],
            "Low": [9.0, 10.0],
            "Close": [11.0, 12.0],
            "Volume": [100, 200],
        },
        index=pd.DatetimeIndex(["2026-10-14", "2026-10-17"]),
    )

    class FakeTicker:
        def history(self, **kwargs):
            return history.copy()

    monkeypatch.setattr(data_loader.yf, "Ticker", lambda ticker: FakeTicker())
    monkeypatch.setattr(data_loader.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="failed to fetch/validate"):
        data_loader.fetchPriceData(
            "AAA",
            datetime(2026, 10, 15, tzinfo=timezone.utc),
            datetime(2026, 10, 17, tzinfo=timezone.utc),
            retries=1,
        )
