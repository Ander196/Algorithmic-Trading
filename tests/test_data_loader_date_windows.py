from datetime import datetime, timezone

import pytest

from data import data_loader


@pytest.mark.parametrize(
    ("now_utc", "expected_end"),
    [
        (datetime(2026, 10, 19, 19, 5, tzinfo=timezone.utc), datetime(2026, 10, 19)),
        (datetime(2026, 10, 19, 22, 30, tzinfo=timezone.utc), datetime(2026, 10, 20)),
    ],
)
def test_process_ticker_uses_naive_exchange_date_boundaries(
    monkeypatch, now_utc, expected_end
):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            current = now_utc.astimezone(tz) if tz is not None else now_utc.replace(tzinfo=None)
            return cls(
                current.year,
                current.month,
                current.day,
                current.hour,
                current.minute,
                current.second,
                current.microsecond,
                tzinfo=current.tzinfo,
                fold=current.fold,
            )

    latest_date = datetime(2026, 10, 16, tzinfo=timezone.utc)
    observed = {}

    def fake_fetch_price_data(ticker, start_date, end_date):
        observed["start"] = start_date
        observed["end"] = end_date
        return None

    monkeypatch.setattr(data_loader, "datetime", FrozenDatetime)
    monkeypatch.setattr(data_loader, "getLatestPriceDate", lambda _client, _ticker: latest_date)
    monkeypatch.setattr(data_loader, "fetchPriceData", fake_fetch_price_data)

    result = data_loader.processTicker(object(), "AAPL")

    assert result["error"] is None
    assert result["skipped"] == 0
    assert observed["start"] == datetime(2026, 10, 17)
    assert observed["end"] == expected_end
    assert observed["start"].tzinfo is None
    assert observed["end"].tzinfo is None
