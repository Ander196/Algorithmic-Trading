from datetime import datetime, timezone

import pandas as pd

from data import data_loader


def test_fetch_price_data_requests_adjusted_ohlc_and_uses_same_adj_close(monkeypatch) -> None:
    history = pd.DataFrame(
        {
            "Open": [10.0],
            "High": [12.0],
            "Low": [9.0],
            "Close": [11.0],
            "Volume": [100],
        },
        index=pd.DatetimeIndex(["2026-10-15"]),
    )
    calls = []

    class FakeTicker:
        def history(self, **kwargs):
            calls.append(kwargs)
            return history.copy()

    monkeypatch.setattr(data_loader.yf, "Ticker", lambda ticker: FakeTicker())

    result = data_loader.fetchPriceData(
        "AAA",
        datetime(2026, 10, 15, tzinfo=timezone.utc),
        datetime(2026, 10, 17, tzinfo=timezone.utc),
    )

    assert result is not None
    assert len(result) == 1
    assert calls[0]["auto_adjust"] is True
    assert calls[0]["actions"] is False
    assert result.loc[0, "close"] == 11.0
    assert result.loc[0, "adj_close"] == 11.0
