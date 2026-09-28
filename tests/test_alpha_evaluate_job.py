import pandas as pd
import pytest

from jobs.evaluate_alpha import _to_ohlcv_frame, get_alpha_universe


class _FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def select(self, *_args):
        return self

    def eq(self, *_args):
        return self

    def execute(self):
        return type("Response", (), {"data": self.rows})()


class _FakeClient:
    def __init__(self, rows):
        self.rows = rows

    def table(self, _name):
        return _FakeQuery(self.rows)


def test_get_alpha_universe_uses_active_tickers_and_excludes_market() -> None:
    client = _FakeClient(
        [
            {"ticker": "spy"},
            {"ticker": "AAA"},
            {"ticker": "bbb"},
            {"ticker": "AAA"},
            {"ticker": None},
        ]
    )

    assert get_alpha_universe(client, "SPY") == ["AAA", "BBB"]


def test_to_ohlcv_frame_normalizes_datetime_and_columns() -> None:
    raw = pd.DataFrame(
        {
            "price_date": ["2024-01-03T00:00:00+00:00", "2024-01-02T00:00:00+00:00"],
            "open": [11, 10],
            "high": [12, 11],
            "low": [9, 9],
            "close": [11.5, 10.5],
            "volume": [1000, 900],
        }
    )

    frame = _to_ohlcv_frame(raw)

    assert frame.index.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]


def test_to_ohlcv_frame_rejects_missing_columns() -> None:
    raw = pd.DataFrame(
        {
            "price_date": ["2024-01-02"],
            "close": [10],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        _to_ohlcv_frame(raw)
