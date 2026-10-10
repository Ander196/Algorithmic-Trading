import pandas as pd
import pytest

from jobs.common import fetch_price_history


class _FakeQuery:
    def __init__(self, rows, ranges):
        self.rows = rows
        self.ranges = ranges
        self.start = 0
        self.end = -1

    def select(self, *_args):
        return self

    def eq(self, *_args):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def range(self, start, end):
        self.start = start
        self.end = end
        self.ranges.append((start, end))
        return self

    def execute(self):
        return type("Response", (), {"data": self.rows[self.start : self.end + 1]})()


class _FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.ranges = []

    def table(self, _name):
        return _FakeQuery(self.rows, self.ranges)


def _rows(count):
    dates = pd.date_range("2000-01-01", periods=count, freq="D")
    # Supabase is queried newest-first; the function must return chronological order.
    return [
        {
            "price_date": date.isoformat(),
            "open": float(i + 1),
            "high": float(i + 2),
            "low": max(0.5, float(i)),
            "close": float(i + 1.5),
            "volume": float(100 + i),
        }
        for i, date in reversed(list(enumerate(dates)))
    ]


def test_fetch_price_history_none_loads_all_pages_in_chronological_order():
    rows = _rows(2205)
    client = _FakeClient(rows)

    frame = fetch_price_history(client, "aapl", limit=None)

    assert len(frame) == 2205
    assert frame["price_date"].is_monotonic_increasing
    assert frame["price_date"].iloc[0] == pd.Timestamp("2000-01-01", tz="UTC")
    assert frame["price_date"].iloc[-1] == pd.Timestamp("2006-01-13", tz="UTC")
    assert client.ranges == [(0, 999), (1000, 1999), (2000, 2999)]


def test_fetch_price_history_default_preserves_1500_row_cap():
    client = _FakeClient(_rows(2001))

    frame = fetch_price_history(client, "AAPL")

    assert len(frame) == 1500
    assert frame["price_date"].iloc[0] == pd.Timestamp("2001-05-16", tz="UTC")
    assert frame["price_date"].iloc[-1] == pd.Timestamp("2005-06-23", tz="UTC")
    assert client.ranges == [(0, 999), (1000, 1499)]


def test_fetch_price_history_limit_caps_rows_and_keeps_latest_observations():
    client = _FakeClient(_rows(2001))

    frame = fetch_price_history(client, "AAPL", limit=1500)

    assert len(frame) == 1500
    assert frame["price_date"].iloc[0] == pd.Timestamp("2001-05-16", tz="UTC")
    assert frame["price_date"].iloc[-1] == pd.Timestamp("2005-06-23", tz="UTC")
    assert client.ranges == [(0, 999), (1000, 1499)]


@pytest.mark.parametrize("limit", [0, -1, -1500])
def test_fetch_price_history_rejects_non_positive_limit(limit):
    with pytest.raises(ValueError, match="limit must be greater than zero"):
        fetch_price_history(_FakeClient(_rows(1)), "AAPL", limit=limit)


def test_fetch_price_history_rejects_excluded_tickers_before_querying():
    client = _FakeClient(_rows(5))

    with pytest.raises(RuntimeError, match="ABI.BR is excluded"):
        fetch_price_history(client, " abi.br ", limit=None)

    assert client.ranges == []


def test_fetch_price_history_raises_when_ticker_has_no_history():
    with pytest.raises(RuntimeError, match="No price history found for AAPL"):
        fetch_price_history(_FakeClient([]), "AAPL", limit=None)
