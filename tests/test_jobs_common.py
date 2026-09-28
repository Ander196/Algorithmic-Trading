import pandas as pd
import pytest

from jobs.common import fetch_price_history


class _FakeQuery:
    def __init__(self, pages):
        self.pages = pages
        self.range_args = []

    def select(self, *_args):
        return self

    def eq(self, *_args):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def range(self, start, end):
        self.range_args.append((start, end))
        return self

    def execute(self):
        start, end = self.range_args.pop(0)
        page = self.pages.get((start, end), [])
        return type("Response", (), {"data": page})()


class _FakeClient:
    def __init__(self, pages):
        self.query = _FakeQuery(pages)

    def table(self, _name):
        return self.query


def _row(day: str, close: float) -> dict:
    return {
        "price_date": day,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 1000,
    }


def test_fetch_price_history_none_reads_all_pages() -> None:
    first = [_row("2024-01-03", 3)]
    second = [_row("2024-01-02", 2)]

    client = _FakeClient(
        {
            (0, 999): first + second,
        }
    )

    result = fetch_price_history(client, "AAA", limit=None)

    assert len(result) == 2
    assert result["price_date"].is_monotonic_increasing
    assert result["close"].tolist() == [2, 3]


def test_fetch_price_history_rejects_non_positive_limit() -> None:
    client = _FakeClient({})

    with pytest.raises(ValueError, match="greater than zero"):
        fetch_price_history(client, "AAA", limit=0)
