from __future__ import annotations

import pandas as pd
import pytest

from jobs.export_alpha_adjusted_prices import _currency_scale, _fetch_adjusted


@pytest.mark.parametrize(
    "ticker,currency,scale",
    [
        ("ASML.AS", "EUR", 1.0),
        ("SAP.DE", "EUR", 1.0),
        ("SHEL.L", "GBP", 0.01),
        ("NESN.SW", "CHF", 1.0),
        ("ERIC-B.SE", "SEK", 1.0),
        ("AAPL", "USD", 1.0),
        ("SPY", "USD", 1.0),
    ],
)
def test_currency_scale_maps_supported_listing_suffixes(
    ticker: str, currency: str, scale: float
) -> None:
    assert _currency_scale(ticker) == (currency, scale)


class _FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def select(self, *_args):
        return self

    def eq(self, *_args):
        return self

    def gte(self, *_args):
        return self

    def order(self, *_args):
        return self

    def range(self, start, end):
        self.rows = self.rows[start : end + 1]
        return self

    def execute(self):
        return type("Response", (), {"data": self.rows})()


class _FakeClient:
    def __init__(self, rows):
        self.rows = rows

    def table(self, _name):
        return _FakeQuery(self.rows)


def test_fetch_adjusted_reads_adj_close_and_returns_chronological_series() -> None:
    client = _FakeClient([
        {"price_date": "2024-01-03", "adj_close": 12.0},
        {"price_date": "2024-01-02", "adj_close": 11.0},
    ])

    series = _fetch_adjusted(client, "AAA", "2024-01-01")

    assert series.index.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert series.tolist() == [11.0, 12.0]


def test_fetch_adjusted_fails_clearly_when_history_is_missing() -> None:
    with pytest.raises(ValueError, match="No adj_close history"):
        _fetch_adjusted(_FakeClient([]), "MISSING", "2024-01-01")
