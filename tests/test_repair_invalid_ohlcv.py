import pandas as pd
import pytest

from jobs.repair_invalid_ohlcv import repair_candidates


def _candidate_rows():
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "price_date": "2026-10-01",
                "open": 100.0,
                "high": 99.0,
                "low": 98.0,
                "close": 99.5,
                "adj_close": 99.5,
                "volume": 1000,
                "reasons": '["high_below_ohlc_max"]',
            },
            {
                "ticker": "AAPL",
                "price_date": "2026-10-02",
                "open": 101.0,
                "high": 102.0,
                "low": 100.0,
                "close": 102.5,
                "adj_close": 102.5,
                "volume": 1200,
                "reasons": '["high_below_ohlc_max"]',
            },
        ]
    )


def _provider_history():
    return pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [98.0],
            "Close": [99.5],
            "Volume": [1000],
        },
        index=pd.to_datetime(["2026-10-01"]),
    )


def test_repair_candidates_is_dry_run_by_default_and_only_proposes_valid_bars():
    calls = []

    def fetcher(ticker, start, end):
        calls.append((ticker, start.date(), end.date()))
        return _provider_history()

    result = repair_candidates(_candidate_rows(), fetcher=fetcher)

    assert calls == [("AAPL", pd.Timestamp("2026-10-01").date(), pd.Timestamp("2026-10-03").date())]
    assert result["repair_status"].tolist() == ["would_update", "provider_bar_not_found"]
    assert result.loc[0, "proposed_high"] == 101.0
    assert result.loc[0, "proposed_low"] == 98.0
    assert pd.isna(result.loc[1, "proposed_high"])


def test_repair_candidates_refuses_invalid_provider_replacement():
    bad_history = _provider_history().copy()
    bad_history.loc[:, "High"] = 97.0

    result = repair_candidates(
        _candidate_rows().iloc[[0]],
        fetcher=lambda *_args: bad_history,
    )

    assert result.loc[0, "repair_status"] == "provider_bar_invalid"
    assert "high_below_ohlc_max" in result.loc[0, "repair_error"]



def test_repair_candidates_sends_zero_volume_bars_for_manual_review():
    zero_volume = _provider_history().copy()
    zero_volume.loc[:, "Volume"] = 0

    result = repair_candidates(
        _candidate_rows().iloc[[0]],
        fetcher=lambda *_args: zero_volume,
    )

    assert result.loc[0, "repair_status"] == "manual_review_zero_volume"
    assert result.loc[0, "proposed_volume"] == 0


def test_apply_requires_a_database_client():
    with pytest.raises(ValueError, match="client is required"):
        repair_candidates(_candidate_rows(), apply=True)


class _UpdateQuery:
    def __init__(self, owner, values):
        self.owner = owner
        self.values = values

    def eq(self, column, value):
        self.owner.filters[column] = value
        return self

    def select(self, _columns):
        return self

    def execute(self):
        self.owner.updates.append((self.values, dict(self.owner.filters)))
        return type("Response", (), {
            "data": [{"ticker": self.owner.filters["ticker"],
                      "price_date": self.owner.filters["price_date"]}]
        })()


class _FakeClient:
    def __init__(self):
        self.updates = []
        self.filters = {}

    def table(self, table):
        assert table == "stock_prices"
        return self

    def update(self, values):
        self.filters = {}
        return _UpdateQuery(self, values)


def test_apply_updates_only_audited_rows_with_valid_provider_data():
    client = _FakeClient()
    result = repair_candidates(
        _candidate_rows(),
        fetcher=lambda *_args: _provider_history(),
        client=client,
        apply=True,
    )

    assert result["repair_status"].tolist() == ["updated", "provider_bar_not_found"]
    assert len(client.updates) == 1
    values, filters = client.updates[0]
    assert filters == {"ticker": "AAPL", "price_date": "2026-10-01"}
    assert values == {
        "open": 100.0,
        "high": 101.0,
        "low": 98.0,
        "close": 99.5,
        "adj_close": 99.5,
        "volume": 1000,
    }
