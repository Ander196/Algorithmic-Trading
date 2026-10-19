import pandas as pd
import pytest

from jobs.repair_invalid_ohlcv import (
    apply_approved_report,
    repair_candidates,
)


def _candidate_row(**overrides):
    row = {
        "ticker": "AAPL",
        "price_date": "2026-10-12",
        "open": 100.0,
        "high": 99.0,
        "low": 98.0,
        "close": 99.5,
        "adj_close": 99.5,
        "volume": 1000,
        "reasons": '["high_below_ohlc_max"]',
    }
    row.update(overrides)
    return row


def _audit_rows():
    return pd.DataFrame([_candidate_row()])


def _provider_daily():
    return pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [98.0],
            "Close": [99.5],
            "Volume": [1000],
        },
        index=pd.to_datetime(["2026-10-12"]),
    )


def _unadjusted_daily_history(raw_close=100.0, adjusted_close=100.0):
    return pd.DataFrame(
        {"Close": [raw_close], "Adj Close": [adjusted_close]},
        index=pd.to_datetime(["2026-10-12"]),
    )


def _intraday_history(count=78):
    index = pd.date_range(
        "2026-10-12 09:30",
        periods=count,
        freq="5min",
        tz="America/New_York",
    )
    highs = [101.0] * count
    lows = [99.0] * count
    if count > 20:
        highs[20] = 103.0
    if count > 25:
        lows[25] = 97.0
    return pd.DataFrame(
        {
            "Open": [100.0] * count,
            "High": highs,
            "Low": lows,
            "Close": [99.5] * count,
            "Volume": [10] * count,
        },
        index=index,
    )


def test_provider_daily_preview_is_read_only_and_only_proposes_valid_bars():
    calls = []

    def fetcher(ticker, start, end):
        calls.append((ticker, start.date(), end.date()))
        return _provider_daily()

    result = repair_candidates(_audit_rows(), fetcher=fetcher)

    assert calls == [("AAPL", pd.Timestamp("2026-10-12").date(),
                      pd.Timestamp("2026-10-13").date())]
    assert result.loc[0, "repair_status"] == "would_update"
    assert result.loc[0, "proposed_high"] == 101.0
    assert result.loc[0, "approved"] == False
    assert "high" in result.loc[0, "repair_fields"]


def test_provider_daily_preview_rejects_invalid_provider_bar():
    invalid = _provider_daily().copy()
    invalid.loc[:, "High"] = 97.0

    result = repair_candidates(
        _audit_rows(),
        fetcher=lambda *_args: invalid,
    )

    assert result.loc[0, "repair_status"] == "provider_bar_invalid"
    assert "high_below_ohlc_max" in result.loc[0, "repair_error"]


def test_intraday_preview_changes_only_high_low_and_records_coverage():
    calls = []

    def fetcher(ticker, start, end):
        calls.append((ticker, start.date(), end.date()))
        return _intraday_history()

    result = repair_candidates(
        _audit_rows(),
        mode="intraday-high-low",
        fetcher=fetcher,
        adjustment_fetcher=lambda *_args: _unadjusted_daily_history(),
    )

    assert calls == [("AAPL", pd.Timestamp("2026-10-12").date(),
                      pd.Timestamp("2026-10-13").date())]
    row = result.iloc[0]
    assert row["repair_status"] == "would_update_intraday_high_low"
    assert row["repair_fields"] == "high,low"
    assert row["proposed_high"] == 103.0
    assert row["proposed_low"] == 97.0
    assert row["proposed_open"] == 100.0
    assert row["proposed_close"] == 99.5
    assert row["proposed_adj_close"] == 99.5
    assert row["proposed_volume"] == 1000
    assert row["intraday_bars"] == 78
    assert row["intraday_first_bar"].endswith("09:30:00-04:00")
    assert row["intraday_last_bar"].endswith("15:55:00-04:00")
    assert row["approved"] == False


@pytest.mark.parametrize("count", [0, 20, 69])
def test_intraday_preview_rejects_insufficient_intraday_history(count):
    history = _intraday_history(count) if count else pd.DataFrame()
    result = repair_candidates(
        _audit_rows(),
        mode="intraday-high-low",
        fetcher=lambda *_args: history,
        adjustment_fetcher=lambda *_args: _unadjusted_daily_history(),
    )

    assert result.loc[0, "repair_status"] == "intraday_data_rejected"
    assert "intraday_data_missing" in result.loc[0, "repair_error"] or "insufficient_intraday_bars" in result.loc[0, "repair_error"]


def test_intraday_extrema_are_converted_to_adjusted_price_basis():
    history = _intraday_history()
    history.loc[history.index[20], "High"] = 110.0
    history.loc[history.index[25], "Low"] = 90.0

    result = repair_candidates(
        _audit_rows(),
        mode="intraday-high-low",
        fetcher=lambda *_args: history,
        adjustment_fetcher=lambda *_args: _unadjusted_daily_history(
            raw_close=100.0, adjusted_close=95.0
        ),
    )

    row = result.iloc[0]
    assert row["repair_status"] == "would_update_intraday_high_low"
    assert row["proposed_high"] == pytest.approx(104.5)
    assert row["proposed_low"] == pytest.approx(85.5)
    assert "factor 0.95" in row["repair_error"]


def test_intraday_preview_fails_closed_when_adjustment_factor_is_unavailable():
    result = repair_candidates(
        _audit_rows(),
        mode="intraday-high-low",
        fetcher=lambda *_args: _intraday_history(),
        adjustment_fetcher=lambda *_args: pd.DataFrame(),
    )

    assert result.loc[0, "repair_status"] == "adjustment_factor_unavailable"
    assert "daily adjustment history is missing" in result.loc[0, "repair_error"]


def test_intraday_preview_rejects_invalid_intraday_bars():
    history = _intraday_history()
    history.loc[history.index[10], "High"] = 90.0

    result = repair_candidates(
        _audit_rows(),
        mode="intraday-high-low",
        fetcher=lambda *_args: history,
    )

    assert result.loc[0, "repair_status"] == "intraday_data_rejected"
    assert "invalid_intraday_high_low_geometry" in result.loc[0, "repair_error"]


def test_intraday_preview_refuses_non_geometry_defects():
    candidate = _audit_rows()
    candidate.loc[0, "volume"] = 0

    result = repair_candidates(
        candidate,
        mode="intraday-high-low",
        fetcher=lambda *_args: _intraday_history(),
        adjustment_fetcher=lambda *_args: _unadjusted_daily_history(),
    )

    assert result.loc[0, "repair_status"] == "manual_review_existing_zero_volume"


def test_duplicate_ticker_date_rows_are_rejected():
    candidates = pd.concat([_audit_rows(), _audit_rows()], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate ticker/date"):
        repair_candidates(candidates, mode="intraday-high-low", fetcher=lambda *_: _intraday_history())


class _FakeQuery:
    def __init__(self, client):
        self.client = client
        self.operation = "select"
        self.values = {}
        self.filters = {}
        self.selected = "*"

    def select(self, columns):
        self.selected = columns
        return self

    def update(self, values):
        self.operation = "update"
        self.values = dict(values)
        return self

    def eq(self, column, value):
        self.filters[column] = value
        return self

    def is_(self, column, value):
        self.filters[column] = None if str(value).lower() == "null" else value
        return self

    def limit(self, _number):
        return self

    def execute(self):
        matches = [
            row for row in self.client.rows
            if all(row.get(column) == value for column, value in self.filters.items())
        ]
        if self.operation == "select":
            return type("Response", (), {"data": [dict(row) for row in matches]})()

        if not matches:
            return type("Response", (), {"data": []})()
        if len(matches) != 1:
            return type("Response", (), {"data": matches})()

        matches[0].update(self.values)
        self.client.updates.append(
            (dict(self.values), dict(self.filters))
        )
        return type("Response", (), {
            "data": [{"ticker": matches[0]["ticker"], "price_date": matches[0]["price_date"]}]
        })()


class _FakeClient:
    def __init__(self, rows=None):
        self.rows = [dict(row) for row in (rows or [])]
        self.updates = []

    def table(self, table_name):
        assert table_name == "stock_prices"
        return _FakeQuery(self)


def _approved_intraday_report(approved=True):
    preview = repair_candidates(
        _audit_rows(),
        mode="intraday-high-low",
        fetcher=lambda *_args: _intraday_history(),
        adjustment_fetcher=lambda *_args: _unadjusted_daily_history(),
    )
    preview.loc[0, "approved"] = approved
    return preview


def test_apply_requires_explicit_approval():
    client = _FakeClient([_candidate_row()])

    with pytest.raises(ValueError, match="approved=True"):
        apply_approved_report(_approved_intraday_report(False), client)

    assert client.updates == []


def test_apply_uses_preview_values_and_only_updates_high_low():
    client = _FakeClient([_candidate_row()])

    result = apply_approved_report(_approved_intraday_report(True), client)

    assert result.loc[0, "repair_status"] == "updated"
    assert len(client.updates) == 1
    values, _filters = client.updates[0]
    assert values == {"high": 103.0, "low": 97.0}
    assert client.rows[0]["open"] == 100.0
    assert client.rows[0]["close"] == 99.5
    assert client.rows[0]["adj_close"] == 99.5
    assert client.rows[0]["volume"] == 1000


def test_apply_refuses_stale_database_snapshot():
    changed = _candidate_row()
    changed["close"] = 99.6
    client = _FakeClient([changed])

    result = apply_approved_report(_approved_intraday_report(True), client)

    assert result.loc[0, "repair_status"] == "stale_preview_snapshot"
    assert client.updates == []


def test_apply_rejects_manual_review_rows_even_if_approved():
    report = _approved_intraday_report(True)
    report.loc[0, "repair_status"] = "intraday_data_rejected"
    client = _FakeClient([_candidate_row()])

    result = apply_approved_report(report, client)

    assert result.loc[0, "repair_status"] == "approval_rejected"
    assert client.updates == []


def test_apply_refuses_invalid_approved_values():
    report = _approved_intraday_report(True)
    report.loc[0, "proposed_high"] = 90.0
    client = _FakeClient([_candidate_row()])

    result = apply_approved_report(report, client)

    assert result.loc[0, "repair_status"] == "approved_values_invalid"
    assert client.updates == []


class _RacingQuery(_FakeQuery):
    def execute(self):
        response = super().execute()
        if self.operation == "select" and not self.client.raced:
            self.client.raced = True
            # Simulate another writer changing the database after the snapshot read.
            self.client.rows[0]["close"] = 99.6
        return response


class _RacingClient(_FakeClient):
    def __init__(self, rows=None):
        super().__init__(rows)
        self.raced = False

    def table(self, table_name):
        assert table_name == "stock_prices"
        return _RacingQuery(self)


def test_apply_conditional_update_refuses_change_between_select_and_update():
    client = _RacingClient([_candidate_row()])

    result = apply_approved_report(_approved_intraday_report(True), client)

    assert result.loc[0, "repair_status"] == "update_not_confirmed"
    assert client.updates == []
    assert client.rows[0]["close"] == 99.6


def test_intraday_preview_rejects_large_gaps_even_when_bar_count_is_sufficient():
    history = _intraday_history().drop(_intraday_history().index[30:33])

    result = repair_candidates(
        _audit_rows(),
        mode="intraday-high-low",
        fetcher=lambda *_args: history,
        adjustment_fetcher=lambda *_args: _unadjusted_daily_history(),
    )

    assert result.loc[0, "repair_status"] == "intraday_data_rejected"
    assert "intraday_gaps_exceed_15_minutes" in result.loc[0, "repair_error"]
