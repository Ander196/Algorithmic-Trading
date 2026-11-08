from __future__ import annotations

import pandas as pd
import pytest
from yfinance.exceptions import YFPricesMissingError

from jobs.audit_alpha_missing_history import audit_missing_history, audit_missing_ticker


def _history(dates: list[str]) -> pd.DataFrame:
    index = pd.DatetimeIndex(dates)
    return pd.DataFrame(
        {
            "Close": [10.0 + i for i in range(len(index))],
            "Adj Close": [9.0 + i for i in range(len(index))],
            "Volume": [1000] * len(index),
            "Dividends": [0.0] * len(index),
            "Stock Splits": [0.0] * len(index),
        },
        index=index,
    )


def test_original_yahoo_symbol_is_only_a_backfill_candidate_not_approved():
    calls = []

    def fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
        calls.append((symbol, start, end))
        return _history(["2000-01-03", "2000-01-04"])

    result = audit_missing_ticker(
        "AABA",
        history_fetcher=fetch,
        delay_seconds=0,
        sleeper=lambda _: None,
    )

    assert result["primary_yahoo_symbol"] == "AABA"
    assert result["primary_sessions"] == 2
    assert result["status"] == "primary_history_available_identity_review"
    assert "confirm issuer/security identity" in result["identity_warning"]
    assert result["alias_sessions"] == 0
    assert calls == [("AABA", "2000-01-01", "2017-06-19")]


def test_dot_to_hyphen_share_class_spelling_is_flagged_as_candidate_only():
    calls = []

    def fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
        calls.append(symbol)
        if symbol == "CIT.A":
            return pd.DataFrame()
        if symbol == "CIT-A":
            return _history(["2000-07-17", "2000-07-18"])
        raise AssertionError(symbol)

    result = audit_missing_ticker(
        "CIT.A",
        history_fetcher=fetch,
        delay_seconds=0,
        sleeper=lambda _: None,
    )

    assert calls == ["CIT.A", "CIT.A", "CIT-A"]
    assert result["primary_sessions"] == 0
    assert result["alias_sessions"] == 2
    assert result["status"] == "possible_share_class_alias_identity_review"


def test_yfinance_no_data_exception_is_retried_and_recorded_as_empty_response():
    calls = []

    def fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
        calls.append(symbol)
        raise YFPricesMissingError(symbol, f"(1d {start} -> {end})")

    result = audit_missing_ticker(
        "AAMRQ",
        history_fetcher=fetch,
        retries=2,
        delay_seconds=0,
        sleeper=lambda _: None,
    )

    assert calls == ["AAMRQ", "AAMRQ"]
    assert result["primary_requests"] == 2
    assert result["primary_empty_responses"] == 2
    assert result["primary_errors"] == 0
    assert result["status"] == "no_history_for_tested_yahoo_symbols"


def test_provider_errors_remain_unclassified_and_are_not_treated_as_no_history():
    def fail(symbol: str, start: str, end: str) -> pd.DataFrame:
        raise RuntimeError("temporary provider failure")

    result = audit_missing_ticker(
        "AAMRQ",
        history_fetcher=fail,
        retries=2,
        delay_seconds=0,
        sleeper=lambda _: None,
    )

    assert result["primary_requests"] == 2
    assert result["primary_empty_responses"] == 0
    assert result["primary_errors"] == 2
    assert result["status"] == "provider_errors_no_conclusion"


def test_repeated_membership_windows_are_probed_separately():
    calls = []

    def fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
        calls.append((start, end))
        return _history([])

    result = audit_missing_ticker(
        "COV",
        history_fetcher=fetch,
        delay_seconds=0,
        sleeper=lambda _: None,
    )

    assert calls == [
        ("2007-07-02", "2009-06-05"),
        ("2007-07-02", "2009-06-05"),
        ("2011-03-01", "2015-01-27"),
        ("2011-03-01", "2015-01-27"),
    ]
    assert result["membership_period_count"] == 2


def test_full_audit_requires_a_working_control_ticker_before_probing_missing_symbols(
    tmp_path,
):
    tickers_csv = tmp_path / "tickers.csv"
    pd.DataFrame({"ticker": ["AABA"]}).to_csv(tickers_csv, index=False)
    calls = []

    def fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
        calls.append(symbol)
        if symbol == "AAPL":
            return pd.DataFrame()
        raise AssertionError("Missing symbols must not be queried when control fails")

    with pytest.raises(RuntimeError, match="Yahoo control check returned no valid observations"):
        audit_missing_history(
            tickers_csv,
            tmp_path / "output",
            history_fetcher=fetch,
            delay_seconds=0,
            control_ticker="AAPL",
        )

    assert calls == ["AAPL"]


def test_full_audit_writes_control_provenance_and_summary(tmp_path):
    tickers_csv = tmp_path / "tickers.csv"
    pd.DataFrame({"ticker": ["AABA"]}).to_csv(tickers_csv, index=False)

    def fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
        if symbol == "AAPL":
            return _history(["2026-10-01", "2026-10-02"])
        return pd.DataFrame()

    output_dir = tmp_path / "output"
    counts = audit_missing_history(
        tickers_csv,
        output_dir,
        history_fetcher=fetch,
        delay_seconds=0,
        control_ticker="AAPL",
    )

    control = pd.read_csv(output_dir / "alpha_missing_history_control.csv")
    summary = pd.read_csv(output_dir / "alpha_missing_history_summary.csv")

    assert counts["control_sessions"] == 2
    assert control.loc[0, "control_ticker"] == "AAPL"
    assert control.loc[0, "status"] == "passed"
    assert summary.loc[0, "status"] == "no_history_for_tested_yahoo_symbols"
