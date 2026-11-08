from __future__ import annotations

import pandas as pd

from jobs.audit_alpha_missing_history import audit_missing_ticker


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

    assert calls == ["CIT.A", "CIT-A"]
    assert result["primary_sessions"] == 0
    assert result["alias_sessions"] == 2
    assert result["status"] == "possible_share_class_alias_identity_review"


def test_empty_history_is_not_misclassified_as_a_provider_error():
    result = audit_missing_ticker(
        "AAMRQ",
        history_fetcher=lambda symbol, start, end: pd.DataFrame(),
        delay_seconds=0,
        sleeper=lambda _: None,
    )

    assert result["primary_requests"] == 1
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

    assert result["primary_requests"] == 0
    assert result["primary_errors"] == 2
    assert result["status"] == "primary_provider_error"


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
        ("2011-03-01", "2015-01-27"),
    ]
    assert result["membership_period_count"] == 2
