from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.alpha_price_anomaly_audit import compare_adjusted_price_histories


def _history(prices: list[float]) -> pd.DataFrame:
    values = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {
            "open": values,
            "high": values * 1.01,
            "low": values * 0.99,
            "close": values,
            "adj_close": values,
            "volume": np.full(len(values), 1000),
        },
        index=pd.bdate_range("2011-05-23", periods=len(values)),
    )


def test_comparison_flags_cpwr_like_crash_rebound_and_basis_break():
    reference = _history([10.0, 10.1, 10.2, 10.3, 10.4, 10.5])
    stored = _history([10.0, 10.1, 1.02, 4.896, 10.4, 10.5])

    comparison, summary = compare_adjusted_price_histories(
        stored, reference, ticker="cpwr"
    )

    flagged = comparison.loc[comparison["flags"].ne("")].set_index("date")
    crash_date = pd.Timestamp("2011-05-25")
    rebound_date = pd.Timestamp("2011-05-26")

    assert "large_stored_observation_return" in flagged.loc[crash_date, "flags"]
    assert "large_stored_observation_return" in flagged.loc[rebound_date, "flags"]
    assert "adjusted_close_basis_deviation" in flagged.loc[crash_date, "flags"]
    assert summary["ticker"] == "CPWR"
    assert summary["overlap_rows"] == 6
    assert summary["flagged_sessions"] >= 2


def test_constant_provider_price_scale_is_not_misclassified_as_time_varying():
    reference = _history([10.0, 10.1, 10.2, 10.3, 10.4, 10.5])
    stored = reference.copy()
    for column in ("open", "high", "low", "close", "adj_close"):
        stored[column] = stored[column] * 100.0

    comparison, summary = compare_adjusted_price_histories(
        stored, reference, ticker="CPWR"
    )

    assert np.isclose(summary["calibrated_scale_factor"], 100.0)
    assert summary["p95_relative_scale_dispersion"] < 1e-12
    assert not comparison["flags"].str.contains(
        "adjusted_close_basis_deviation"
    ).any()
    assert not comparison["flags"].str.contains(
        "large_stored_observation_return"
    ).any()


def test_duplicate_dates_and_bad_thresholds_are_rejected():
    history = _history([10.0, 10.1])
    duplicate = pd.concat([history, history.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate session dates"):
        compare_adjusted_price_histories(duplicate, history)

    with pytest.raises(ValueError, match="large_return_threshold"):
        compare_adjusted_price_histories(
            history, history, large_return_threshold=0
        )
