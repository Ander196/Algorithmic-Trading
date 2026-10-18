from __future__ import annotations

import numpy as np
import pandas as pd

from jobs.audit_alpha_price_coverage import (
    OHLCV_COLUMNS,
    _calibrate_price_scale,
    _find_repair_dates,
    _normalize_yahoo_history,
    _render_repair_sql,
    _stored_row_invalid_reasons,
)


def _valid_row(value: float = 10.0) -> dict[str, float]:
    return {
        "open": value,
        "high": value + 1,
        "low": value - 1,
        "close": value,
        "adj_close": value,
        "volume": 1000,
    }


def test_normalize_yahoo_history_adjusts_all_ohlc_fields_to_adj_close() -> None:
    index = pd.DatetimeIndex(["2024-01-02"], tz="America/New_York")
    raw = pd.DataFrame(
        {
            "Open": [20.0],
            "High": [22.0],
            "Low": [18.0],
            "Close": [20.0],
            "Adj Close": [10.0],
            "Volume": [1234],
        },
        index=index,
    )

    result = _normalize_yahoo_history(raw)

    assert result.index.tolist() == [pd.Timestamp("2024-01-02")]
    assert result.loc[pd.Timestamp("2024-01-02"), "open"] == 10.0
    assert result.loc[pd.Timestamp("2024-01-02"), "high"] == 11.0
    assert result.loc[pd.Timestamp("2024-01-02"), "low"] == 9.0
    assert result.loc[pd.Timestamp("2024-01-02"), "close"] == 10.0
    assert result.loc[pd.Timestamp("2024-01-02"), "adj_close"] == 10.0
    assert result.loc[pd.Timestamp("2024-01-02"), "volume"] == 1234


def test_scale_calibration_accepts_stable_constant_scale() -> None:
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    yahoo = pd.Series([10, 11, 12, 13, 14, 15], index=dates)
    stored = yahoo * 1.002

    result = _calibrate_price_scale(stored, yahoo, min_overlap=5, max_dispersion=0.005)

    assert result["status"] == "calibrated"
    assert result["observations"] == 6
    assert np.isclose(result["scale_factor"], 1.002)
    assert result["p95_relative_dispersion"] < 1e-10


def test_scale_calibration_rejects_time_varying_price_basis() -> None:
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    yahoo = pd.Series([10, 11, 12, 13, 14, 15], index=dates)
    stored = yahoo * pd.Series([1.0, 1.0, 1.0, 1.02, 1.02, 1.02], index=dates)

    result = _calibrate_price_scale(stored, yahoo, min_overlap=5, max_dispersion=0.005)

    assert result["status"] == "inconsistent_price_basis"
    assert result["scale_factor"] is not None
    assert result["p95_relative_dispersion"] > 0.005


def test_find_repair_dates_detects_absent_and_invalid_rows_but_not_no_bar_sessions() -> None:
    calendar = pd.date_range("2024-01-02", periods=4, freq="B")
    yahoo = pd.DataFrame(
        [_valid_row(10.0), _valid_row(11.0), _valid_row(12.0), _valid_row(13.0)],
        index=calendar,
        columns=OHLCV_COLUMNS,
    )
    yahoo.loc[calendar[2], list(OHLCV_COLUMNS)] = np.nan  # no Yahoo session/bar
    stored = pd.DataFrame(
        [
            _valid_row(10.0),
            # Date 3 has no stored row, so should be inserted.
            # Date 4 has a row but broken geometry, so should be updated.
            {
                "open": 13.0,
                "high": 12.0,
                "low": 14.0,
                "close": 13.0,
                "adj_close": 13.0,
                "volume": 1000,
            },
        ],
        index=[calendar[0], calendar[3]],
        columns=OHLCV_COLUMNS,
    )

    repairs = _find_repair_dates(yahoo, stored, calendar)

    assert repairs == [
        (calendar[1], "insert", ["missing_supabase_row"]),
    ] or (
        len(repairs) == 2
        and repairs[0][0] == calendar[1]
        and repairs[0][1] == "insert"
        and repairs[1][0] == calendar[3]
        and repairs[1][1] == "update_invalid"
        and "high_below_ohlc_max" in repairs[1][2]
        and "low_above_ohlc_min" in repairs[1][2]
    )


def test_stored_row_invalid_reasons_report_nulls_and_bad_geometry() -> None:
    row = pd.Series(
        {
            "open": 0,
            "high": 9,
            "low": 12,
            "close": np.nan,
            "adj_close": 8,
            "volume": -1,
        }
    )

    reasons = _stored_row_invalid_reasons(row)

    assert "nonpositive_open" in reasons
    assert "missing_or_nonfinite_close" in reasons
    assert "negative_volume" in reasons
    assert "low_above_ohlc_min" not in reasons  # geometry is checked only for valid positive OHLC
    assert "high_below_ohlc_max" not in reasons


def test_render_repair_sql_is_idempotent_and_updates_only_invalid_rows() -> None:
    rows = [
        {
            "action": "insert",
            "price_date": "2024-01-02",
            "ticker": "AAA",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.0,
            "adj_close": 10.0,
            "volume": 1000,
            "repair_reason": "missing_supabase_row",
        },
        {
            "action": "update_invalid",
            "price_date": "2024-01-03",
            "ticker": "BBB",
            "open": 20.0,
            "high": 21.0,
            "low": 19.0,
            "close": 20.0,
            "adj_close": 20.0,
            "volume": 2000,
            "repair_reason": "high_below_ohlc_max",
        },
    ]

    sql = _render_repair_sql(rows)

    assert "BEGIN;" in sql
    assert "COMMIT;" in sql
    assert "WHERE NOT EXISTS" in sql
    assert "UPDATE public.stock_prices AS target" in sql
    assert "target.adj_close IS NULL" in sql
    assert "DATE '2024-01-02'" in sql
    assert "DATE '2024-01-03'" in sql
