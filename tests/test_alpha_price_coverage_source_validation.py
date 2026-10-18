from __future__ import annotations

import pandas as pd

from jobs.audit_alpha_price_coverage import (
    OHLCV_COLUMNS,
    _find_repair_dates,
    _render_repair_sql,
)


def _row(value: float = 10.0) -> dict[str, float | int]:
    return {
        "open": value,
        "high": value + 1,
        "low": value - 1,
        "close": value,
        "adj_close": value,
        "volume": 1000,
    }


def test_invalid_yahoo_ohlcv_is_flagged_for_review_not_inserted() -> None:
    date = pd.Timestamp("2011-09-06")
    source_row = _row(27.0)
    source_row["high"] = 26.0  # high must be >= close
    yahoo = pd.DataFrame([source_row], index=pd.DatetimeIndex([date]), columns=OHLCV_COLUMNS)
    stored = pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([]))

    result = _find_repair_dates(yahoo, stored, pd.DatetimeIndex([date]))

    assert result == [
        (date, "review_source_invalid", ["yahoo_source_high_below_ohlc_max"])
    ]


def test_review_only_source_rows_are_excluded_from_generated_sql() -> None:
    sql = _render_repair_sql([
        {
            "action": "review_source_invalid",
            "price_date": "2011-09-06",
            "ticker": "ASML.AS",
            "open": 26.08,
            "high": 26.47,
            "low": 25.52,
            "close": 26.92,
            "adj_close": 26.92,
            "volume": 1000,
            "repair_reason": "yahoo_source_high_below_ohlc_max",
        }
    ])

    assert "No automatically repairable price gaps" in sql
    assert "INSERT INTO public.stock_prices" not in sql
    assert "UPDATE public.stock_prices" not in sql
