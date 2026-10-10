import pandas as pd

from data.price_validation import validate_ohlcv_frame


def _valid_frame():
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [12.0, 13.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [11.0, 12.0, 12.5],
            "volume": [100, 110, 120],
            "price_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        }
    )


def test_validator_removes_zero_negative_and_inconsistent_ohlc_rows():
    frame = _valid_frame()
    frame.loc[1, "open"] = 0
    frame.loc[2, "low"] = 14

    result = validate_ohlcv_frame(frame, "TEST")

    assert result is not None
    assert result["price_date"].tolist() == ["2026-01-01"]


def test_validator_refuses_to_fabricate_missing_ohlcv_columns():
    frame = _valid_frame().drop(columns=["high"])

    result = validate_ohlcv_frame(frame, "TEST")

    assert result is None


def test_validator_keeps_valid_positive_ohlcv_rows():
    frame = _valid_frame()

    result = validate_ohlcv_frame(frame, "TEST")

    assert result is not None
    assert len(result) == 3
