import pandas as pd

from data.price_validation import audit_ohlcv_frame, validate_ohlcv_frame


def _valid_frame():
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [12.0, 13.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [11.0, 12.0, 12.5],
            "adj_close": [11.0, 12.0, 12.5],
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


def test_validator_removes_rows_with_non_positive_adjusted_close():
    frame = _valid_frame()
    frame.loc[1, "adj_close"] = -1

    result = validate_ohlcv_frame(frame, "TEST")

    assert result is not None
    assert result["price_date"].tolist() == ["2026-01-01", "2026-01-03"]



def test_audit_classifies_missing_negative_and_inconsistent_values():
    frame = _valid_frame()
    frame.loc[0, "open"] = 0
    frame.loc[1, "volume"] = -1
    frame.loc[2, "high"] = 10

    audit = audit_ohlcv_frame(frame, "TEST")

    assert audit.invalid_row_count == 3
    assert audit.reason_counts["non_positive_ohlc_values"] == 1
    assert audit.reason_counts["negative_volume"] == 1
    assert audit.reason_counts["high_below_ohlc_max"] == 1
    assert all(audit.invalid_rows["validation_reasons"].map(bool))


def test_audit_distinguishes_missing_columns_from_bad_rows():
    frame = _valid_frame().drop(columns=["high"])

    audit = audit_ohlcv_frame(frame, "TEST")

    assert audit.missing_columns == ("high",)
    assert audit.reason_counts == {"missing_required_columns": 1}
    assert audit.invalid_row_count == 0


def test_audit_classifies_missing_negative_and_inconsistent_values():
    frame = _valid_frame()
    frame.loc[0, "open"] = 0
    frame.loc[1, "volume"] = -1
    frame.loc[2, "high"] = 10

    audit = audit_ohlcv_frame(frame, "TEST")

    assert audit.invalid_row_count == 3
    assert audit.reason_counts["non_positive_ohlc_values"] == 1
    assert audit.reason_counts["negative_volume"] == 1
    assert audit.reason_counts["high_below_ohlc_max"] == 1
    assert all(audit.invalid_rows["validation_reasons"].map(bool))


def test_audit_distinguishes_missing_columns_from_bad_rows():
    frame = _valid_frame().drop(columns=["high"])

    audit = audit_ohlcv_frame(frame, "TEST")

    assert audit.missing_columns == ("high",)
    assert audit.reason_counts == {"missing_required_columns": 1}
    assert audit.invalid_row_count == 0


def test_audit_classifies_non_numeric_non_finite_and_missing_values():
    frame = _valid_frame()
    frame.loc[0, "open"] = "not-a-price"
    frame.loc[1, "volume"] = float("inf")
    frame.loc[2, "close"] = None

    audit = audit_ohlcv_frame(frame, "TEST")

    assert audit.reason_counts["non_numeric_ohlc_values"] == 1
    assert audit.reason_counts["non_finite_volume"] == 1
    assert audit.reason_counts["missing_ohlc_values"] == 1

def test_validator_keeps_valid_positive_ohlcv_rows():
    frame = _valid_frame()

    result = validate_ohlcv_frame(frame, "TEST")

    assert result is not None
    assert len(result) == 3
