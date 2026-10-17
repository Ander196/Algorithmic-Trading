import pandas as pd
import pytest

from jobs.backtest_alpha_portfolios import (
    _load_adjusted_price_panel,
    _load_predictions,
)


def test_load_predictions_csv_builds_named_multiindex(tmp_path) -> None:
    path = tmp_path / "predictions.csv"
    pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02"],
        "ticker": ["AAA", "BBB"],
        "expected_excess_return": [0.01, -0.02],
    }).to_csv(path, index=False)

    result = _load_predictions(path)

    assert list(result.index.names) == ["date", "ticker"]
    assert result.loc[(pd.Timestamp("2024-01-02"), "AAA"), "expected_excess_return"] == pytest.approx(0.01)


def test_load_adjusted_price_panel_pivots_long_format(tmp_path) -> None:
    path = tmp_path / "prices.csv"
    pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02", "2024-01-03"],
        "ticker": ["AAA", "SPY", "AAA"],
        "adjusted_close_base": [10.0, 100.0, 11.0],
    }).to_csv(path, index=False)

    result = _load_adjusted_price_panel(path)

    assert result.loc[pd.Timestamp("2024-01-02"), "AAA"] == pytest.approx(10.0)
    assert result.loc[pd.Timestamp("2024-01-02"), "SPY"] == pytest.approx(100.0)
    assert result.loc[pd.Timestamp("2024-01-03"), "AAA"] == pytest.approx(11.0)


def test_load_adjusted_price_panel_rejects_duplicate_ticker_dates(tmp_path) -> None:
    path = tmp_path / "prices.csv"
    pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02"],
        "ticker": ["AAA", "AAA"],
        "adjusted_close_base": [10.0, 10.1],
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        _load_adjusted_price_panel(path)
