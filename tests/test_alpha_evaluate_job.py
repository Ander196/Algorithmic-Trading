import pandas as pd
import pytest

from core.alpha_diagnostics import evaluate_alpha_diagnostics
from core.alpha_evaluation import AlphaEvaluationReport
from jobs.evaluate_alpha import _print_diagnostics, _to_ohlcv_frame, get_alpha_universe


class _FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def select(self, *_args):
        return self

    def eq(self, *_args):
        return self

    def execute(self):
        return type("Response", (), {"data": self.rows})()


class _FakeClient:
    def __init__(self, rows):
        self.rows = rows

    def table(self, _name):
        return _FakeQuery(self.rows)


def test_get_alpha_universe_uses_active_tickers_and_excludes_market() -> None:
    client = _FakeClient(
        [
            {"ticker": "spy"},
            {"ticker": "AAA"},
            {"ticker": "bbb"},
            {"ticker": "AAA"},
            {"ticker": None},
        ]
    )

    assert get_alpha_universe(client, "SPY") == ["AAA", "BBB"]


def test_to_ohlcv_frame_normalizes_datetime_and_columns() -> None:
    raw = pd.DataFrame(
        {
            "price_date": ["2024-01-03T00:00:00+00:00", "2024-01-02T00:00:00+00:00"],
            "open": [11, 10],
            "high": [12, 11],
            "low": [9, 9],
            "close": [11.5, 10.5],
            "volume": [1000, 900],
        }
    )

    frame = _to_ohlcv_frame(raw)

    assert frame.index.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]


def test_to_ohlcv_frame_rejects_missing_columns() -> None:
    raw = pd.DataFrame(
        {
            "price_date": ["2024-01-02"],
            "close": [10],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        _to_ohlcv_frame(raw)



def test_print_diagnostics_writes_reviewable_csv_files(tmp_path) -> None:
    rows = []
    for date_idx, date in enumerate(pd.bdate_range("2024-01-01", periods=3)):
        for ticker_idx in range(30):
            momentum = float(ticker_idx) / 100.0
            rows.append({
                "date": date,
                "ticker": f"T{ticker_idx:03d}",
                "expected_excess_return": momentum,
                "return_20": momentum,
                "return_60": momentum / 2.0,
                "target_excess_return": momentum + (date_idx - 1) / 100.0,
            })
    predictions = pd.DataFrame(rows).set_index(["date", "ticker"])
    report = AlphaEvaluationReport(
        model_version="test",
        folds=(),
        oos_predictions=predictions,
    )

    _print_diagnostics(
        report,
        top_n=5,
        random_repeats=2,
        random_seed=17,
        diagnostics_dir=tmp_path,
    )

    expected_files = {
        "alpha_oos_predictions.csv",
        "alpha_ranking_comparison.csv",
        "alpha_outlier_sensitivity.csv",
        "alpha_extreme_returns.csv",
    }
    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})
