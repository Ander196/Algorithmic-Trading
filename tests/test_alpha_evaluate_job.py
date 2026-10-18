import pandas as pd
import pytest

from core.alpha_evaluation import AlphaEvaluationReport
from jobs.evaluate_alpha import (
    _convert_stock_histories_to_usd,
    _filter_dataset_by_evaluation_end_date,
    _print_diagnostics,
    _to_ohlcv_frame,
    get_alpha_universe,
)
import jobs.evaluate_alpha as evaluate_alpha_module


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


def test_evaluation_end_date_filters_signal_dates_only() -> None:
    dates = pd.to_datetime(["2026-07-20", "2026-07-21", "2026-07-22"])
    index = pd.MultiIndex.from_arrays(
        [dates, ["AAA", "AAA", "AAA"]],
        names=["date", "ticker"],
    )
    dataset = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
            "target_excess_return": [0.01, 0.02, 0.03],
            # Outcomes may mature after the last signal date; they're scored
            # out-of-sample and do not participate in that fold's fit.
            "label_end_date": pd.to_datetime(["2026-07-27", "2026-07-28", "2026-07-29"]),
        },
        index=index,
    )

    filtered = _filter_dataset_by_evaluation_end_date(dataset, "2026-07-21")

    assert filtered.index.get_level_values("date").max() == pd.Timestamp("2026-07-21")
    assert filtered["label_end_date"].max() == pd.Timestamp("2026-07-28")


def test_evaluation_end_date_rejects_cutoff_before_dataset() -> None:
    dates = pd.to_datetime(["2026-07-20", "2026-07-21"])
    index = pd.MultiIndex.from_arrays(
        [dates, ["AAA", "AAA"]],
        names=["date", "ticker"],
    )
    dataset = pd.DataFrame({"target_excess_return": [0.01, 0.02]}, index=index)

    with pytest.raises(ValueError, match="No Alpha dataset rows are available"):
        _filter_dataset_by_evaluation_end_date(dataset, "2020-01-01")


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




def test_to_ohlcv_frame_adjusts_all_price_fields_to_adj_close() -> None:
    raw = pd.DataFrame(
        {
            "price_date": ["2024-01-02"],
            "open": [99.0],
            "high": [110.0],
            "low": [90.0],
            "close": [100.0],
            "adj_close": [90.0],
            "volume": [1000],
        }
    )

    frame = _to_ohlcv_frame(raw)

    assert frame.iloc[0]["open"] == pytest.approx(89.1)
    assert frame.iloc[0]["high"] == pytest.approx(99.0)
    assert frame.iloc[0]["low"] == pytest.approx(81.0)
    assert frame.iloc[0]["close"] == pytest.approx(90.0)


def test_convert_stock_histories_to_usd_aligns_fx_without_lookahead(monkeypatch) -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    local_history = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [98.0, 99.0, 100.0],
            "close": [102.0, 103.0, 104.0],
            "volume": [1000.0, 1100.0, 1200.0],
        },
        index=dates,
    )
    fx_dates = pd.DatetimeIndex([dates[1], dates[2]])
    fx_series = pd.Series([1.1, 1.2], index=fx_dates, name="EUR")
    market = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [98.0, 99.0, 100.0],
            "close": [102.0, 103.0, 104.0],
            "volume": [1_000_000.0] * 3,
        },
        index=dates,
    )
    monkeypatch.setattr(
        evaluate_alpha_module,
        "_fx_usd_per_unit",
        lambda currency, start, end: fx_series,
    )

    converted = _convert_stock_histories_to_usd({"AAA.DE": local_history}, market)
    usd_history = converted["AAA.DE"]

    assert usd_history.index.tolist() == [dates[1], dates[2]]
    assert usd_history["close"].tolist() == pytest.approx([103.0 * 1.1, 104.0 * 1.2])
    assert usd_history["volume"].tolist() == [1100.0, 1200.0]


def test_convert_stock_histories_to_usd_rejects_internal_fx_gap(monkeypatch) -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    local_history = pd.DataFrame(
        {
            "open": [100.0] * 8,
            "high": [105.0] * 8,
            "low": [98.0] * 8,
            "close": [102.0] * 8,
            "volume": [1000.0] * 8,
        },
        index=dates,
    )
    fx_series = pd.Series(
        [1.1, 1.2],
        index=pd.DatetimeIndex([dates[0], dates[7]]),
        name="EUR",
    )
    market = pd.DataFrame({"close": [100.0] * 8}, index=dates)
    monkeypatch.setattr(
        evaluate_alpha_module,
        "_fx_usd_per_unit",
        lambda currency, start, end: fx_series,
    )

    with pytest.raises(ValueError, match="internal/trailing EUR/USD FX gap"):
        _convert_stock_histories_to_usd({"AAA.DE": local_history}, market)


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
