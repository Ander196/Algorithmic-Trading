import numpy as np
import pandas as pd
import pytest

from core.alpha_dataset import DEFAULT_FEATURE_COLUMNS
from core.alpha_evaluation import AlphaEvaluationConfig, AlphaEvaluator
from core.alpha_model import AlphaModelConfig
from core.walk_forward import WalkForwardConfig, WalkForwardFold, WalkForwardProtocol


def _predictions(
    dates: int = 3,
    tickers: int = 10,
    inverted: bool = False,
) -> pd.DataFrame:
    rows = []
    sessions = pd.bdate_range("2024-01-01", periods=dates)
    for date in sessions:
        for ticker_id in range(tickers):
            score = float(ticker_id)
            target = -score if inverted else score
            rows.append(
                {
                    "date": date,
                    "ticker": f"T{ticker_id:02d}",
                    "expected_excess_return": score,
                    "target_excess_return": target,
                }
            )
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def _fold(test_rows: int) -> WalkForwardFold:
    return WalkForwardFold(
        fold_id=0,
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2022-12-30"),
        test_start=pd.Timestamp("2023-01-03"),
        test_end=pd.Timestamp("2023-06-30"),
        purged_train_end=pd.Timestamp("2022-12-30"),
        train_rows=1000,
        test_rows=test_rows,
    )


def test_perfect_ranking_has_positive_ic_and_top_n_spread() -> None:
    evaluator = AlphaEvaluator(
        AlphaEvaluationConfig(top_n_values=(2,), min_cross_section=5)
    )
    metrics = evaluator.evaluate_fold(_predictions(), _fold(30))

    assert metrics.mean_ic == pytest.approx(1.0)
    assert metrics.ic_positive_fraction == pytest.approx(1.0)

    top = metrics.top_n_metrics[0]
    assert top.top_n == 2
    assert top.mean_top_vs_universe_spread > 0
    assert top.mean_top_vs_bottom_spread > 0
    assert top.mean_hit_rate == pytest.approx(1.0)


def test_inverted_ranking_has_negative_ic() -> None:
    evaluator = AlphaEvaluator(
        AlphaEvaluationConfig(top_n_values=(2,), min_cross_section=5)
    )
    metrics = evaluator.evaluate_fold(_predictions(inverted=True), _fold(30))

    assert metrics.mean_ic == pytest.approx(-1.0)
    assert metrics.ic_positive_fraction == pytest.approx(0.0)


def test_dates_below_minimum_cross_section_are_excluded() -> None:
    predictions = _predictions(dates=2, tickers=10)
    predictions = predictions.reset_index()
    predictions = predictions[predictions["date"] != pd.Timestamp("2024-01-01")]
    predictions = predictions[predictions["ticker"].isin(["T00", "T01", "T02"])]
    predictions = predictions.set_index(["date", "ticker"])

    evaluator = AlphaEvaluator(
        AlphaEvaluationConfig(top_n_values=(2,), min_cross_section=5)
    )
    with pytest.raises(ValueError, match="no dates meet"):
        evaluator.evaluate_fold(predictions, _fold(len(predictions)))


def test_predictions_require_expected_and_realized_returns() -> None:
    predictions = _predictions().drop(columns=["target_excess_return"])
    evaluator = AlphaEvaluator()
    with pytest.raises(ValueError, match="missing"):
        evaluator.evaluate_fold(predictions, _fold(len(predictions)))



def test_walk_forward_report_preserves_features_for_baseline_diagnostics() -> None:
    rng = np.random.default_rng(23)
    dates = pd.bdate_range("2022-01-03", periods=80)
    tickers = [f"T{ticker_id:02d}" for ticker_id in range(25)]
    rows = []

    for date in dates:
        for ticker_id, ticker in enumerate(tickers):
            features = {
                column: float(rng.normal())
                for column in DEFAULT_FEATURE_COLUMNS
            }
            features["return_20"] = (
                (ticker_id - 12) / 100.0 + float(rng.normal(0.0, 0.02))
            )
            features["return_60"] = (
                (ticker_id - 12) / 120.0 + float(rng.normal(0.0, 0.02))
            )
            target = (
                0.2 * features["return_20"]
                - 0.05 * features["realized_vol_21"]
                + float(rng.normal(0.0, 0.03))
            )
            rows.append({
                "date": date,
                "ticker": ticker,
                **features,
                "target_excess_return": target,
                "label_end_date": date + pd.Timedelta(days=5),
            })

    dataset = pd.DataFrame(rows).set_index(["date", "ticker"])
    protocol = WalkForwardProtocol(
        WalkForwardConfig(
            train_period=40,
            test_period=10,
            step_period=10,
            horizon_days=5,
            min_train_rows=100,
        )
    )
    evaluator = AlphaEvaluator(
        AlphaEvaluationConfig(top_n_values=(5,), min_cross_section=20)
    )

    report = evaluator.evaluate_walk_forward(
        dataset,
        protocol,
        model_config=AlphaModelConfig(min_samples=100),
    )

    assert report.fold_count >= 1
    assert {
        "target_excess_return",
        "expected_excess_return",
        "return_20",
        "return_60",
    }.issubset(report.oos_predictions.columns)
    assert len(report.oos_predictions) > 0
