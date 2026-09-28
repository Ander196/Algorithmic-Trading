import numpy as np
import pandas as pd
import pytest

from core.alpha_evaluation import AlphaEvaluationConfig, AlphaEvaluator
from core.walk_forward import WalkForwardFold


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
    metrics = evaluator.evaluate_fold(predictions, _fold(len(predictions)))

    assert metrics.valid_dates == 0 if False else metrics.valid_dates == 0


def test_predictions_require_expected_and_realized_returns() -> None:
    predictions = _predictions().drop(columns=["target_excess_return"])
    evaluator = AlphaEvaluator()
    with pytest.raises(ValueError, match="missing"):
        evaluator.evaluate_fold(predictions, _fold(len(predictions)))
