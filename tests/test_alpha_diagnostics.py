import pandas as pd
import pytest

from core.alpha_diagnostics import evaluate_alpha_diagnostics, extreme_return_rows


def _oos_predictions(dates: int = 4, tickers: int = 30) -> pd.DataFrame:
    rows = []
    for date_idx, date in enumerate(pd.bdate_range("2024-01-01", periods=dates)):
        for ticker_idx in range(tickers):
            momentum = float(ticker_idx)
            target = momentum / 100.0 + (date_idx + 1) / 1000.0
            rows.append(
                {
                    "date": date,
                    "ticker": f"T{ticker_idx:03d}",
                    "expected_excess_return": momentum,
                    "return_20": momentum,
                    "return_60": momentum,
                    "target_excess_return": target,
                }
            )
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def test_diagnostics_compare_model_momentum_and_random_baseline() -> None:
    report = evaluate_alpha_diagnostics(
        _oos_predictions(),
        top_n=5,
        min_cross_section=20,
        random_repeats=5,
        random_seed=17,
    )

    comparison = report.ranking_comparison.set_index("signal")
    assert comparison.loc["ridge_alpha", "mean_ic"] == pytest.approx(1.0)
    assert comparison.loc["momentum_20", "mean_ic"] == pytest.approx(1.0)
    assert comparison.loc["momentum_60", "mean_ic"] == pytest.approx(1.0)
    assert comparison.loc["ridge_alpha", "top_vs_universe_spread"] > 0
    assert comparison.loc["random_baseline_mean_5_runs", "valid_dates"] == 4


def test_random_baseline_is_reproducible_for_same_seed() -> None:
    first = evaluate_alpha_diagnostics(
        _oos_predictions(), random_repeats=3, random_seed=123
    ).ranking_comparison
    second = evaluate_alpha_diagnostics(
        _oos_predictions(), random_repeats=3, random_seed=123
    ).ranking_comparison

    pd.testing.assert_frame_equal(first, second)


def test_outlier_sensitivity_uses_same_dates_and_caps_target_values() -> None:
    predictions = _oos_predictions()
    predictions.loc[(pd.Timestamp("2024-01-01"), "T029"), "target_excess_return"] = 100.0
    report = evaluate_alpha_diagnostics(
        predictions,
        top_n=5,
        min_cross_section=20,
        random_repeats=2,
    )

    sensitivity = report.outlier_sensitivity
    assert sensitivity["valid_dates"].tolist() == [4, 4]
    assert sensitivity.loc[0, "top_n_excess_return"] > sensitivity.loc[1, "top_n_excess_return"]
    assert sensitivity.loc[1, "target_treatment"] == "winsorized_cross_section_1%_99%"


def test_extreme_return_rows_include_both_tails() -> None:
    rows = extreme_return_rows(_oos_predictions(), n_per_side=2)

    assert len(rows) == 4
    assert set(rows["extreme_side"]) == {
        "largest_positive_excess",
        "largest_negative_excess",
    }
    assert rows["target_excess_return"].max() > 0
    assert rows["extreme_side"].eq("largest_negative_excess").any()


@pytest.mark.parametrize("kwargs,match", [
    ({"top_n": 0}, "top_n"),
    ({"random_repeats": 0}, "random_repeats"),
])
def test_diagnostics_validate_configuration(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        evaluate_alpha_diagnostics(_oos_predictions(), **kwargs)
