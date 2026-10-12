"""Diagnostics for out-of-sample Alpha rankings and return outliers.

All ranking comparisons use the same OOS dates and realized excess-return labels.
These diagnostics do not fit models or select production parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class AlphaDiagnosticReport:
    """Baseline comparison and outlier-sensitivity tables."""

    strategy_comparison: pd.DataFrame
    outlier_sensitivity: pd.DataFrame


_METRIC_COLUMNS = (
    "mean_ic",
    "median_daily_ic",
    "positive_ic_fraction",
    "top_n_excess_return",
    "top_vs_universe_spread",
    "top_vs_bottom_spread",
    "hit_rate",
)


def _validate_predictions(
    predictions: pd.DataFrame,
    required_columns: set[str],
) -> None:
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("predictions must be a pandas DataFrame")
    if not isinstance(predictions.index, pd.MultiIndex):
        raise ValueError("predictions must use a MultiIndex of (date, ticker)")
    if list(predictions.index.names) != ["date", "ticker"]:
        raise ValueError("predictions index levels must be named ('date', 'ticker')")
    missing = required_columns - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")


def _evaluate_score(
    predictions: pd.DataFrame,
    score_column: str,
    target_column: str,
    top_n: int,
    min_cross_section: int,
) -> dict[str, float | int]:
    """Evaluate a ranking score with date-equal-weighted cross-sectional metrics."""
    rows = predictions[[score_column, target_column]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    daily_ics: list[float] = []
    top_n_excess: list[float] = []
    top_vs_universe: list[float] = []
    top_vs_bottom: list[float] = []
    hit_rates: list[float] = []
    valid_dates = 0

    for _, group in rows.groupby(level="date", sort=True):
        # One observation per symbol/date; keep input order deterministic.
        group = group.reset_index()
        group = group.drop_duplicates(subset=["ticker"], keep="last")
        if len(group) < max(min_cross_section, top_n):
            continue
        scores = group[score_column].to_numpy(dtype=float)
        targets = group[target_column].to_numpy(dtype=float)
        valid_dates += 1

        if np.unique(scores).size > 1 and np.unique(targets).size > 1:
            correlation = spearmanr(scores, targets).statistic
            if np.isfinite(correlation):
                daily_ics.append(float(correlation))

        ordered = group.sort_values(
            [score_column, "ticker"],
            ascending=[False, True],
            kind="mergesort",
        )
        top = ordered.head(top_n)
        bottom = ordered.tail(top_n)
        top_return = float(top[target_column].mean())
        universe_return = float(group[target_column].mean())
        bottom_return = float(bottom[target_column].mean())

        top_n_excess.append(top_return)
        top_vs_universe.append(top_return - universe_return)
        top_vs_bottom.append(top_return - bottom_return)
        hit_rates.append(float((top[target_column] > 0.0).mean()))

    def mean_or_nan(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        "mean_ic": mean_or_nan(daily_ics),
        "median_daily_ic": float(np.median(daily_ics)) if daily_ics else float("nan"),
        "positive_ic_fraction": (
            float(np.mean(np.asarray(daily_ics) > 0.0))
            if daily_ics
            else float("nan")
        ),
        "top_n_excess_return": mean_or_nan(top_n_excess),
        "top_vs_universe_spread": mean_or_nan(top_vs_universe),
        "top_vs_bottom_spread": mean_or_nan(top_vs_bottom),
        "hit_rate": mean_or_nan(hit_rates),
        "valid_dates": valid_dates,
    }


def evaluate_alpha_diagnostics(
    oos_predictions: pd.DataFrame,
    *,
    top_n: int = 5,
    min_cross_section: int = 20,
    random_repeats: int = 20,
    random_seed: int = 42,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> AlphaDiagnosticReport:
    """Compare Ridge with momentum/random rankings and test target outlier impact.

    The random baseline is the mean over independent seeded random rankings.
    For outlier sensitivity, realized target returns are winsorized within each
    date at the supplied cross-sectional quantiles; OOS scores are unchanged.
    """
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")
    if min_cross_section < 2:
        raise ValueError("min_cross_section must be >= 2")
    if random_repeats <= 0:
        raise ValueError("random_repeats must be greater than zero")
    if not (0.0 <= winsor_lower < winsor_upper <= 1.0):
        raise ValueError("winsor quantiles must satisfy 0 <= lower < upper <= 1")

    _validate_predictions(
        oos_predictions,
        {
            "expected_excess_return",
            "target_excess_return",
            "return_20",
            "return_60",
        },
    )
    if oos_predictions.empty:
        raise ValueError("oos_predictions must not be empty")

    signal_columns = (
        ("ridge_alpha", "expected_excess_return"),
        ("momentum_20", "return_20"),
        ("momentum_60", "return_60"),
    )
    comparison_rows: list[dict[str, float | int | str]] = []
    for signal_name, score_column in signal_columns:
        metrics = _evaluate_score(
            oos_predictions,
            score_column,
            "target_excess_return",
            top_n,
            min_cross_section,
        )
        comparison_rows.append({"signal": signal_name, **metrics})

    rng = np.random.default_rng(random_seed)
    random_metrics: list[dict[str, float | int]] = []
    for _ in range(random_repeats):
        randomized = oos_predictions[["target_excess_return"]].copy()
        randomized["__random_score"] = rng.random(len(randomized))
        metrics = _evaluate_score(
            randomized,
            "__random_score",
            "target_excess_return",
            top_n,
            min_cross_section,
        )
        random_metrics.append(metrics)

    random_frame = pd.DataFrame(random_metrics)
    random_row: dict[str, float | int | str] = {
        "signal": f"random_baseline_mean_{random_repeats}_runs"
    }
    for column in (*_METRIC_COLUMNS, "valid_dates"):
        random_row[column] = float(random_frame[column].mean())
    for column in (
        "mean_ic",
        "top_n_excess_return",
        "top_vs_universe_spread",
        "top_vs_bottom_spread",
        "hit_rate",
    ):
        random_row[f"{column}_std"] = float(random_frame[column].std(ddof=1))
    comparison_rows.append(random_row)

    raw_metrics = _evaluate_score(
        oos_predictions,
        "expected_excess_return",
        "target_excess_return",
        top_n,
        min_cross_section,
    )

    winsorized = oos_predictions.copy()
    winsorized["__winsorized_target"] = winsorized.groupby(level="date")[
        "target_excess_return"
    ].transform(
        lambda values: values.clip(
            lower=values.quantile(winsor_lower),
            upper=values.quantile(winsor_upper),
        )
    )
    winsor_metrics = _evaluate_score(
        winsorized,
        "expected_excess_return",
        "__winsorized_target",
        top_n,
        min_cross_section,
    )
    sensitivity_rows = [
        {"target_treatment": "raw", **raw_metrics},
        {
            "target_treatment": (
                f"winsorized_cross_section_{winsor_lower:.0%}_{winsor_upper:.0%}"
            ),
            **winsor_metrics,
        },
    ]

    return AlphaDiagnosticReport(
        strategy_comparison=pd.DataFrame(comparison_rows),
        outlier_sensitivity=pd.DataFrame(sensitivity_rows),
    )


def extreme_return_rows(
    oos_predictions: pd.DataFrame,
    *,
    n_per_side: int = 50,
) -> pd.DataFrame:
    """Return the largest positive and negative OOS excess-return observations."""
    if n_per_side <= 0:
        raise ValueError("n_per_side must be greater than zero")
    _validate_predictions(oos_predictions, {"target_excess_return"})
    rows = oos_predictions.reset_index().copy()
    rows = rows.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["target_excess_return"]
    )
    positive = rows.nlargest(n_per_side, "target_excess_return").copy()
    positive["extreme_side"] = "largest_positive_excess"
    negative = rows.nsmallest(n_per_side, "target_excess_return").copy()
    negative["extreme_side"] = "largest_negative_excess"
    result = pd.concat([positive, negative], ignore_index=True)
    return result.sort_values(
        ["extreme_side", "target_excess_return"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)


__all__ = [
    "AlphaDiagnosticReport",
    "evaluate_alpha_diagnostics",
    "extreme_return_rows",
]
