"""Walk-forward evaluation for the Alpha / Stock Selection layer.

The evaluator measures the ranking quality of Alpha predictions strictly on
out-of-sample rows produced by WalkForwardProtocol.

Primary metric:
    daily cross-sectional Spearman rank IC

Secondary metrics:
    top-N realized excess return
    top-N versus universe spread
    top-N versus bottom-N spread
    top-N hit rate
    cross-sectional coverage

This module evaluates a model; it does not tune model hyperparameters or
select a production top-N configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from core.alpha_model import AlphaModel, AlphaModelConfig
from core.walk_forward import (
    WalkForwardFold,
    WalkForwardProtocol,
    materialize_fold,
)


@dataclass(frozen=True)
class AlphaEvaluationConfig:
    """Configuration for out-of-sample Alpha evaluation."""

    top_n_values: tuple[int, ...] = (5, 10, 20, 50)
    min_cross_section: int = 20

    def __post_init__(self) -> None:
        if not self.top_n_values:
            raise ValueError("top_n_values must not be empty")
        if any(value <= 0 for value in self.top_n_values):
            raise ValueError("top_n_values must contain only positive values")
        if len(set(self.top_n_values)) != len(self.top_n_values):
            raise ValueError("top_n_values must not contain duplicates")
        if self.min_cross_section < 2:
            raise ValueError("min_cross_section must be >= 2")


@dataclass(frozen=True)
class TopNMetrics:
    """Aggregated metrics for one candidate-universe size."""

    top_n: int
    mean_excess_return: float
    mean_universe_excess_return: float
    mean_top_vs_universe_spread: float
    mean_top_vs_bottom_spread: float
    mean_hit_rate: float
    valid_dates: int


@dataclass(frozen=True)
class FoldMetrics:
    """Out-of-sample metrics for one walk-forward fold."""

    fold_id: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_rows: int
    test_rows: int
    valid_dates: int
    mean_ic: float
    median_ic: float
    ic_std: float
    ic_positive_fraction: float
    mean_cross_section: float
    top_n_metrics: tuple[TopNMetrics, ...]


@dataclass(frozen=True)
class AlphaEvaluationReport:
    """Complete walk-forward evaluation report."""

    model_version: str
    folds: tuple[FoldMetrics, ...]
    oos_predictions: pd.DataFrame

    @property
    def fold_count(self) -> int:
        return len(self.folds)

    def aggregate_top_n(self, top_n: int) -> TopNMetrics:
        """Aggregate one Top-N configuration across folds."""
        metrics = [
            metric
            for fold in self.folds
            for metric in fold.top_n_metrics
            if metric.top_n == top_n
        ]
        if not metrics:
            raise ValueError(f"No evaluation metrics available for top_n={top_n}")

        return TopNMetrics(
            top_n=top_n,
            mean_excess_return=float(
                np.mean([metric.mean_excess_return for metric in metrics])
            ),
            mean_universe_excess_return=float(
                np.mean([metric.mean_universe_excess_return for metric in metrics])
            ),
            mean_top_vs_universe_spread=float(
                np.mean([metric.mean_top_vs_universe_spread for metric in metrics])
            ),
            mean_top_vs_bottom_spread=float(
                np.mean([metric.mean_top_vs_bottom_spread for metric in metrics])
            ),
            mean_hit_rate=float(
                np.mean([metric.mean_hit_rate for metric in metrics])
            ),
            valid_dates=sum(metric.valid_dates for metric in metrics),
        )

    @property
    def mean_ic(self) -> float:
        return float(np.mean([fold.mean_ic for fold in self.folds]))

    @property
    def median_ic(self) -> float:
        return float(np.median([fold.median_ic for fold in self.folds]))

    @property
    def ic_positive_fraction(self) -> float:
        return float(
            np.mean([fold.ic_positive_fraction for fold in self.folds])
        )


class AlphaEvaluator:
    """Evaluate Alpha with fresh model fits on chronological OOS folds."""

    def __init__(
        self,
        config: AlphaEvaluationConfig | None = None,
    ) -> None:
        self.config = config or AlphaEvaluationConfig()

    def evaluate_walk_forward(
        self,
        dataset: pd.DataFrame,
        protocol: WalkForwardProtocol,
        model_config: AlphaModelConfig | None = None,
    ) -> AlphaEvaluationReport:
        """Fit and evaluate Alpha separately on every walk-forward fold."""
        folds = list(protocol.split(dataset))
        if not folds:
            raise ValueError("Walk-forward protocol produced no valid folds")

        model_config = model_config or AlphaModelConfig()
        fold_metrics: list[FoldMetrics] = []
        prediction_frames: list[pd.DataFrame] = []

        for fold in folds:
            train, test = materialize_fold(dataset, fold)

            model = AlphaModel(model_config)
            model.fit(train)
            predictions = model.predict(test)

            fold_metric = self.evaluate_fold(predictions, fold)
            fold_metrics.append(fold_metric)

            prediction_frames.append(
                predictions[
                    ["target_excess_return", "expected_excess_return"]
                ].copy()
            )

        oos_predictions = pd.concat(prediction_frames).sort_index()
        oos_predictions = oos_predictions[
            ~oos_predictions.index.duplicated(keep="first")
        ]

        return AlphaEvaluationReport(
            model_version=AlphaModel.MODEL_VERSION,
            folds=tuple(fold_metrics),
            oos_predictions=oos_predictions,
        )

    def evaluate_fold(
        self,
        predictions: pd.DataFrame,
        fold: WalkForwardFold,
    ) -> FoldMetrics:
        """Evaluate one fold's OOS predictions by date."""
        self._validate_predictions(predictions)

        rows = predictions.dropna(
            subset=["expected_excess_return", "target_excess_return"]
        ).copy()
        if rows.empty:
            raise ValueError(f"fold {fold.fold_id}: no valid prediction rows")

        rows = rows.reset_index()
        rows["date"] = pd.to_datetime(rows["date"])

        daily_ics: list[float] = []
        daily_cross_sections: list[int] = []
        top_n_daily: dict[int, list[tuple[float, float, float, float, float]]] = {
            top_n: [] for top_n in self.config.top_n_values
        }

        for date, group in rows.groupby("date", sort=True):
            group = group.drop_duplicates(subset=["ticker"])
            if len(group) < self.config.min_cross_section:
                continue

            prediction = group["expected_excess_return"].to_numpy(dtype=float)
            target = group["target_excess_return"].to_numpy(dtype=float)

            if np.unique(prediction).size > 1 and np.unique(target).size > 1:
                correlation = spearmanr(prediction, target).statistic
                if np.isfinite(correlation):
                    daily_ics.append(float(correlation))

            daily_cross_sections.append(len(group))

            ordered = group.sort_values(
                ["expected_excess_return", "ticker"],
                ascending=[False, True],
            )
            universe_return = float(group["target_excess_return"].mean())

            for top_n in self.config.top_n_values:
                if len(group) < top_n:
                    continue

                top = ordered.head(top_n)
                bottom = ordered.tail(top_n)

                top_return = float(top["target_excess_return"].mean())
                bottom_return = float(bottom["target_excess_return"].mean())
                hit_rate = float(
                    (top["target_excess_return"] > 0.0).mean()
                )

                top_n_daily[top_n].append(
                    (
                        top_return,
                        universe_return,
                        top_return - universe_return,
                        top_return - bottom_return,
                        hit_rate,
                    )
                )

        if not daily_cross_sections:
            raise ValueError(
                f"fold {fold.fold_id}: no dates meet min_cross_section="
                f"{self.config.min_cross_section}"
            )

        fold_top_metrics = tuple(
            self._aggregate_top_n(top_n, values)
            for top_n, values in top_n_daily.items()
            if values
        )

        return FoldMetrics(
            fold_id=fold.fold_id,
            test_start=fold.test_start,
            test_end=fold.test_end,
            train_rows=fold.train_rows,
            test_rows=fold.test_rows,
            valid_dates=len(daily_cross_sections),
            mean_ic=float(np.mean(daily_ics)) if daily_ics else float("nan"),
            median_ic=float(np.median(daily_ics)) if daily_ics else float("nan"),
            ic_std=float(np.std(daily_ics, ddof=1))
            if len(daily_ics) > 1
            else 0.0,
            ic_positive_fraction=float(
                np.mean(np.asarray(daily_ics) > 0.0)
            )
            if daily_ics
            else float("nan"),
            mean_cross_section=float(np.mean(daily_cross_sections)),
            top_n_metrics=fold_top_metrics,
        )

    @staticmethod
    def _aggregate_top_n(
        top_n: int,
        values: list[tuple[float, float, float, float, float]],
    ) -> TopNMetrics:
        array = np.asarray(values, dtype=float)
        return TopNMetrics(
            top_n=top_n,
            mean_excess_return=float(array[:, 0].mean()),
            mean_universe_excess_return=float(array[:, 1].mean()),
            mean_top_vs_universe_spread=float(array[:, 2].mean()),
            mean_top_vs_bottom_spread=float(array[:, 3].mean()),
            mean_hit_rate=float(array[:, 4].mean()),
            valid_dates=len(values),
        )

    @staticmethod
    def _validate_predictions(predictions: pd.DataFrame) -> None:
        if not isinstance(predictions, pd.DataFrame):
            raise ValueError("predictions must be a pandas DataFrame")
        if not isinstance(predictions.index, pd.MultiIndex):
            raise ValueError(
                "predictions must use a MultiIndex of (date, ticker)"
            )
        if list(predictions.index.names) != ["date", "ticker"]:
            raise ValueError(
                "predictions index levels must be named ('date', 'ticker')"
            )

        required = {"target_excess_return", "expected_excess_return"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(
                f"predictions missing required columns: {sorted(missing)}"
            )


__all__ = [
    "AlphaEvaluationConfig",
    "AlphaEvaluator",
    "AlphaEvaluationReport",
    "FoldMetrics",
    "TopNMetrics",
]
