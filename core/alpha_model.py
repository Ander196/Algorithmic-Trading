"""Baseline Alpha model for cross-sectional stock selection.

The model consumes a point-in-time Alpha dataset and estimates the probability
that a stock's future excess return will be positive. The probability is used
as an interpretable signal; the model's decision score is the primary ranking
signal.

This module intentionally does not:
- perform walk-forward splitting;
- choose hyperparameters using future data;
- apply AllocationSignal constraints;
- generate StrategySignal objects;
- size positions or place orders.

Those responsibilities remain in the temporal protocol, selection/allocation,
Strategy, RiskManager and Execution layers respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.alpha_dataset import DEFAULT_FEATURE_COLUMNS


@dataclass(frozen=True)
class AlphaModelConfig:
    """Configuration for the V1 logistic-regression Alpha baseline."""

    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS
    c: float = 1.0
    max_iter: int = 1000
    random_state: int = 42
    min_samples: int = 200

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must not contain duplicates")
        if self.c <= 0:
            raise ValueError("c must be > 0")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if self.min_samples <= 1:
            raise ValueError("min_samples must be > 1")


class AlphaModel:
    """Fit and score a point-in-time binary Alpha model.

    The training target is:
        1  -> future excess return > 0
        0  -> future excess return <= 0

    The model is deliberately simple for the first baseline. A regularized
    logistic regression in a StandardScaler pipeline provides a transparent
    benchmark before introducing more complex learners.
    """

    MODEL_VERSION = "logistic-v1"

    def __init__(self, config: AlphaModelConfig | None = None) -> None:
        self.config = config or AlphaModelConfig()
        self._pipeline: Pipeline | None = None
        self._trained_rows: int = 0

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None

    @property
    def trained_rows(self) -> int:
        return self._trained_rows

    def fit(self, dataset: pd.DataFrame) -> "AlphaModel":
        """Fit the model on one already-defined training window.

        Temporal splitting and label purging must happen before this method is
        called, normally via WalkForwardProtocol.materialize_fold().
        """
        self._validate_dataset(dataset)

        train = dataset.dropna(
            subset=[*self.config.feature_columns, "target_excess_return"]
        ).copy()
        if len(train) < self.config.min_samples:
            raise ValueError(
                f"training dataset has {len(train)} rows; "
                f"minimum is {self.config.min_samples}"
            )

        x = train.loc[:, self.config.feature_columns]
        y = (train["target_excess_return"] > 0.0).astype(int)

        if y.nunique() != 2:
            raise ValueError(
                "target_excess_return must contain both positive and "
                "non-positive classes in the training data"
            )

        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.config.c,
                        max_iter=self.config.max_iter,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )
        self._pipeline.fit(x, y)
        self._trained_rows = len(train)
        return self

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Return Alpha scores/probabilities for each input row."""
        if not self.is_fitted:
            raise RuntimeError("AlphaModel must be fitted before predict()")

        self._validate_dataset(dataset)
        rows = dataset.dropna(subset=list(self.config.feature_columns)).copy()
        if rows.empty:
            raise ValueError("No rows with complete Alpha features are available")

        x = rows.loc[:, self.config.feature_columns]
        assert self._pipeline is not None

        rows["alpha_score"] = self._pipeline.decision_function(x)
        rows["probability_positive"] = self._pipeline.predict_proba(x)[:, 1]
        return rows

    def coefficients(self) -> pd.Series:
        """Return standardized feature coefficients for model inspection."""
        if not self.is_fitted:
            raise RuntimeError("AlphaModel must be fitted before coefficients()")

        assert self._pipeline is not None
        classifier = self._pipeline.named_steps["classifier"]
        return pd.Series(
            classifier.coef_[0],
            index=self.config.feature_columns,
            name="coefficient",
        )

    def _validate_dataset(self, dataset: pd.DataFrame) -> None:
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("dataset must be a pandas DataFrame")
        if not isinstance(dataset.index, pd.MultiIndex):
            raise ValueError("dataset must use a MultiIndex of (date, ticker)")
        if list(dataset.index.names) != ["date", "ticker"]:
            raise ValueError(
                "dataset index levels must be named ('date', 'ticker')"
            )

        missing = set(self.config.feature_columns) - set(dataset.columns)
        if missing:
            raise ValueError(
                f"dataset is missing Alpha features: {sorted(missing)}"
            )

        if "target_excess_return" not in dataset.columns:
            raise ValueError("dataset must contain 'target_excess_return'")


__all__ = ["AlphaModel", "AlphaModelConfig"]
