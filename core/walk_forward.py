"""Leakage-aware walk-forward protocol for Alpha research.

The protocol separates model-development periods from future evaluation periods.
Training rows whose future label overlaps the OOS boundary are purged.

This module only defines temporal folds. It does not fit models or select
hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    train_period: int = 756
    test_period: int = 126
    step_period: int = 126
    horizon_days: int = 5
    min_train_rows: int = 500

    def __post_init__(self) -> None:
        for name in ("train_period", "test_period", "step_period", "horizon_days"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.min_train_rows <= 0:
            raise ValueError("min_train_rows must be > 0")


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purged_train_end: pd.Timestamp
    train_rows: int
    test_rows: int


class WalkForwardProtocol:
    """Generate chronological, purged train/OOS folds."""

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    def split(self, dataset: pd.DataFrame) -> Iterator[WalkForwardFold]:
        dates = self._unique_dates(dataset)
        required_dates = self.config.train_period + self.config.test_period
        if len(dates) < required_dates:
            return

        start = 0
        fold_id = 0

        while start + required_dates <= len(dates):
            train_start = dates[start]
            train_end = dates[start + self.config.train_period - 1]
            test_start = dates[start + self.config.train_period]
            test_end = dates[
                start + self.config.train_period + self.config.test_period - 1
            ]

            # Purge labels whose future endpoint reaches the OOS boundary.
            label_end = pd.to_datetime(dataset["label_end_date"])
            dates_index = dataset.index.get_level_values("date")
            train_mask = (
                (dates_index >= train_start)
                & (dates_index <= train_end)
                & (label_end < test_start)
            )
            test_mask = (
                (dates_index >= test_start)
                & (dates_index <= test_end)
            )

            train_rows = int(train_mask.sum())
            test_rows = int(test_mask.sum())
            if train_rows >= self.config.min_train_rows and test_rows > 0:
                yield WalkForwardFold(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    purged_train_end=test_start - pd.Timedelta(days=1),
                    train_rows=train_rows,
                    test_rows=test_rows,
                )
                fold_id += 1

            start += self.config.step_period

    @staticmethod
    def _unique_dates(dataset: pd.DataFrame) -> list[pd.Timestamp]:
        if not isinstance(dataset.index, pd.MultiIndex):
            raise ValueError("dataset must use a MultiIndex of (date, ticker)")
        if dataset.index.nlevels != 2:
            raise ValueError("dataset index must have exactly two levels")
        if list(dataset.index.names) != ["date", "ticker"]:
            raise ValueError(
                "dataset index levels must be named ('date', 'ticker')"
            )
        if "label_end_date" not in dataset.columns:
            raise ValueError("dataset must contain 'label_end_date'")

        dates = pd.DatetimeIndex(
            pd.to_datetime(dataset.index.get_level_values("date")).unique()
        ).sort_values()
        return list(dates)


def materialize_fold(
    dataset: pd.DataFrame,
    fold: WalkForwardFold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the training and OOS rows for a generated fold."""
    dates = dataset.index.get_level_values("date")
    label_end = pd.to_datetime(dataset["label_end_date"])

    train_mask = (
        (dates >= fold.train_start)
        & (dates <= fold.train_end)
        & (label_end < fold.test_start)
    )
    test_mask = (dates >= fold.test_start) & (dates <= fold.test_end)

    return dataset.loc[train_mask].copy(), dataset.loc[test_mask].copy()


__all__ = [
    "WalkForwardConfig",
    "WalkForwardFold",
    "WalkForwardProtocol",
    "materialize_fold",
]
