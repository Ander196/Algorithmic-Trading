import numpy as np
import pandas as pd
import pytest

from core.walk_forward import WalkForwardConfig, WalkForwardProtocol, materialize_fold


def _dataset(periods: int = 40, tickers: int = 3) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=periods)
    rows = []
    for date in dates:
        for ticker_id in range(tickers):
            rows.append(
                {
                    "date": date,
                    "ticker": f"T{ticker_id}",
                    "feature": float(ticker_id),
                    "target_excess_return": 0.01,
                    "label_end_date": date + pd.Timedelta(days=5),
                }
            )
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def test_walk_forward_is_chronological_and_purged() -> None:
    dataset = _dataset()
    protocol = WalkForwardProtocol(
        WalkForwardConfig(
            train_period=20,
            test_period=10,
            step_period=10,
            horizon_days=5,
            min_train_rows=1,
        )
    )
    folds = list(protocol.split(dataset))

    assert len(folds) == 2
    assert folds[0].train_end < folds[0].test_start
    assert folds[0].test_end < folds[1].test_start

    train, test = materialize_fold(dataset, folds[0])
    assert train.index.get_level_values("date").max() < test.index.get_level_values("date").min()
    assert pd.to_datetime(train["label_end_date"]).max() < folds[0].test_start


def test_invalid_dataset_index_is_rejected() -> None:
    dataset = _dataset().reset_index()
    protocol = WalkForwardProtocol(
        WalkForwardConfig(train_period=10, test_period=10, min_train_rows=1)
    )
    with pytest.raises(ValueError, match="MultiIndex"):
        list(protocol.split(dataset))
