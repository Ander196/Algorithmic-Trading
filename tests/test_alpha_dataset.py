import numpy as np
import pandas as pd
import pytest

from core.alpha_dataset import AlphaDatasetBuilder, AlphaDatasetConfig


def _bars(seed: int, periods: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=periods)
    returns = rng.normal(0.0004, 0.01, periods)
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, periods),
        },
        index=dates,
    )


def test_dataset_is_cross_sectional_and_point_in_time() -> None:
    dataset = AlphaDatasetBuilder(
        AlphaDatasetConfig(min_history=252, horizon_days=5)
    ).build({"AAA": _bars(1), "BBB": _bars(2)}, _bars(3))

    assert dataset.index.names == ["date", "ticker"]
    assert {"AAA", "BBB"} == set(dataset.index.get_level_values("ticker"))
    assert "target_excess_return" in dataset.columns
    row = dataset.xs("AAA", level="ticker").iloc[0]
    assert pd.notna(row["return_20"])
    assert pd.notna(row["label_end_date"])


def test_features_do_not_change_when_future_bars_are_appended() -> None:
    base = _bars(1, 320)
    extended = _bars(1, 340)
    market_base = _bars(2, 320)
    market_extended = _bars(2, 340)

    config = AlphaDatasetConfig(min_history=252, horizon_days=5)
    builder = AlphaDatasetBuilder(config)
    base_ds = builder.build({"AAA": base}, market_base)
    extended_ds = builder.build({"AAA": extended}, market_extended)

    common = base_ds.index.intersection(extended_ds.index)
    feature = config.feature_columns[0]
    pd.testing.assert_series_equal(
        base_ds.loc[common, feature],
        extended_ds.loc[common, feature],
        check_names=False,
    )


def test_invalid_inputs_are_rejected() -> None:
    bad = _bars(1).drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing required columns"):
        AlphaDatasetBuilder().build({"AAA": bad}, _bars(2))
