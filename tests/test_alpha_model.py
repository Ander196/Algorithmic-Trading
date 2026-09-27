import numpy as np
import pandas as pd
import pytest

from core.alpha_dataset import DEFAULT_FEATURE_COLUMNS
from core.alpha_model import AlphaModel, AlphaModelConfig


def _dataset(rows: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-01", periods=rows)
    frame = pd.DataFrame(
        rng.normal(size=(rows, len(DEFAULT_FEATURE_COLUMNS))),
        columns=DEFAULT_FEATURE_COLUMNS,
        index=dates,
    )
    frame["target_excess_return"] = (
        0.4 * frame["return_20"]
        - 0.2 * frame["realized_vol_21"]
        + rng.normal(0, 0.5, rows)
    )
    frame["ticker"] = ["AAA"] * rows
    frame["label_end_date"] = dates + pd.Timedelta(days=5)
    frame.index.name = "date"
    return frame.set_index(["date", "ticker"])


def test_alpha_model_fits_and_scores() -> None:
    dataset = _dataset()
    model = AlphaModel(AlphaModelConfig(min_samples=100))
    model.fit(dataset)

    predictions = model.predict(dataset)

    assert model.is_fitted
    assert model.trained_rows == len(dataset)
    assert {"alpha_score", "probability_positive"} <= set(predictions.columns)
    assert predictions["probability_positive"].between(0, 1).all()
    assert len(model.coefficients()) == len(DEFAULT_FEATURE_COLUMNS)


def test_alpha_model_requires_both_target_classes() -> None:
    dataset = _dataset()
    dataset["target_excess_return"] = 0.01

    with pytest.raises(ValueError, match="both positive"):
        AlphaModel(AlphaModelConfig(min_samples=100)).fit(dataset)


def test_alpha_model_validates_index() -> None:
    dataset = _dataset().reset_index()
    with pytest.raises(ValueError, match="MultiIndex"):
        AlphaModel().predict(dataset)


def test_alpha_model_requires_fit_before_prediction() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        AlphaModel().predict(_dataset())
