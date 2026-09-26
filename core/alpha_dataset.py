"""Point-in-time dataset construction for the Alpha / Stock Selection layer.

The dataset builder is independent from model training. It converts daily OHLCV
histories into one cross-sectional row per (date, ticker), with features computed
only from information available on that date and a future excess-return label
used only for training/evaluation.

V1 target:
    excess_return_h = stock_return(t -> t+h) - market_return(t -> t+h)

V1 horizon: 5 trading sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = (
    "return_5", "return_10", "return_20", "return_60",
    "relative_return_20", "relative_return_60",
    "price_sma20", "price_sma50", "sma20_sma50",
    "realized_vol_10", "realized_vol_21", "atr_pct_14",
    "volume_ratio_20", "distance_52w_high", "distance_52w_low",
)


@dataclass(frozen=True)
class AlphaDatasetConfig:
    horizon_days: int = 5
    price_column: str = "close"
    min_history: int = 252
    market_ticker: str = "MARKET"
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS

    def __post_init__(self) -> None:
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be > 0")
        if self.min_history <= 0:
            raise ValueError("min_history must be > 0")
        if not self.market_ticker:
            raise ValueError("market_ticker must not be empty")


class AlphaDatasetBuilder:
    """Build a cross-sectional, point-in-time Alpha research dataset."""

    REQUIRED_COLUMNS = {"high", "low", "close", "volume"}

    def __init__(self, config: AlphaDatasetConfig | None = None) -> None:
        self.config = config or AlphaDatasetConfig()

    def build(
        self,
        stock_data: Mapping[str, pd.DataFrame],
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build one row per date/ticker with backward-looking features."""
        market = self._prepare_frame(market_data, self.config.market_ticker)
        market_features = self._market_returns(market)

        rows: list[pd.DataFrame] = []
        for ticker, raw_data in stock_data.items():
            if not ticker:
                raise ValueError("ticker names must not be empty")
            stock = self._prepare_frame(raw_data, ticker)
            if len(stock) < self.config.min_history + self.config.horizon_days:
                continue
            frame = self._build_ticker_frame(ticker, stock, market_features)
            if not frame.empty:
                rows.append(frame)

        if not rows:
            raise ValueError("No valid Alpha dataset rows were produced")

        dataset = pd.concat(rows, axis=0, ignore_index=True)
        dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)

        required = list(self.config.feature_columns) + [
            "target_excess_return", "label_end_date"
        ]
        dataset = dataset.dropna(subset=required)

        if dataset.empty:
            raise ValueError("No valid Alpha dataset rows remain after NaN filtering")

        return dataset.set_index(["date", "ticker"])

    def _build_ticker_frame(
        self,
        ticker: str,
        stock: pd.DataFrame,
        market_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        price = stock[self.config.price_column]
        high = stock["high"]
        low = stock["low"]
        volume = stock["volume"]

        frame = pd.DataFrame(index=stock.index)
        frame["ticker"] = ticker

        # Features use only current/past observations.
        frame["return_5"] = price.pct_change(5)
        frame["return_10"] = price.pct_change(10)
        frame["return_20"] = price.pct_change(20)
        frame["return_60"] = price.pct_change(60)

        market = market_returns.reindex(stock.index)
        frame["relative_return_20"] = (
            frame["return_20"] - market["market_return_20"]
        )
        frame["relative_return_60"] = (
            frame["return_60"] - market["market_return_60"]
        )

        sma20 = price.rolling(20, min_periods=20).mean()
        sma50 = price.rolling(50, min_periods=50).mean()
        frame["price_sma20"] = price / sma20 - 1.0
        frame["price_sma50"] = price / sma50 - 1.0
        frame["sma20_sma50"] = sma20 / sma50 - 1.0

        daily_return = price.pct_change()
        frame["realized_vol_10"] = (
            daily_return.rolling(10, min_periods=10).std() * np.sqrt(252)
        )
        frame["realized_vol_21"] = (
            daily_return.rolling(21, min_periods=21).std() * np.sqrt(252)
        )

        previous_close = price.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr14 = true_range.rolling(14, min_periods=14).mean()
        frame["atr_pct_14"] = atr14 / price

        volume_mean = volume.rolling(20, min_periods=20).mean()
        frame["volume_ratio_20"] = volume / volume_mean

        rolling_high = price.rolling(252, min_periods=252).max()
        rolling_low = price.rolling(252, min_periods=252).min()
        frame["distance_52w_high"] = price / rolling_high - 1.0
        frame["distance_52w_low"] = price / rolling_low - 1.0

        # Labels are the only intentionally forward-looking fields.
        future_price = price.shift(-self.config.horizon_days)
        market_price = market_returns["market_price"].reindex(stock.index)
        future_market_price = market_price.shift(-self.config.horizon_days)

        stock_future_return = future_price / price - 1.0
        market_future_return = future_market_price / market_price - 1.0
        frame["target_excess_return"] = (
            stock_future_return - market_future_return
        )

        frame["label_end_date"] = pd.Series(
            stock.index.to_numpy(), index=stock.index
        ).shift(-self.config.horizon_days)

        frame["date"] = frame.index
        frame = frame.iloc[self.config.min_history - 1 :].copy()
        return frame.reset_index(drop=True)

    def _market_returns(self, market: pd.DataFrame) -> pd.DataFrame:
        price = market[self.config.price_column]
        return pd.DataFrame(
            {
                "market_price": price,
                "market_return_20": price.pct_change(20),
                "market_return_60": price.pct_change(60),
            },
            index=market.index,
        )

    def _prepare_frame(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"{ticker}: data must be a pandas DataFrame")

        missing = self.REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise ValueError(
                f"{ticker}: missing required columns: {sorted(missing)}"
            )

        if self.config.price_column not in data.columns:
            raise ValueError(
                f"{ticker}: missing configured price column "
                f"'{self.config.price_column}'"
            )

        frame = data.copy()
        try:
            frame.index = pd.to_datetime(frame.index)
        except Exception as exc:
            raise ValueError(f"{ticker}: index must be datetime-like") from exc

        if frame.index.tz is not None:
            frame.index = frame.index.tz_convert(None)

        frame = frame.sort_index()
        if frame.index.has_duplicates:
            raise ValueError(f"{ticker}: duplicate session dates are not allowed")

        numeric_columns = list(self.REQUIRED_COLUMNS | {self.config.price_column})
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame.dropna(subset=numeric_columns)
        if (frame["volume"] < 0).any():
            raise ValueError(f"{ticker}: volume cannot be negative")

        return frame


__all__ = [
    "DEFAULT_FEATURE_COLUMNS",
    "AlphaDatasetBuilder",
    "AlphaDatasetConfig",
]
