"""Alpha ranking and candidate selection.

This module converts model scores into a small candidate universe. It does not
place trades, choose entries, or perform portfolio sizing.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.types import AllocationSignal


@dataclass(frozen=True)
class AlphaSelectionConfig:
    """Rules for turning Alpha scores into candidates."""

    top_n: int = 20
    min_probability_positive: float = 0.50

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be > 0")
        if not 0.0 <= self.min_probability_positive <= 1.0:
            raise ValueError("min_probability_positive must be between 0 and 1")


class AlphaSelector:
    """Rank stocks by Alpha while respecting Layer 2 eligibility."""

    def __init__(self, config: AlphaSelectionConfig | None = None) -> None:
        self.config = config or AlphaSelectionConfig()

    def rank(
        self,
        predictions: pd.DataFrame,
        allocations: dict[str, AllocationSignal],
    ) -> pd.DataFrame:
        """Return the ranked candidate universe for one decision date.

        Only tickers with a positive AllocationSignal budget are eligible.
        The Alpha model determines ranking; AllocationSignal acts as an
        upstream risk/exposure gate and is not converted into position size.
        """
        required = {"ticker", "alpha_score", "probability_positive"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"predictions missing required columns: {sorted(missing)}")

        rows = predictions.copy()
        rows["allocation_multiplier"] = rows["ticker"].map(
            lambda ticker: allocations[ticker].final_multiplier
            if ticker in allocations
            else float("nan")
        )
        rows = rows.dropna(subset=["allocation_multiplier"])
        rows = rows[rows["allocation_multiplier"] > 0.0]
        rows = rows[
            rows["probability_positive"]
            >= self.config.min_probability_positive
        ]

        rows = rows.sort_values(
            ["alpha_score", "probability_positive", "ticker"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        rows["alpha_rank"] = range(1, len(rows) + 1)
        return rows.head(self.config.top_n).copy()


__all__ = ["AlphaSelectionConfig", "AlphaSelector"]
