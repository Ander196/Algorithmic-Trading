"""Alpha ranking and candidate selection.

This module converts model predictions into a small candidate universe. It does
not place trades, choose entries, or perform portfolio sizing.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.types import AllocationSignal


@dataclass(frozen=True)
class AlphaSelectionConfig:
    """Rules for turning Alpha predictions into candidates."""

    top_n: int = 20
    min_expected_excess_return: float = 0.0

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be > 0")


class AlphaSelector:
    """Rank stocks by predicted expected excess return.

    AllocationSignal remains an eligibility/risk-budget gate. It is not used
    to alter the Alpha score or determine final position size.
    """

    def __init__(self, config: AlphaSelectionConfig | None = None) -> None:
        self.config = config or AlphaSelectionConfig()

    def rank(
        self,
        predictions: pd.DataFrame,
        allocations: dict[str, AllocationSignal],
    ) -> pd.DataFrame:
        """Return the ranked candidate universe for one decision date."""
        required = {"ticker", "expected_excess_return"}
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
            rows["expected_excess_return"]
            >= self.config.min_expected_excess_return
        ]

        rows = rows.sort_values(
            ["expected_excess_return", "ticker"],
            ascending=[False, True],
        ).reset_index(drop=True)

        rows["alpha_rank"] = range(1, len(rows) + 1)
        return rows.head(self.config.top_n).copy()


__all__ = ["AlphaSelectionConfig", "AlphaSelector"]
