"""Backward-compatible facade for the Layer 2 stock risk model.

The canonical implementation now lives in :mod:`core.stock_risk`.
"""
from core.stock_risk import StockRiskConfig, StockRiskModel


class StockVolatilityAdjuster(StockRiskModel):
    """Deprecated compatibility name for ``StockRiskModel``."""

    def __init__(
        self,
        ticker: str,
        vol_window: int = 21,
        beta_window: int = 63,
        min_scalar: float = 0.25,
        max_scalar: float = 1.0,
    ) -> None:
        super().__init__(
            ticker,
            StockRiskConfig(
                vol_window=vol_window,
                beta_window=beta_window,
                min_scalar=min_scalar,
                max_scalar=max_scalar,
            ),
        )
