import unittest
from datetime import datetime, timezone

import pandas as pd

from core.regime_strategies import StrategyOrchestrator
from core.types import AllocationSignal, MarketRegimeState, StrategySignal


class StrategyContractTests(unittest.TestCase):
    def _allocation(self, bucket: str = "CALM", multiplier: float = 0.75) -> AllocationSignal:
        timestamp = datetime.now(timezone.utc)
        regime = MarketRegimeState(
            label=bucket,
            state_id=0,
            volatility_bucket=bucket,
            base_multiplier=multiplier,
            probability=0.9,
            state_probabilities=[0.9],
            timestamp=timestamp,
            is_confirmed=True,
            consecutive_bars=5,
            is_flickering=False,
        )
        return AllocationSignal(
            ticker="AAPL",
            final_multiplier=multiplier,
            market_regime=regime,
            base_multiplier=multiplier,
            vol_scalar=1.0,
            regime_label=bucket,
            beta=1.0,
            relative_vol=1.0,
            is_regime_confirmed=True,
            is_flickering=False,
            timestamp=timestamp,
            reasoning="test",
        )

    def _bars(self, n: int = 60) -> pd.DataFrame:
        close = pd.Series([100.0 + i * 0.25 for i in range(n)])
        return pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1_000_000.0,
            }
        )

    def test_strategy_emits_canonical_signal_without_sizing(self) -> None:
        allocation = self._allocation()
        signal = StrategyOrchestrator().generate_signal(allocation, self._bars())

        self.assertIsInstance(signal, StrategySignal)
        self.assertEqual(signal.ticker, "AAPL")
        self.assertNotIn("position_size_pct", signal.__dataclass_fields__)
        self.assertNotIn("leverage", signal.__dataclass_fields__)
        self.assertEqual(signal.metadata["allocation_cap_pct"], 0.75)

    def test_zero_allocation_does_not_create_new_strategy_intent(self) -> None:
        allocation = self._allocation(multiplier=0.0)
        signal = StrategyOrchestrator().generate_signal(allocation, self._bars())
        self.assertIsNone(signal)

    def test_strategy_selection_uses_volatility_bucket(self) -> None:
        orchestrator = StrategyOrchestrator()

        calm = orchestrator.get_strategy_for_allocation(self._allocation("CALM"))
        moderate = orchestrator.get_strategy_for_allocation(self._allocation("MODERATE"))
        turbulent = orchestrator.get_strategy_for_allocation(self._allocation("TURBULENT"))

        self.assertEqual(type(calm).__name__, "LowVolBullStrategy")
        self.assertEqual(type(moderate).__name__, "MidVolCautiousStrategy")
        self.assertEqual(type(turbulent).__name__, "HighVolDefensiveStrategy")


if __name__ == "__main__":
    unittest.main()
