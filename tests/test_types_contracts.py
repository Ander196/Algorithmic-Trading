import unittest
from datetime import datetime, timezone

from core.types import (
    AllocationSignal,
    CircuitBreakerLevel,
    MarketRegimeState,
    RiskDecision,
    RiskDecisionStatus,
    StrategySignal,
    TradeDirection,
)


class TypesContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.timestamp = datetime.now(timezone.utc)
        self.market_state = MarketRegimeState(
            label="TURBULENT",
            state_id=2,
            volatility_bucket="HIGH",
            base_multiplier=0.5,
            probability=0.9,
            state_probabilities=[0.05, 0.05, 0.9],
            timestamp=self.timestamp,
            is_confirmed=True,
            consecutive_bars=5,
            is_flickering=False,
        )

    def test_allocation_signal_remains_the_layer_2_output(self) -> None:
        signal = AllocationSignal(
            ticker="AAPL",
            final_multiplier=0.35,
            market_regime=self.market_state,
            base_multiplier=0.5,
            vol_scalar=0.7,
            regime_label="TURBULENT",
            beta=1.1,
            relative_vol=1.2,
            is_regime_confirmed=True,
            is_flickering=False,
            timestamp=self.timestamp,
            reasoning="Layer 1 cap combined with Layer 2 volatility scalar.",
        )

        self.assertEqual(signal.ticker, "AAPL")
        self.assertAlmostEqual(signal.final_multiplier, 0.35)

    def test_strategy_signal_contains_intent_not_portfolio_sizing(self) -> None:
        signal = StrategySignal(
            ticker="AAPL",
            direction=TradeDirection.LONG,
            entry_price=250.0,
            stop_loss=242.0,
            take_profit=266.0,
            confidence=0.82,
            strategy_name="TrendStrategy",
            timestamp=self.timestamp,
            reasoning="Trend condition satisfied.",
        )

        self.assertEqual(signal.direction, TradeDirection.LONG)
        self.assertEqual(signal.strategy_name, "TrendStrategy")
        self.assertNotIn("position_size_pct", signal.__dataclass_fields__)
        self.assertNotIn("leverage", signal.__dataclass_fields__)

    def test_risk_decision_can_represent_approved_sizing(self) -> None:
        strategy_signal = StrategySignal(
            ticker="AAPL",
            direction=TradeDirection.LONG,
            entry_price=250.0,
            stop_loss=242.0,
            take_profit=266.0,
            confidence=0.82,
            strategy_name="TrendStrategy",
            timestamp=self.timestamp,
            reasoning="Trend condition satisfied.",
        )

        decision = RiskDecision(
            status=RiskDecisionStatus.APPROVED,
            strategy_signal=strategy_signal,
            quantity=10.0,
            notional=2500.0,
            risk_amount=80.0,
            risk_pct=0.008,
            allocation_cap_pct=0.35,
            allocation_cap_notional=3500.0,
            rejection_reason=None,
            modifications=(),
            circuit_breaker_level=CircuitBreakerLevel.NONE,
        )

        self.assertEqual(decision.status, RiskDecisionStatus.APPROVED)
        self.assertEqual(decision.quantity, 10.0)
        self.assertEqual(decision.circuit_breaker_level, CircuitBreakerLevel.NONE)


if __name__ == "__main__":
    unittest.main()
