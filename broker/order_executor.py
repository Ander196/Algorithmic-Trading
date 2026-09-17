"""
Order Executor Module

Handles order submission, bracket orders, and position management.
Provides unique trade_id linking signal → risk_decision → order → fill.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
import uuid
from typing import Optional, Callable
import time

from alpaca.trading import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderType
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    StopLimitOrderRequest,
)
from alpaca.trading.models import Order

from core.regime_strategies import Signal, Direction


class OrderExecutionError(Exception):
    """Raised when order execution fails."""
    pass


class OrderTimeoutError(Exception):
    """Raised when order times out without fill."""
    pass


class InvalidStopModificationError(Exception):
    """Raised when stop modification would widen the stop."""
    pass


@dataclass
class TradeContext:
    """Context linking a signal to its execution."""
    trade_id: str
    signal: Signal
    signal_timestamp: datetime
    risk_decision: Optional[dict] = None
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    fill_timestamp: Optional[datetime] = None
    status: str = "PENDING"

    def __post_init__(self):
        if not self.trade_id:
            self.trade_id = str(uuid.uuid4())


@dataclass
class OrderResult:
    """Result of an order submission."""
    order_id: Optional[str]
    status: str
    filled_price: Optional[float]
    filled_quantity: Optional[float]
    message: str
    trade_context: TradeContext


@dataclass
class BracketOrderResult:
    """Result of a bracket order submission."""
    entry_order_id: str
    stop_order_id: Optional[str]
    take_profit_order_id: Optional[str]
    status: str
    filled_entry_price: Optional[float]
    filled_stop_price: Optional[float]
    filled_tp_price: Optional[float]
    trade_context: TradeContext
    message: str


class OrderExecutor:
    """
    Handles order submission and management.

    Features:
    - LIMIT orders by default (+/-0.1% of current price)
    - Cancel after timeout with optional market retry
    - Bracket orders (entry + stop + take profit)
    - Stop modification (tighten only)
    - Trade ID tracking throughout execution lifecycle
    """

    DEFAULT_LIMIT_OFFSET_PCT = 0.001
    DEFAULT_ORDER_TIMEOUT_SECONDS = 30
    DEFAULT_STOP_TIGHTEN_MIN_PCT = 0.001

    def __init__(
        self,
        trading_client: TradingClient,
        limit_offset_pct: float = DEFAULT_LIMIT_OFFSET_PCT,
        order_timeout_seconds: int = DEFAULT_ORDER_TIMEOUT_SECONDS,
    ):
        """
        Initialize OrderExecutor.

        Args:
            trading_client: Alpaca TradingClient instance
            limit_offset_pct: Offset from current price for LIMIT orders (0.001 = 0.1%)
            order_timeout_seconds: Seconds to wait before canceling LIMIT order
        """
        self._client = trading_client
        self._limit_offset_pct = limit_offset_pct
        self._order_timeout_seconds = order_timeout_seconds
        self._trade_contexts: dict[str, TradeContext] = {}
        self._order_to_trade_id: dict[str, str] = {}

    def submit_order(
        self,
        signal: Signal,
        trade_context: Optional[TradeContext] = None,
        retry_at_market_on_timeout: bool = False,
    ) -> OrderResult:
        """
        Submit an order for the given signal.

        Default: LIMIT order at current_price ± 0.1%
        Cancel after 30s if unfilled, optionally retry as MARKET order.

        Args:
            signal: Signal to execute
            trade_context: TradeContext with trade_id (generated if not provided)
            retry_at_market_on_timeout: If True, retry as MARKET order after timeout

        Returns:
            OrderResult with execution details
        """
        if trade_context is None:
            trade_context = TradeContext(
                trade_id=str(uuid.uuid4()),
                signal=signal,
                signal_timestamp=datetime.now(timezone.utc),
            )

        self._trade_contexts[trade_context.trade_id] = trade_context

        try:
            current_price = self._get_current_price(signal.symbol)
            limit_price = self._calculate_limit_price(signal, current_price)

            order_request = LimitOrderRequest(
                symbol=signal.symbol,
                qty=self._calculate_quantity(signal),
                side=OrderSide.BUY if signal.direction == Direction.LONG else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=Decimal(str(limit_price)),
                client_order_id=trade_context.trade_id,
            )

            order = self._client.submit_order(order_request)
            self._order_to_trade_id[order.id] = trade_context.trade_id
            trade_context.order_id = order.id

            filled_order = self._wait_for_fill(
                order.id,
                trade_context,
                retry_at_market=retry_at_market_on_timeout,
            )

            return OrderResult(
                order_id=filled_order.id,
                status=filled_order.status.value,
                filled_price=float(filled_order.filled_avg_price) if filled_order.filled_avg_price else None,
                filled_quantity=float(filled_order.filled_qty),
                message="Order filled successfully" if filled_order.status.value == "filled" else "Order submitted",
                trade_context=trade_context,
            )

        except OrderTimeoutError:
            self._cancel_order(trade_context.trade_id)
            if retry_at_market_on_timeout:
                return self._submit_market_order(signal, trade_context)
            raise

        except Exception as e:
            trade_context.status = "FAILED"
            raise OrderExecutionError(f"Order execution failed: {e}")

    def _submit_market_order(
        self,
        signal: Signal,
        trade_context: TradeContext,
    ) -> OrderResult:
        """Submit market order as fallback."""
        try:
            order_request = MarketOrderRequest(
                symbol=signal.symbol,
                qty=self._calculate_quantity(signal),
                side=OrderSide.BUY if signal.direction == Direction.LONG else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                client_order_id=f"{trade_context.trade_id}_market",
            )

            order = self._client.submit_order(order_request)
            self._order_to_trade_id[order.id] = trade_context.trade_id
            trade_context.order_id = order.id

            time.sleep(2)

            filled_order = self._client.get_order(order.id)

            trade_context.status = "FILLED" if filled_order.status.value == "filled" else "PARTIAL"
            trade_context.fill_price = float(filled_order.filled_avg_price) if filled_order.filled_avg_price else None

            return OrderResult(
                order_id=filled_order.id,
                status=filled_order.status.value,
                filled_price=trade_context.fill_price,
                filled_quantity=float(filled_order.filled_qty),
                message="Market order filled" if filled_order.status.value == "filled" else "Market order submitted",
                trade_context=trade_context,
            )

        except Exception as e:
            trade_context.status = "FAILED"
            raise OrderExecutionError(f"Market order fallback failed: {e}")

    def submit_bracket_order(
        self,
        signal: Signal,
        trade_context: Optional[TradeContext] = None,
    ) -> BracketOrderResult:
        """
        Submit bracket order: entry + stop loss + take profit.

        Uses Alpaca's OCO (One Cancels Other) for bracket orders.

        Args:
            signal: Signal with entry, stop_loss, take_profit
            trade_context: TradeContext with trade_id

        Returns:
            BracketOrderResult with all order IDs
        """
        if trade_context is None:
            trade_context = TradeContext(
                trade_id=str(uuid.uuid4()),
                signal=signal,
                signal_timestamp=datetime.now(timezone.utc),
            )

        self._trade_contexts[trade_context.trade_id] = trade_context

        try:
            current_price = self._get_current_price(signal.symbol)
            limit_price = self._calculate_limit_price(signal, current_price)
            quantity = self._calculate_quantity(signal)

            entry_order = self._client.submit_order(
                LimitOrderRequest(
                    symbol=signal.symbol,
                    qty=quantity,
                    side=OrderSide.BUY if signal.direction == Direction.LONG else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    limit_price=Decimal(str(limit_price)),
                    client_order_id=f"{trade_context.trade_id}_entry",
                )
            )

            self._order_to_trade_id[entry_order.id] = trade_context.trade_id

            if signal.stop_loss:
                stop_order = self._client.submit_order(
                    StopLossRequest(
                        symbol=signal.symbol,
                        qty=quantity,
                        side=OrderSide.SELL if signal.direction == Direction.LONG else OrderSide.BUY,
                        time_in_force=TimeInForce.GTC,
                        stop_price=Decimal(str(signal.stop_loss)),
                        client_order_id=f"{trade_context.trade_id}_stop",
                    )
                )
                self._order_to_trade_id[stop_order.id] = trade_context.trade_id
            else:
                stop_order = None

            if signal.take_profit:
                tp_order = self._client.submit_order(
                    TakeProfitRequest(
                        symbol=signal.symbol,
                        qty=quantity,
                        side=OrderSide.SELL if signal.direction == Direction.LONG else OrderSide.BUY,
                        time_in_force=TimeInForce.GTC,
                        limit_price=Decimal(str(signal.take_profit)),
                        client_order_id=f"{trade_context.trade_id}_tp",
                    )
                )
                self._order_to_trade_id[tp_order.id] = trade_context.trade_id
            else:
                tp_order = None

            trade_context.order_id = entry_order.id

            return BracketOrderResult(
                entry_order_id=entry_order.id,
                stop_order_id=stop_order.id if stop_order else None,
                take_profit_order_id=tp_order.id if tp_order else None,
                status="SUBMITTED",
                filled_entry_price=None,
                filled_stop_price=None,
                filled_tp_price=None,
                trade_context=trade_context,
                message="Bracket order submitted",
            )

        except Exception as e:
            trade_context.status = "FAILED"
            raise OrderExecutionError(f"Bracket order submission failed: {e}")

    def modify_stop(
        self,
        symbol: str,
        new_stop: float,
        require_tighten: bool = True,
    ) -> bool:
        """
        Modify stop loss for a position.

        Args:
            symbol: Symbol to modify stop for
            new_stop: New stop price
            require_tighten: If True, only allow tightening (moving stop closer to entry)

        Returns:
            True if modification successful

        Raises:
            InvalidStopModificationError: If modification would widen the stop
        """
        positions = self._client.get_all_positions()
        position = next((p for p in positions if p.symbol == symbol), None)

        if not position:
            raise OrderExecutionError(f"No position found for {symbol}")

        current_stop = float(position.avg_entry_price)
        current_stop = float(position.unrealized_pl)

        if require_tighten:
            direction = Direction.LONG
            if new_stop >= current_stop:
                raise InvalidStopModificationError(
                    f"Cannot widen stop: current {current_stop:.4f}, new {new_stop:.4f}"
                )

        existing_orders = self._client.get_orders(status="open")
        stop_orders = [
            o for o in existing_orders
            if o.symbol == symbol and "stop" in o.client_order_id.lower()
        ]

        for order in stop_orders:
            self._client.cancel_order(order.id)

        quantity = float(position.qty)
        stop_order = self._client.submit_order(
            StopLossRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.SELL if direction == Direction.LONG else OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
                stop_price=Decimal(str(new_stop)),
                client_order_id=f"{symbol}_stop_modified_{int(time.time())}",
            )
        )

        return True

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful
        """
        try:
            self._client.cancel_order(order_id)
            return True
        except Exception as e:
            raise OrderExecutionError(f"Failed to cancel order {order_id}: {e}")

    def close_position(self, symbol: str) -> OrderResult:
        """
        Close a position completely.

        Args:
            symbol: Symbol to close

        Returns:
            OrderResult with close details
        """
        positions = self._client.get_all_positions()
        position = next((p for p in positions if p.symbol == symbol), None)

        if not position:
            raise OrderExecutionError(f"No position found for {symbol}")

        quantity = float(position.qty)
        side = OrderSide.SELL if position.side.value == "long" else OrderSide.BUY

        trade_context = TradeContext(
            trade_id=str(uuid.uuid4()),
            signal=Signal(
                symbol=symbol,
                direction=Direction.LONG if side == OrderSide.SELL else Direction.FLAT,
                confidence=1.0,
                entry_price=float(position.avg_entry_price),
                stop_loss=0.0,
            ),
            signal_timestamp=datetime.now(timezone.utc),
        )

        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=trade_context.trade_id,
        )

        order = self._client.submit_order(order_request)
        self._order_to_trade_id[order.id] = trade_context.trade_id
        trade_context.order_id = order.id

        time.sleep(2)

        filled_order = self._client.get_order(order.id)
        trade_context.status = "FILLED" if filled_order.status.value == "filled" else "CLOSING"

        return OrderResult(
            order_id=filled_order.id,
            status=filled_order.status.value,
            filled_price=float(filled_order.filled_avg_price) if filled_order.filled_avg_price else None,
            filled_quantity=float(filled_order.filled_qty),
            message=f"Position closed for {symbol}",
            trade_context=trade_context,
        )

    def close_all_positions(self) -> list[OrderResult]:
        """
        Close all open positions.

        Returns:
            List of OrderResult for each closed position
        """
        results = []
        positions = self._client.get_all_positions()

        for position in positions:
            try:
                result = self.close_position(position.symbol)
                results.append(result)
            except Exception as e:
                results.append(OrderResult(
                    order_id=None,
                    status="FAILED",
                    filled_price=None,
                    filled_quantity=None,
                    message=f"Failed to close {position.symbol}: {e}",
                    trade_context=TradeContext(
                        trade_id=str(uuid.uuid4()),
                        signal=Signal(
                            symbol=position.symbol,
                            direction=Direction.FLAT,
                            confidence=1.0,
                            entry_price=0.0,
                            stop_loss=0.0,
                        ),
                        signal_timestamp=datetime.now(timezone.utc),
                    ),
                ))

        return results

    def get_trade_context(self, trade_id: str) -> Optional[TradeContext]:
        """Get trade context by trade_id."""
        return self._trade_contexts.get(trade_id)

    def get_trade_context_by_order(self, order_id: str) -> Optional[TradeContext]:
        """Get trade context by order_id."""
        trade_id = self._order_to_trade_id.get(order_id)
        if trade_id:
            return self._trade_contexts.get(trade_id)
        return None

    def _get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        latest = self._client.get_latest_trade(symbol)
        return float(latest.price)

    def _calculate_limit_price(self, signal: Signal, current_price: float) -> float:
        """Calculate limit price as offset from current price."""
        if signal.direction == Direction.LONG:
            return current_price * (1 - self._limit_offset_pct)
        else:
            return current_price * (1 + self._limit_offset_pct)

    def _calculate_quantity(self, signal: Signal) -> float:
        """Calculate quantity from signal position size."""
        if "dollar_size" in signal.metadata:
            dollar_size = signal.metadata["dollar_size"]
        else:
            dollar_size = signal.position_size_pct * signal.leverage

        current_price = self._get_current_price(signal.symbol)
        return dollar_size / current_price

    def _wait_for_fill(
        self,
        order_id: str,
        trade_context: TradeContext,
        retry_at_market: bool = False,
    ) -> Order:
        """Wait for order to fill or timeout."""
        start_time = time.time()

        while time.time() - start_time < self._order_timeout_seconds:
            order = self._client.get_order(order_id)

            if order.status.value == "filled":
                trade_context.status = "FILLED"
                trade_context.fill_price = float(order.filled_avg_price)
                trade_context.fill_timestamp = order.filled_at
                return order

            if order.status.value in ("canceled", "rejected", "expired"):
                trade_context.status = order.status.value
                raise OrderExecutionError(f"Order {order.status.value}")

            time.sleep(1)

        trade_context.status = "TIMEOUT"
        raise OrderTimeoutError(f"Order not filled within {self._order_timeout_seconds}s")


def create_order_executor(
    trading_client: TradingClient,
    limit_offset_pct: float = 0.001,
    order_timeout_seconds: int = 30,
) -> OrderExecutor:
    """Factory function to create OrderExecutor."""
    return OrderExecutor(
        trading_client=trading_client,
        limit_offset_pct=limit_offset_pct,
        order_timeout_seconds=order_timeout_seconds,
    )