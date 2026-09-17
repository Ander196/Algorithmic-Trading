"""
Position Tracker Module

Tracks positions with fill notifications via polling.
Updates PortfolioState and CircuitBreaker on every fill.
Provides per-position tracking with entry time/price, current price, unrealized P&L,
stop level, holding period, and regime tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable
import uuid

from alpaca.trading import TradingClient

from core.regime_strategies import Direction


@dataclass
class TrackedPosition:
    """Position with tracking information."""
    symbol: str
    direction: Direction
    quantity: float
    avg_entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pl: float
    unrealized_pl_pct: float
    market_value: float
    cost_basis: float
    stop_level: Optional[float] = None
    take_profit_level: Optional[float] = None
    regime_at_entry: str = "UNKNOWN"
    regime_id_at_entry: int = 0
    holding_period_seconds: float = 0.0
    fill_id: str = ""
    trade_id: str = ""

    @property
    def holding_period_hours(self) -> float:
        """Holding period in hours."""
        return self.holding_period_seconds / 3600

    @property
    def realized_pnl(self) -> float:
        """Realized P&L (placeholder for now)."""
        return 0.0


@dataclass
class FillEvent:
    """Fill event from order update."""
    symbol: str
    order_id: str
    client_order_id: str
    side: str
    qty: float
    price: float
    timestamp: datetime
    trade_id: Optional[str] = None


@dataclass
class PortfolioSnapshot:
    """Current portfolio state snapshot."""
    equity: float
    cash: float
    buying_power: float
    positions: list[TrackedPosition]
    total_unrealized_pl: float
    total_market_value: float
    timestamp: datetime


class PositionTracker:
    """
    Tracks positions with updates via polling.

    Features:
    - Periodic sync with broker for fill notifications
    - Update callbacks for PortfolioState and CircuitBreaker
    - Per-position tracking with regime and holding period
    - Sync with broker on startup (reconcile tracked vs actual)
    """

    def __init__(
        self,
        trading_client: TradingClient,
        on_fill_callback: Optional[Callable[[FillEvent, TrackedPosition], None]] = None,
        on_position_update_callback: Optional[Callable[[PortfolioSnapshot], None]] = None,
    ):
        """
        Initialize PositionTracker.

        Args:
            trading_client: Alpaca TradingClient
            on_fill_callback: Called on each fill event
            on_position_update_callback: Called on position updates
        """
        self._trading_client = trading_client
        self._on_fill_callback = on_fill_callback
        self._on_position_update_callback = on_position_update_callback

        self._tracked_positions: dict[str, TrackedPosition] = {}
        self._last_known_order_statuses: dict[str, str] = {}

    def sync_with_broker(self) -> list[TrackedPosition]:
        """
        Sync tracked positions with broker on startup.

        Reconciles tracked vs actual positions.

        Returns:
            List of synced TrackedPosition objects
        """
        self._tracked_positions.clear()

        broker_positions = self._trading_client.get_all_positions()

        for pos in broker_positions:
            tracked = TrackedPosition(
                symbol=pos.symbol,
                direction=Direction.LONG if pos.side.value == "long" else Direction.FLAT,
                quantity=float(pos.qty),
                avg_entry_price=float(pos.avg_entry_price),
                entry_time=datetime.now(timezone.utc),
                current_price=float(pos.current_price),
                unrealized_pl=float(pos.unrealized_pl),
                unrealized_pl_pct=float(pos.unrealized_plpc),
                market_value=float(pos.market_value),
                cost_basis=float(pos.cost_basis),
                trade_id=pos.client_order_id or str(uuid.uuid4()),
            )
            self._tracked_positions[pos.symbol] = tracked

        return list(self._tracked_positions.values())

    def check_for_fills(self) -> list[FillEvent]:
        """
        Check for new fills by comparing order statuses.

        Returns:
            List of new FillEvent objects
        """
        new_fills = []
        open_orders = self._trading_client.get_orders(status="open")

        for order in open_orders:
            if order.status.value == "filled" and order.id not in self._last_known_order_statuses:
                fill = FillEvent(
                    symbol=order.symbol,
                    order_id=order.id,
                    client_order_id=order.client_order_id,
                    side=order.side.value,
                    qty=float(order.filled_qty),
                    price=float(order.filled_avg_price),
                    timestamp=order.filled_at or datetime.now(timezone.utc),
                    trade_id=order.client_order_id,
                )
                new_fills.append(fill)
                self._handle_fill(fill)

            self._last_known_order_statuses[order.id] = order.status.value

        return new_fills

    def update_position(
        self,
        symbol: str,
        current_price: float,
        unrealized_pl: Optional[float] = None,
    ) -> Optional[TrackedPosition]:
        """
        Update position with current market data.

        Args:
            symbol: Position symbol
            current_price: Current market price
            unrealized_pl: Unrealized P&L (calculated if not provided)

        Returns:
            Updated TrackedPosition or None if not tracked
        """
        position = self._tracked_positions.get(symbol)
        if not position:
            return None

        position.current_price = current_price
        position.holding_period_seconds = (
            datetime.now(timezone.utc) - position.entry_time
        ).total_seconds()

        if unrealized_pl is not None:
            position.unrealized_pl = unrealized_pl

        position.unrealized_pl_pct = (
            position.unrealized_pl / position.cost_basis if position.cost_basis > 0 else 0
        )

        if self._on_position_update_callback:
            self._on_position_update_callback(self.get_portfolio_snapshot())

        return position

    def update_all_positions(self) -> None:
        """Update all positions with latest market data from broker."""
        broker_positions = self._trading_client.get_all_positions()

        for pos in broker_positions:
            if pos.symbol in self._tracked_positions:
                tracked = self._tracked_positions[pos.symbol]
                tracked.current_price = float(pos.current_price)
                tracked.unrealized_pl = float(pos.unrealized_pl)
                tracked.unrealized_pl_pct = float(pos.unrealized_plpc)
                tracked.market_value = float(pos.market_value)
                tracked.quantity = float(pos.qty)
                tracked.holding_period_seconds = (
                    datetime.now(timezone.utc) - tracked.entry_time
                ).total_seconds()

        if self._on_position_update_callback:
            self._on_position_update_callback(self.get_portfolio_snapshot())

    def add_position(
        self,
        symbol: str,
        direction: Direction,
        quantity: float,
        entry_price: float,
        regime_id: int = 0,
        regime_name: str = "UNKNOWN",
        trade_id: str = "",
        fill_id: str = "",
        stop_level: Optional[float] = None,
        take_profit_level: Optional[float] = None,
    ) -> TrackedPosition:
        """
        Add a new position to track.

        Args:
            symbol: Position symbol
            direction: Trade direction
            quantity: Number of shares
            entry_price: Entry price
            regime_id: HMM regime ID at entry
            regime_name: HMM regime name at entry
            trade_id: Trade ID linking to signal
            fill_id: Fill ID
            stop_level: Stop loss level
            take_profit_level: Take profit level

        Returns:
            New TrackedPosition
        """
        position = TrackedPosition(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            avg_entry_price=entry_price,
            entry_time=datetime.now(timezone.utc),
            current_price=entry_price,
            unrealized_pl=0.0,
            unrealized_pl_pct=0.0,
            market_value=quantity * entry_price,
            cost_basis=quantity * entry_price,
            regime_at_entry=regime_name,
            regime_id_at_entry=regime_id,
            trade_id=trade_id,
            fill_id=fill_id,
            stop_level=stop_level,
            take_profit_level=take_profit_level,
        )

        self._tracked_positions[symbol] = position
        return position

    def remove_position(self, symbol: str) -> Optional[TrackedPosition]:
        """
        Remove a position from tracking.

        Args:
            symbol: Position symbol to remove

        Returns:
            Removed TrackedPosition or None
        """
        return self._tracked_positions.pop(symbol, None)

    def get_position(self, symbol: str) -> Optional[TrackedPosition]:
        """Get tracked position for symbol."""
        return self._tracked_positions.get(symbol)

    def get_all_positions(self) -> list[TrackedPosition]:
        """Get all tracked positions."""
        return list(self._tracked_positions.values())

    def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        """Get current portfolio snapshot."""
        account = self._trading_client.get_account()

        positions = list(self._tracked_positions.values())
        total_unrealized_pl = sum(p.unrealized_pl for p in positions)
        total_market_value = sum(p.market_value for p in positions)

        return PortfolioSnapshot(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            positions=positions,
            total_unrealized_pl=total_unrealized_pl,
            total_market_value=total_market_value,
            timestamp=datetime.now(timezone.utc),
        )

    def check_regime_change(self, current_regime_id: int) -> dict[str, bool]:
        """
        Check which positions have regime mismatch.

        Args:
            current_regime_id: Current HMM regime ID

        Returns:
            Dict of symbol -> True if regime changed
        """
        return {
            pos.symbol: pos.regime_id_at_entry != current_regime_id
            for pos in self._tracked_positions.values()
        }

    def get_positions_by_regime(self, regime_id: int) -> list[TrackedPosition]:
        """Get positions entered in a specific regime."""
        return [
            pos for pos in self._tracked_positions.values()
            if pos.regime_id_at_entry == regime_id
        ]

    def _handle_fill(self, fill: FillEvent) -> None:
        """Handle fill event."""
        if fill.side.lower() in ("buy", "long"):
            direction = Direction.LONG
        else:
            direction = Direction.FLAT

        if direction != Direction.FLAT:
            self.add_position(
                symbol=fill.symbol,
                direction=direction,
                quantity=fill.qty,
                entry_price=fill.price,
                trade_id=fill.trade_id or "",
                fill_id=fill.order_id,
            )
        else:
            self.remove_position(fill.symbol)

        if self._on_fill_callback:
            position = self._tracked_positions.get(fill.symbol)
            self._on_fill_callback(fill, position)


class PositionTrackerError(Exception):
    """Raised for position tracker errors."""
    pass


def create_position_tracker(
    trading_client: TradingClient,
    on_fill_callback: Optional[Callable[[FillEvent, TrackedPosition], None]] = None,
    on_position_update_callback: Optional[Callable[[PortfolioSnapshot], None]] = None,
) -> PositionTracker:
    """Factory function to create PositionTracker."""
    return PositionTracker(
        trading_client=trading_client,
        on_fill_callback=on_fill_callback,
        on_position_update_callback=on_position_update_callback,
    )