"""
Alpaca Client Module

SDK wrapper for Alpaca trading API with paper/live mode support.
Credentials are loaded from environment variables.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import os
from dotenv import load_dotenv

from alpaca.common.enums import BaseURL
from alpaca.trading import TradingClient
from alpaca.trading.enums import AccountStatus


load_dotenv(".env.secrets")


class AlpacaConnectionError(Exception):
    """Raised when Alpaca connection fails."""
    pass


class AlpacaHealthCheckFailed(Exception):
    """Raised when Alpaca health check fails."""
    pass


@dataclass
class AccountInfo:
    """Account information wrapper."""
    account_id: str
    status: str
    currency: str
    cash: float
    portfolio_value: float
    buying_power: float
    day_trading_buying_power: float
    equity: float
    last_equity: float
    multiplier: float
    pattern_day_trader: bool
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool


@dataclass
class PositionInfo:
    """Position information wrapper."""
    symbol: str
    quantity: float
    side: str
    avg_entry_price: float
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float
    current_price: float
    avg_entry_price_updated: float


@dataclass
class OrderInfo:
    """Order information wrapper."""
    id: str
    client_order_id: str
    symbol: str
    side: str
    type: str
    time_in_force: str
    status: str
    filled_avg_price: Optional[float]
    filled_qty: float
    created_at: datetime
    updated_at: datetime
    submitted_at: Optional[datetime]
    filled_at: Optional[datetime]
    expired_at: Optional[datetime]


@dataclass
class ClockInfo:
    """Market clock information wrapper."""
    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime


class AlpacaClient:
    """
    Alpaca SDK wrapper with health check and auto-reconnect.

    Supports both paper and live trading modes.
    """

    PAPER_URL = BaseURL.TRADING_PAPER
    LIVE_URL = BaseURL.TRADING_LIVE

    def __init__(
        self,
        paper_trading: Optional[bool] = None,
        base_url: Optional[BaseURL] = None,
    ):
        """
        Initialize Alpaca client.

        Args:
            paper_trading: If True, use paper trading API. If False, live.
                         Defaults to ALPACA_PAPER env var.
            base_url: Override base URL (useful for testing)
        """
        self._api_key = os.getenv("ALPACA_API_KEY")
        self._secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not self._api_key or not self._secret_key:
            raise AlpacaConnectionError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env.secrets"
            )

        if paper_trading is None:
            paper_str = os.getenv("ALPACA_PAPER", "true").lower()
            paper_trading = paper_str == "true"

        self._paper_trading = paper_trading
        self._base_url = base_url or (self.PAPER_URL if paper_trading else self.LIVE_URL)

        self._trading_client: Optional[TradingClient] = None
        self._connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._base_reconnect_delay = 1.0

    @property
    def is_paper_trading(self) -> bool:
        """Return True if running in paper trading mode."""
        return self._paper_trading

    @property
    def is_connected(self) -> bool:
        """Return True if connected to Alpaca."""
        return self._connected

    def connect(self) -> bool:
        """
        Connect to Alpaca and perform health check.

        Returns:
            True if connection successful

        Raises:
            AlpacaConnectionError: If connection fails
            AlpacaHealthCheckFailed: If health check fails
        """
        try:
            self._trading_client = TradingClient(
                self._api_key,
                self._secret_key,
                paper=self._paper_trading,
                url_override=str(self._base_url.value) if self._base_url else None,
            )

            self._health_check()
            self._connected = True
            self._reconnect_attempts = 0
            return True

        except Exception as e:
            self._connected = False
            raise AlpacaConnectionError(f"Failed to connect to Alpaca: {e}")

    def _health_check(self) -> None:
        """Perform health check by fetching account info."""
        if not self._trading_client:
            raise AlpacaConnectionError("Trading client not initialized")

        account = self._trading_client.get_account()
        if account.status != AccountStatus.ACTIVE:
            raise AlpacaHealthCheckFailed(
                f"Account status is {account.status}, not ACTIVE"
            )

    def reconnect(self) -> bool:
        """
        Attempt to reconnect with exponential backoff.

        Returns:
            True if reconnection successful
        """
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            return False

        self._reconnect_attempts += 1
        delay = self._base_reconnect_delay * (2 ** (self._reconnect_attempts - 1))

        import time
        time.sleep(delay)

        try:
            self.connect()
            return True
        except Exception:
            return False

    def get_account(self) -> AccountInfo:
        """
        Get current account information.

        Returns:
            AccountInfo with current account state

        Raises:
            AlpacaConnectionError: If not connected
        """
        if not self._connected or not self._trading_client:
            raise AlpacaConnectionError("Not connected to Alpaca")

        account = self._trading_client.get_account()

        return AccountInfo(
            account_id=account.id,
            status=account.status.value,
            currency=account.currency,
            cash=float(account.cash),
            portfolio_value=float(account.portfolio_value),
            buying_power=float(account.buying_power),
            day_trading_buying_power=float(account.daytrading_buying_power),
            equity=float(account.equity),
            last_equity=float(account.last_equity),
            multiplier=float(account.multiplier),
            pattern_day_trader=account.pattern_day_trader,
            trading_blocked=account.trading_blocked,
            transfers_blocked=account.transfers_blocked,
            account_blocked=account.account_blocked,
        )

    def get_positions(self) -> list[PositionInfo]:
        """
        Get all open positions.

        Returns:
            List of PositionInfo for open positions

        Raises:
            AlpacaConnectionError: If not connected
        """
        if not self._connected or not self._trading_client:
            raise AlpacaConnectionError("Not connected to Alpaca")

        positions = self._trading_client.get_all_positions()

        return [
            PositionInfo(
                symbol=p.symbol,
                quantity=float(p.qty),
                side=p.side.value,
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                cost_basis=float(p.cost_basis),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc),
                current_price=float(p.current_price),
                avg_entry_price_updated=float(p.avg_entry_price),
            )
            for p in positions
        ]

    def get_order_history(
        self,
        limit: int = 50,
        status: Optional[str] = None,
    ) -> list[OrderInfo]:
        """
        Get order history.

        Args:
            limit: Maximum number of orders to return
            status: Filter by order status (e.g., "filled", "canceled")

        Returns:
            List of OrderInfo for orders

        Raises:
            AlpacaConnectionError: If not connected
        """
        if not self._connected or not self._trading_client:
            raise AlpacaConnectionError("Not connected to Alpaca")

        orders = self._trading_client.get_orders(
            status=status,
            limit=limit,
        )

        return [
            OrderInfo(
                id=o.id,
                client_order_id=o.client_order_id,
                symbol=o.symbol,
                side=o.side.value,
                type=o.type.value,
                time_in_force=o.time_in_force.value,
                status=o.status.value,
                filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
                filled_qty=float(o.filled_qty),
                created_at=o.created_at,
                updated_at=o.updated_at,
                submitted_at=o.submitted_at,
                filled_at=o.filled_at,
                expired_at=o.expired_at,
            )
            for o in orders
        ]

    def is_market_open(self) -> bool:
        """
        Check if market is currently open.

        Returns:
            True if market is open

        Raises:
            AlpacaConnectionError: If not connected
        """
        clock = self.get_clock()
        return clock.is_open

    def get_clock(self) -> ClockInfo:
        """
        Get current market clock information.

        Returns:
            ClockInfo with market timing data

        Raises:
            AlpacaConnectionError: If not connected
        """
        if not self._connected or not self._trading_client:
            raise AlpacaConnectionError("Not connected to Alpaca")

        clock = self._trading_client.get_clock()

        return ClockInfo(
            timestamp=clock.timestamp,
            is_open=clock.is_open,
            next_open=clock.next_open,
            next_close=clock.next_close,
        )

    def get_available_margin(self) -> float:
        """
        Get available margin for trading.

        Returns:
            Available margin in dollars

        Raises:
            AlpacaConnectionError: If not connected
        """
        account = self.get_account()
        return account.buying_power


def confirm_live_trading() -> bool:
    """
    Prompt user for live trading confirmation.

    Returns:
        True if user confirms, False otherwise
    """
    print("\n" + "=" * 60)
    print("WARNING: LIVE TRADING MODE DETECTED")
    print("=" * 60)
    print("You are about to connect to Alpaca's LIVE trading API.")
    print("Real money will be at risk. Orders will execute against")
    print("real market conditions.")
    print("\nType 'YES I UNDERSTAND THE RISKS' to confirm:")
    response = input("> ").strip()
    return response == "YES I UNDERSTAND THE RISKS"


def create_alpaca_client(paper_trading: Optional[bool] = None) -> AlpacaClient:
    """
    Factory function to create and connect an AlpacaClient.

    Args:
        paper_trading: Override paper trading setting

    Returns:
        Connected AlpacaClient instance

    Raises:
        AlpacaConnectionError: If connection fails
    """
    if paper_trading is None:
        paper_str = os.getenv("ALPACA_PAPER", "true").lower()
        paper_trading = paper_str == "true"

    if not paper_trading:
        if not confirm_live_trading():
            raise AlpacaConnectionError("Live trading not confirmed")

    client = AlpacaClient(paper_trading=paper_trading)
    client.connect()
    return client