"""
Market Data Module

Provides market data fetching via Alpaca.
Handles historical bars, latest bar, quote, and snapshot.
Gracefully handles gaps (weekends, holidays, halts).
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable
import asyncio

import pandas as pd
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestBarRequest,
    StockLatestQuoteRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame

from alpaca.trading import TradingClient


class MarketDataError(Exception):
    """Raised for market data errors."""
    pass


class GapDetectedError(Exception):
    """Raised when a gap is detected in market data."""
    pass


@dataclass
class BarData:
    """OHLCV bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class QuoteData:
    """Quote data (bid/ask)."""
    timestamp: datetime
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def spread_pct(self) -> float:
        return self.spread / self.ask_price if self.ask_price > 0 else 0


@dataclass
class SnapshotData:
    """Snapshot data for a symbol."""
    symbol: str
    latest_bar: Optional[BarData]
    latest_quote: Optional[QuoteData]
    bid_price: float
    ask_price: float
    last_price: float
    open_price: float
    high_price: float
    low_price: float
    prev_close: float
    volume: int
    timestamp: datetime


def _parse_timeframe(tf_str: str) -> TimeFrame:
    """Parse timeframe string to Alpaca TimeFrame."""
    mapping = {
        "1Min": TimeFrame.Minute,
        "5Min": TimeFrame(5, "Min"),
        "15Min": TimeFrame(15, "Min"),
        "1H": TimeFrame.Hour,
        "1D": TimeFrame.Day,
    }
    return mapping.get(tf_str, TimeFrame.Minute)


class MarketDataClient:
    """
    Market data client for fetching market data.

    Features:
    - Historical bars with configurable timeframe
    - Latest bar, quote, and snapshot
    - Gap detection (weekends, holidays, halts)
    """

    def __init__(
        self,
        trading_client: Optional[TradingClient] = None,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper_trading: bool = True,
    ):
        """
        Initialize MarketDataClient.

        Args:
            trading_client: Existing Alpaca TradingClient
            api_key: Alpaca API key (if not using trading_client)
            secret_key: Alpaca secret key
            paper_trading: Use paper trading data
        """
        if trading_client:
            self._api_key = trading_client._api_key
            self._secret_key = trading_client._secret_key
        else:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            self._api_key = api_key or os.getenv("ALPACA_API_KEY")
            self._secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        self._paper_trading = paper_trading
        self._data_client = None

        self._latest_bars: dict[str, BarData] = {}
        self._latest_quotes: dict[str, QuoteData] = {}

        if not self._api_key or not self._secret_key:
            raise MarketDataError("API credentials required")

    def _get_data_client(self):
        """Get or create data client."""
        if self._data_client is None:
            from alpaca.data import StockHistoricalDataClient
            self._data_client = StockHistoricalDataClient(
                self._api_key,
                self._secret_key,
            )
        return self._data_client

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV bars.

        Args:
            symbol: Ticker symbol
            timeframe: Timeframe (1Min, 5Min, 15Min, 1H, 1D)
            start: Start datetime (defaults to 1 day ago)
            end: End datetime (defaults to now)
            limit: Max number of bars

        Returns:
            DataFrame with OHLCV columns (lowercase)
        """
        data_client = self._get_data_client()

        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            start = end - timedelta(days=1)

        timeframe_obj = _parse_timeframe(timeframe)

        request = GetHistoricalBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe_obj,
            start=start,
            end=end,
            limit=limit,
        )

        bars = data_client.get_stock_bars(request)

        if symbol not in bars or not bars[symbol]:
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars[symbol]
        ])

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()

        return df

    def get_latest_bar(self, symbol: str) -> Optional[BarData]:
        """
        Get latest bar for symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            BarData or None
        """
        if symbol in self._latest_bars:
            return self._latest_bars[symbol]

        data_client = self._get_data_client()

        request = GetLatestBarRequest(symbol_or_symbols=symbol)

        try:
            bars = data_client.get_stock_latest_bar(request)
            if symbol in bars:
                bar = bars[symbol]
                bar_data = BarData(
                    timestamp=bar.timestamp,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )
                self._latest_bars[symbol] = bar_data
                return bar_data
        except Exception:
            pass

        return None

    def get_latest_quote(self, symbol: str) -> Optional[QuoteData]:
        """
        Get latest quote for symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            QuoteData or None
        """
        if symbol in self._latest_quotes:
            return self._latest_quotes[symbol]

        data_client = self._get_data_client()

        request = GetLatestQuoteRequest(symbol_or_symbols=symbol)

        try:
            quotes = data_client.get_stock_latest_quote(request)
            if symbol in quotes:
                q = quotes[symbol]
                quote_data = QuoteData(
                    timestamp=q.timestamp,
                    symbol=symbol,
                    bid_price=q.bid_price,
                    ask_price=q.ask_price,
                    bid_size=q.bid_size,
                    ask_size=q.ask_size,
                )
                self._latest_quotes[symbol] = quote_data
                return quote_data
        except Exception:
            pass

        return None

    def get_snapshot(self, symbol: str) -> Optional[SnapshotData]:
        """
        Get snapshot (full market data) for symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            SnapshotData or None
        """
        data_client = self._get_data_client()

        request = GetSnapshotRequest(
            symbol_or_symbols=symbol,
        )

        try:
            snapshots = data_client.get_stock_snapshot(request)
            if symbol in snapshots:
                s = snapshots[symbol]
                return SnapshotData(
                    symbol=symbol,
                    latest_bar=self._bar_to_data(s.latest_bar) if s.latest_bar else None,
                    latest_quote=self._quote_to_data(s.latest_quote) if s.latest_quote else None,
                    bid_price=s.bid_price,
                    ask_price=s.ask_price,
                    last_price=s.last_price,
                    open_price=s.open_price,
                    high_price=s.high_price,
                    low_price=s.low_price,
                    prev_close=s.prev_close_price,
                    volume=s.volume,
                    timestamp=datetime.now(timezone.utc),
                )
        except Exception:
            pass

        return None

    def check_for_gaps(
        self,
        symbol: str,
        max_gap_pct: float = 0.05,
    ) -> Optional[dict]:
        """
        Check for gaps in market data.

        Args:
            symbol: Ticker symbol
            max_gap_pct: Maximum acceptable gap percentage

        Returns:
            Dict with gap info or None if no gap
        """
        bars = self.get_historical_bars(symbol, limit=5)

        if len(bars) < 2:
            return None

        bars = bars.sort_index()

        prev_close = bars["close"].iloc[-2]
        curr_open = bars["open"].iloc[-1]

        gap_pct = abs(curr_open - prev_close) / prev_close

        if gap_pct > max_gap_pct:
            return {
                "symbol": symbol,
                "prev_close": prev_close,
                "curr_open": curr_open,
                "gap_pct": gap_pct,
                "gap_type": "up" if curr_open > prev_close else "down",
                "timestamp": bars.index[-1],
            }

        return None

    def is_market_hours(self, symbol: str) -> bool:
        """
        Check if market is open for the symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            True if market is open
        """
        client = TradingClient(self._api_key, self._secret_key, paper=self._paper_trading)

        try:
            clock = client.get_clock()
            return clock.is_open
        except Exception:
            return False

    def _bar_to_data(self, bar) -> BarData:
        """Convert Alpaca bar to BarData."""
        return BarData(
            timestamp=bar.timestamp,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
        )

    def _quote_to_data(self, quote) -> QuoteData:
        """Convert Alpaca quote to QuoteData."""
        return QuoteData(
            timestamp=quote.timestamp,
            symbol=quote.symbol,
            bid_price=quote.bid_price,
            ask_price=quote.ask_price,
            bid_size=quote.bid_size,
            ask_size=quote.ask_size,
        )


def create_market_data_client(
    trading_client: Optional[TradingClient] = None,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    paper_trading: bool = True,
) -> MarketDataClient:
    """Factory function to create MarketDataClient."""
    return MarketDataClient(
        trading_client=trading_client,
        api_key=api_key,
        secret_key=secret_key,
        paper_trading=paper_trading,
    )