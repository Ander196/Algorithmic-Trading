"""
core/stock_adjuster.py - Layer 2: Per-Stock Volatility Adjuster

Maintains rolling volatility and beta calculations for each stock relative
to the market index. Combines both into a vol_scalar that adjusts the
base multiplier from Layer 1.

Key design:
- Rolling deque for memory efficiency
- Returns None during warm-up (insufficient history)
- Never produces unreliable early estimates
"""
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from core.types import StockVolatilityProfile

logger = logging.getLogger(__name__)


class StockVolatilityAdjuster:
    """
    Per-stock volatility and beta calculator.

    Maintains rolling history of stock and index prices to compute:
    1. Realized volatility (annualized)
    2. Relative volatility (stock / market)
    3. Rolling beta via OLS
    4. Combined vol_scalar for position sizing

    The vol_scalar penalizes stocks with higher relative volatility
    and higher beta (more market exposure during turbulent times).
    """

    def __init__(
        self,
        ticker: str,
        vol_window: int = 21,
        beta_window: int = 63,
        min_scalar: float = 0.25,
        max_scalar: float = 1.0,
    ):
        """
        Initialize the stock volatility adjuster.

        Args:
            ticker: Stock symbol
            vol_window: Window for realized volatility calculation (days)
            beta_window: Window for beta calculation (days)
            min_scalar: Minimum vol_scalar (clipped)
            max_scalar: Maximum vol_scalar (clipped)
        """
        self.ticker = ticker
        self.vol_window = vol_window
        self.beta_window = beta_window
        self.min_scalar = min_scalar
        self.max_scalar = max_scalar

        # Rolling price history (close prices)
        self.stock_prices: deque = deque(maxlen=beta_window + 1)
        self.index_prices: deque = deque(maxlen=beta_window + 1)
        self.timestamps: deque = deque(maxlen=beta_window + 1)

        # Computed values (updated each step)
        self.current_profile: Optional[StockVolatilityProfile] = None

    def step(self, stock_close: float, index_close: float, timestamp: Optional[datetime] = None) -> Optional[StockVolatilityProfile]:
        """
        Add a new price observation and compute updated volatility profile.

        Args:
            stock_close: Stock's closing price
            index_close: Market index closing price
            timestamp: When this price was observed

        Returns:
            Updated StockVolatilityProfile or None if not warmed up
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Add to rolling history
        self.stock_prices.append(stock_close)
        self.index_prices.append(index_close)
        self.timestamps.append(timestamp)

        # Need at least vol_window + 1 for volatility
        if len(self.stock_prices) < self.vol_window + 1:
            return None

        # Convert to numpy arrays
        stock_arr = np.array(self.stock_prices)
        index_arr = np.array(self.index_prices)

        # Compute log returns
        stock_returns = np.diff(np.log(stock_arr))
        index_returns = np.diff(np.log(index_arr))

        # Align lengths
        min_len = min(len(stock_returns), len(index_returns))
        stock_returns = stock_returns[-min_len:]
        index_returns = index_returns[-min_len:]

        # Compute realized volatility (annualized)
        stock_vol = self._realized_volatility(stock_returns)
        market_vol = self._realized_volatility(index_returns)

        # Compute relative volatility
        relative_vol = stock_vol / market_vol if market_vol > 0 else 1.0

        # Compute beta via OLS: cov(stock, index) / var(index)
        beta = self._rolling_beta(stock_returns, index_returns)

        # Clip beta to reasonable range
        beta = np.clip(beta, 0.0, 5.0)

        # Compute vol_scalar from both factors
        vol_scalar = self._compute_vol_scalar(relative_vol, beta)

        # Store current profile
        self.current_profile = StockVolatilityProfile(
            ticker=self.ticker,
            realised_vol=stock_vol,
            market_vol=market_vol,
            relative_vol=relative_vol,
            beta=beta,
            vol_scalar=vol_scalar,
            timestamp=timestamp,
        )

        return self.current_profile

    def _realized_volatility(self, returns: np.ndarray) -> float:
        """
        Compute annualized realized volatility.

        Uses the most recent vol_window returns.
        """
        if len(returns) < 2:
            return 0.0

        # Use most recent vol_window returns
        recent_returns = returns[-self.vol_window :]

        if len(recent_returns) < 2:
            return 0.0

        # Standard deviation of returns
        std = np.std(recent_returns, ddof=1)

        if np.isnan(std) or std == 0:
            return 0.0

        # Annualize (trading days)
        annual_vol = std * np.sqrt(252)

        return annual_vol

    def _rolling_beta(self, stock_returns: np.ndarray, index_returns: np.ndarray) -> float:
        """
        Compute rolling beta via OLS.

        beta = cov(stock, market) / var(market)
        """
        if len(stock_returns) < 2 or len(index_returns) < 2:
            return 1.0

        # Use most recent beta_window returns
        stock_rets = stock_returns[-self.beta_window :]
        index_rets = index_returns[-self.beta_window :]

        if len(stock_rets) < 10:  # Need sufficient data
            return 1.0

        # Compute covariance and variance
        cov = np.cov(stock_rets, index_rets, ddof=1)
        if cov.shape != (2, 2):
            return 1.0

        var_index = cov[1, 1]
        cov_stock_index = cov[0, 1]

        if var_index == 0:
            return 1.0

        beta = cov_stock_index / var_index

        return beta if not np.isnan(beta) else 1.0

    def _compute_vol_scalar(self, relative_vol: float, beta: float) -> float:
        """
        Compute vol_scalar from relative volatility and beta.

        Uses weighted penalty functions:
        - High relative vol -> penalize (lower scalar)
        - High beta -> penalize (lower scalar during turbulent times)

        Formula: penalty = 1 / max(ratio, 1.0), then clip to [min_scalar, max_scalar]
        """
        # Relative volatility penalty
        # If stock vol > market vol, reduce allocation
        if relative_vol > 1.0:
            vol_penalty = 1.0 / relative_vol
        else:
            vol_penalty = 1.0  # Stock less volatile than market = no penalty

        # Beta penalty
        # Higher beta = more exposure to market moves
        # During turbulent times, higher beta is penalized more
        # Simple approach: penalty = 1 / (1 + (beta - 1) * 0.5)
        if beta > 1.0:
            beta_penalty = 1.0 / (1.0 + (beta - 1.0) * 0.3)
        else:
            beta_penalty = 1.0  # Low beta = no penalty

        # Combined scalar
        vol_scalar = vol_penalty * beta_penalty

        # Clip to bounds
        vol_scalar = np.clip(vol_scalar, self.min_scalar, self.max_scalar)

        return vol_scalar

    def get_profile(self) -> Optional[StockVolatilityProfile]:
        """
        Get current volatility profile without advancing (side-effect-free).

        Returns:
            Current StockVolatilityProfile or None if not initialized
        """
        return self.current_profile

    def is_warmed_up(self) -> bool:
        """
        Check if the adjuster has enough history to produce estimates.

        Returns:
            True if warmed up
        """
        return len(self.stock_prices) >= self.vol_window + 1

    def reset(self) -> None:
        """
        Clear all history and start fresh.

        Useful when switching to a new time period or after a gap.
        """
        self.stock_prices.clear()
        self.index_prices.clear()
        self.timestamps.clear()
        self.current_profile = None