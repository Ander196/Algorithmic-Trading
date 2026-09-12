"""
HMM Model Module - Hidden Markov Model for Volatility Regime Detection

The HMM is a VOLATILITY CLASSIFIER. It detects whether the market is calm,
moderate, or turbulent volatility environment. It does NOT predict price
direction. The strategy layer uses the volatility classification to set
portfolio allocation - fully invested when calm, reduced when turbulent.
"""
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from data.indicators import calculateRsi, prepareFeaturesForHMM

try:
    from storage import storeHMMResult
    SUPPABASE_AVAILABLE = True
except ImportError:
    SUPPABASE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RegimeInfo:
    """Metadata for each regime state."""
    regime_id: int
    regime_name: str
    expected_return: float
    expected_volatility: float
    recommended_strategy_type: str
    max_leverage_allowed: float
    max_position_size_pct: float
    min_confidence_to_act: float


@dataclass
class RegimeState:
    """Current state of regime detection."""
    label: str
    state_id: int
    probability: float
    state_probabilities: np.ndarray
    timestamp: Optional[datetime] = None
    is_confirmed: bool = False
    consecutive_bars: int = 0


class HMMVolatilityClassifier:
    """
    Gaussian HMM for market volatility regime classification.

    Uses automatic model selection (BIC) to choose optimal number of regimes.
    Implements forward-only inference to avoid look-ahead bias.
    """

    REGIME_LABELS = {
        3: ['BEAR', 'NEUTRAL', 'BULL'],
        4: ['CRASH', 'BEAR', 'BULL', 'EUPHORIA'],
        5: ['CRASH', 'BEAR', 'NEUTRAL', 'BULL', 'EUPHORIA'],
        6: ['CRASH', 'STRONG_BEAR', 'WEAK_BEAR', 'WEAK_BULL', 'STRONG_BULL', 'EUPHORIA'],
        7: ['CRASH', 'STRONG_BEAR', 'WEAK_BEAR', 'NEUTRAL', 'WEAK_BULL', 'STRONG_BULL', 'EUPHORIA'],
    }

    def __init__(
        self,
        min_periods: int = 504,
        n_init: int = 10,
        confidence_threshold: float = 0.6,
        confirmation_bars: int = 3,
        flicker_threshold: int = 4,
        flicker_window: int = 20,
    ):
        """
        Initialize HMM classifier.

        Args:
            min_periods: Minimum trading days for training (default 504 = 2 years)
            n_init: Number of random initializations per model
            confidence_threshold: Minimum probability to act on regime signal
            confirmation_bars: Bars required to confirm regime change
            flicker_threshold: Max regime changes per window before triggering uncertainty
            flicker_window: Window size for flicker calculation
        """
        self.min_periods = min_periods
        self.n_init = n_init
        self.confidence_threshold = confidence_threshold
        self.confirmation_bars = confirmation_bars
        self.flicker_threshold = flicker_threshold
        self.flicker_window = flicker_window

        self.model: Optional[GaussianHMM] = None
        self.scaler: Optional[StandardScaler] = None
        self.n_regimes: int = 0
        self.bic_score: float = 0
        self.training_date: Optional[datetime] = None
        self.regime_labels: list[str] = []
        self.regime_info: dict[int, RegimeInfo] = {}

        self._previous_alpha: Optional[np.ndarray] = None
        self._previous_state: Optional[int] = None
        self._consecutive_bars: int = 0
        self._regime_history: list[int] = []
        self._last_confirmed_state: Optional[int] = None

    def fit(self, df: pd.DataFrame) -> 'HMMVolatilityClassifier':
        """
        Train HMM with automatic model selection.

        Tests n_components = [3,4,5,6,7] and selects lowest BIC.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            Self for method chaining
        """
        features = prepareFeaturesForHMM(df, self.min_periods)

        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        candidate_components = [3, 4, 5, 6, 7]
        results = []

        logger.info(f"Testing HMM with components: {candidate_components}")

        for n_comp in candidate_components:
            best_bic = np.inf
            best_model = None
            best_score = None

            for init in range(self.n_init):
                model = GaussianHMM(
                    n_components=n_comp,
                    covariance_type='full',
                    n_iter=1000,
                    random_state=42 + init,
                )

                try:
                    model.fit(features_scaled)
                    score = model.score(features_scaled)
                    n_params = self._countParameters(n_comp, features_scaled.shape[1])
                    n_samples = len(features_scaled)
                    bic = -2 * score * n_samples + n_params * np.log(n_samples)

                    if bic < best_bic:
                        best_bic = bic
                        best_model = model
                        best_score = score

                    logger.debug(
                        f"n_components={n_comp}, init={init+1}, "
                        f"log_likelihood={score:.2f}, BIC={bic:.2f}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to train n_components={n_comp}, init={init+1}: {e}")
                    continue

            if best_model is not None:
                results.append({
                    'n_components': n_comp,
                    'bic': best_bic,
                    'log_likelihood': best_score,
                    'model': best_model,
                })
                logger.info(f"n_components={n_comp}: BIC={best_bic:.2f}, log_likelihood={best_score:.2f}")

        if not results:
            raise ValueError("No valid HMM models could be trained")

        selected = min(results, key=lambda x: x['bic'])
        self.n_regimes = selected['n_components']
        self.model = selected['model']
        self.bic_score = selected['bic']

        logger.info(f"Selected n_components={self.n_regimes} with BIC={self.bic_score:.2f}")

        self._assignRegimeLabels(features_scaled)
        self.training_date = datetime.now()

        return self

    def _countParameters(self, n_components: int, n_features: int) -> int:
        """Count number of free parameters in GaussianHMM."""
        n_states = n_components

        startprob = n_states - 1
        transmat = n_states * (n_states - 1)
        means = n_states * n_features
        covars = n_states * n_features * (n_features + 1) // 2

        return startprob + transmat + means + covars

    def _assignRegimeLabels(self, features_scaled: np.ndarray):
        """Assign regime labels sorted by mean return (ascending)."""
        hidden_states = self.model.predict(features_scaled)

        state_returns = {}
        state_volatilities = {}

        for state in range(self.n_regimes):
            mask = hidden_states == state
            if mask.sum() > 0:
                state_returns[state] = features_scaled[mask, :3].mean()
                state_volatilities[state] = features_scaled[mask, :3].std()

        sorted_states = sorted(state_returns.items(), key=lambda x: x[1])
        self.regime_labels = self.REGIME_LABELS[self.n_regimes]

        strategy_types = ['mean_reversion', 'momentum', 'trend_following', 'volatility_arb']
        leverage_values = [1.0, 1.5, 2.0, 1.25]

        for idx, (state_id, _) in enumerate(sorted_states):
            label = self.regime_labels[idx]
            exp_return = state_returns[state_id]
            exp_vol = state_volatilities[state_id]

            if 'CRASH' in label or 'BEAR' in label:
                strat_type = 'mean_reversion'
                max_lev = 1.0
                max_pos = 0.25
            elif 'BULL' in label or 'EUPHORIA' in label:
                strat_type = 'trend_following'
                max_lev = leverage_values[idx % len(leverage_values)]
                max_pos = 0.75
            else:
                strat_type = strategy_types[idx % len(strategy_types)]
                max_lev = 1.25
                max_pos = 0.50

            self.regime_info[state_id] = RegimeInfo(
                regime_id=state_id,
                regime_name=label,
                expected_return=exp_return,
                expected_volatility=exp_vol,
                recommended_strategy_type=strat_type,
                max_leverage_allowed=max_lev,
                max_position_size_pct=max_pos,
                min_confidence_to_act=self.confidence_threshold,
            )

        self._state_id_to_label = {state_id: self.regime_labels[idx]
                                   for idx, (state_id, _) in enumerate(sorted_states)}

    def _computeEmissionProbability(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute P(observation | state) for each state using Gaussian emissions.

        Returns:
            Array of shape (n_states,) with emission probabilities
        """
        means = self.model.means_
        covars = self.model.covars_

        probabilities = np.zeros(self.n_regimes)
        for state in range(self.n_regimes):
            diff = observation - means[state]
            cov = covars[state]

            if cov.ndim == 1:
                cov = np.diag(cov)

            try:
                exponent = -0.5 * diff.T @ np.linalg.inv(cov) @ diff
                normalizer = 0.5 * np.log(np.linalg.det(2 * np.pi * cov))
                probabilities[state] = np.exp(exponent - normalizer)
            except np.linalg.LinAlgError:
                probabilities[state] = 1e-10

        probabilities = np.maximum(probabilities, 1e-10)
        return probabilities

    def predict_regime_filtered(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Compute P(state_t | observations_1:t) using forward algorithm.

        Uses ONLY past and present data - no future data. This avoids
        look-ahead bias that would occur if using model.predict().

        Args:
            features: DataFrame or array of shape (n_observations, n_features)

        Returns:
            Array of most likely state ID for each observation
        """
        if isinstance(features, pd.DataFrame):
            features = features.values

        features_scaled = self.scaler.transform(features)

        n_observations = len(features_scaled)
        predicted_states = np.zeros(n_observations, dtype=int)

        startprob = self.model.startprob_
        transmat = self.model.transmat_

        if self._previous_alpha is None or n_observations == 1:
            alpha = startprob * self._computeEmissionProbability(features_scaled[0])
            alpha = alpha / alpha.sum()
        else:
            alpha = self._previous_alpha.copy()

        for t in range(n_observations):
            if t > 0:
                alpha_forward = alpha @ transmat
                emission = self._computeEmissionProbability(features_scaled[t])
                alpha = alpha_forward * emission
                alpha = alpha / alpha.sum()

            alpha = np.maximum(alpha, 1e-10)
            alpha = alpha / alpha.sum()

            predicted_states[t] = np.argmax(alpha)

        self._previous_alpha = alpha.copy()
        self._previous_state = predicted_states[-1]

        return predicted_states

    def predict_regime_proba(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Get probability distribution over all states.

        Args:
            features: DataFrame or array of features up to current time

        Returns:
            Array of shape (n_states,) with probability for each state
        """
        if isinstance(features, pd.DataFrame):
            features = features.values

        features_scaled = self.scaler.transform(features)

        startprob = self.model.startprob_
        transmat = self.model.transmat_

        alpha = startprob * self._computeEmissionProbability(features_scaled[-1])
        alpha = alpha / alpha.sum()

        for t in range(len(features_scaled) - 1):
            alpha_forward = alpha @ transmat
            emission = self._computeEmissionProbability(features_scaled[t])
            alpha = alpha_forward * emission
            alpha = alpha / alpha.sum()

        return alpha

    def get_regime_stability(self) -> int:
        """Get number of consecutive bars in current regime."""
        if self._previous_state is None:
            return 0

        stability = 0
        for i in range(len(self._regime_history) - 1, -1, -1):
            if self._regime_history[i] == self._previous_state:
                stability += 1
            else:
                break
        return stability

    def get_transition_matrix(self) -> np.ndarray:
        """Get learned transition probability matrix."""
        return self.model.transmat_.copy()

    def detect_regime_change(self, features: pd.DataFrame) -> bool:
        """
        Detect if regime has changed (confirmed after N bars).

        Args:
            features: Latest features

        Returns:
            True only if regime change is confirmed
        """
        states = self.predict_regime_filtered(features)
        current_state = states[-1]

        if len(self._regime_history) > 0 and current_state != self._regime_history[-1]:
            self._consecutive_bars = 1
        elif len(self._regime_history) > 0 and current_state == self._regime_history[-1]:
            self._consecutive_bars += 1

        self._regime_history.append(current_state)

        if len(self._regime_history) > self.confirmation_bars:
            self._regime_history.pop(0)

        is_confirmed = (
            self._consecutive_bars >= self.confirmation_bars and
            current_state != self._last_confirmed_state
        )

        if is_confirmed:
            logger.warning(f"Regime change confirmed: {self._state_id_to_label.get(current_state, 'UNKNOWN')}")
            self._last_confirmed_state = current_state

        return is_confirmed

    def get_regime_flicker_rate(self) -> float:
        """
        Calculate regime changes per window (flicker rate).

        Returns:
            Number of regime changes in last flicker_window bars
        """
        if len(self._regime_history) < 2:
            return 0.0

        window = min(self.flicker_window, len(self._regime_history))
        recent_history = self._regime_history[-window:]
        changes = sum(1 for i in range(1, len(recent_history))
                     if recent_history[i] != recent_history[i-1])

        return changes

    def is_flickering(self) -> bool:
        """Check if flicker rate exceeds threshold."""
        return self.get_regime_flicker_rate() > self.flicker_threshold

    def get_current_regime_state(self, features: pd.DataFrame) -> RegimeState:
        """
        Get comprehensive current regime state.

        Args:
            features: Latest feature DataFrame

        Returns:
            RegimeState with all information
        """
        states = self.predict_regime_filtered(features)
        current_state = int(states[-1])
        probabilities = self.predict_regime_proba(features)

        label = self._state_id_to_label.get(current_state, 'UNKNOWN')
        confidence = probabilities[current_state]

        is_confirmed = self.get_regime_stability() >= self.confirmation_bars

        return RegimeState(
            label=label,
            state_id=current_state,
            probability=confidence,
            state_probabilities=probabilities,
            timestamp=datetime.now(),
            is_confirmed=is_confirmed,
            consecutive_bars=self.get_regime_stability(),
        )

    def get_volatility_sorting(self) -> list[tuple[int, float]]:
        """
        Get regime states sorted by VOLATILITY (not returns).

        The strategy layer uses this for allocation decisions.
        Labels are for human readability; volatility drives strategy.

        Returns:
            List of (state_id, mean_volatility) sorted ascending
        """
        volatility_map = {}
        for state_id, info in self.regime_info.items():
            volatility_map[state_id] = info.expected_volatility

        return sorted(volatility_map.items(), key=lambda x: x[1])

    def get_regime_for_allocation(self, features: pd.DataFrame) -> tuple[str, float]:
        """
        Get regime classification for portfolio allocation.

        Strategy layer uses this - sorts by VOLATILITY.

        Args:
            features: Latest features

        Returns:
            Tuple of (volatility_level, position_multiplier)
        """
        vol_sorted = self.get_volatility_sorting()
        current_state = self._previous_state

        if current_state is None:
            return ('UNKNOWN', 0.5)

        vol_rank = next(i for i, (sid, _) in enumerate(vol_sorted) if sid == current_state)
        n_levels = len(vol_sorted)

        if vol_rank < n_levels // 3:
            level = 'CALM'
            multiplier = 1.0
        elif vol_rank < 2 * n_levels // 3:
            level = 'MODERATE'
            multiplier = 0.75
        else:
            level = 'TURBULENT'
            multiplier = 0.50

        if self.is_flickering():
            logger.warning("Flickering detected - entering uncertainty mode")
            multiplier *= 0.75

        stability = self.get_regime_stability()
        if stability < self.confirmation_bars:
            multiplier *= 0.75

        return (level, multiplier)

    def store_to_supabase(
        self,
        ticker: str,
        features: pd.DataFrame,
    ) -> dict | None:
        """
        Store current HMM results to Supabase.

        Args:
            ticker: Stock ticker symbol
            features: Latest feature DataFrame

        Returns:
            The inserted record dict, or None if Supabase unavailable
        """
        if not SUPPABASE_AVAILABLE:
            logger.warning("Supabase not available, skipping storage")
            return None

        regime_state = self.get_current_regime_state(features)
        volatility_level, position_multiplier = self.get_regime_for_allocation(features)

        result = storeHMMResult(
            ticker=ticker,
            result_date=regime_state.timestamp or datetime.now(),
            regime_label=regime_state.label,
            state_id=regime_state.state_id,
            probability=regime_state.probability,
            state_probabilities=regime_state.state_probabilities.tolist(),
            volatility_level=volatility_level,
            position_multiplier=position_multiplier,
            is_confirmed=regime_state.is_confirmed,
            is_flickering=self.is_flickering(),
            stability_bars=regime_state.consecutive_bars,
            bic_score=self.bic_score,
            n_regimes=self.n_regimes,
            training_date=self.training_date,
        )

        logger.info(f"Stored HMM result for {ticker}: {regime_state.label}")
        return result

    def save_model(self, path: str | Path):
        """Save model to pickle file with metadata."""
        path = Path(path)
        metadata = {
            'n_regimes': self.n_regimes,
            'bic': self.bic_score,
            'training_date': self.training_date,
            'regime_labels': self.regime_labels,
            'regime_info': self.regime_info,
        }

        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'metadata': metadata,
            }, f)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str | Path):
        """Load model from pickle file."""
        path = Path(path)

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.scaler = data['scaler']
        metadata = data['metadata']

        self.n_regimes = metadata['n_regimes']
        self.bic_score = metadata['bic']
        self.training_date = metadata['training_date']
        self.regime_labels = metadata['regime_labels']
        self.regime_info = metadata['regime_info']

        self._state_id_to_label = {info.regime_id: info.regime_name
                                   for info in self.regime_info.values()}

        logger.info(f"Model loaded from {path}")


def train_hmm(df: pd.DataFrame, **kwargs) -> HMMVolatilityClassifier:
    """
    Convenience function to train HMM model.

    Args:
        df: DataFrame with OHLCV data
        **kwargs: Additional arguments for HMMVolatilityClassifier

    Returns:
        Trained HMMVolatilityClassifier
    """
    classifier = HMMVolatilityClassifier(**kwargs)
    return classifier.fit(df)


# =============================================================================
# Two-Layer Regime Engine (NEW)
# =============================================================================

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Union

import numpy as np
import pandas as pd
import supabase

from core.hmm_market import MarketRegimeClassifier
from core.stock_adjuster import StockVolatilityAdjuster
from core.types import AllocationSignal, MarketRegimeState, StockVolatilityProfile

logger = logging.getLogger("two_layer_regime")


class TwoLayerRegimeEngine:
    """
    Two-layer regime detection and allocation system.

    Combines:
    - Layer 1: MarketRegimeClassifier (macro market environment)
    - Layer 2: StockVolatilityAdjuster (per-stock adjustments)

    Produces AllocationSignal objects that combine both layers into
    a final_multiplier for position sizing.

    Supabase Integration:
    - Reads OHLCV data from stock_prices table
    - Writes AllocationSignal results to hmm_results table
    """

    def __init__(
        self,
        market_ticker: str = "SPY",
        min_confidence: float = 0.60,
        stability_bars: int = 5,
        vol_window: int = 21,
        beta_window: int = 63,
    ):
        """
        Initialize the two-layer regime engine.

        Args:
            market_ticker: Market index to use for Layer 1 (SPY, QQQ, etc.)
            min_confidence: Min probability to confirm regime
            stability_bars: Bars needed to confirm regime
            vol_window: Window for volatility calc (days)
            beta_window: Window for beta calc (days)
        """
        self.market_ticker = market_ticker

        # Layer 1: Market classifier
        self.market_classifier: Optional[MarketRegimeClassifier] = None

        # Layer 2: Per-stock adjusters
        self.stock_adjusters: dict[str, StockVolatilityAdjuster] = {}

        # Runtime state
        self.min_confidence = min_confidence
        self.stability_bars = stability_bars
        self.vol_window = vol_window
        self.beta_window = beta_window
        self.is_trained = False

        # Supabase client (lazy init)
        self._supabase_client: Optional[supabase.Client] = None

    # -------------------------------------------------------------------------
    # Supabase Integration
    # -------------------------------------------------------------------------

    def _get_supabase_client(self) -> supabase.Client:
        """Get or create Supabase client."""
        if self._supabase_client is not None:
            return self._supabase_client

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY (or SUPABASE_ANON_KEY) must be set in environment"
            )

        self._supabase_client = supabase.create_client(supabase_url, supabase_key)
        logger.info("Connected to Supabase")
        return self._supabase_client

    def load_stock_prices_from_supabase(
        self,
        ticker_or_tickers: Union[str, list[str]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from Supabase stock_prices table.

        Args:
            ticker_or_tickers: Single ticker or list of tickers
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            DataFrame with OHLCV data, indexed by date, sorted ascending
        """
        client = self._get_supabase_client()

        if isinstance(ticker_or_tickers, str):
            ticker_or_tickers = [ticker_or_tickers]

        all_data = []

        for ticker in ticker_or_tickers:
            try:
                query = client.table("stock_prices").select(
                    "ticker, price_date, open, high, low, close, volume"
                ).eq("ticker", ticker)

                if start_date:
                    query = query.gte("price_date", start_date.strftime("%Y-%m-%d"))
                if end_date:
                    query = query.lte("price_date", end_date.strftime("%Y-%m-%d"))

                query = query.order("price_date")
                response = query.execute()

                if response.data:
                    df = pd.DataFrame(response.data)
                    all_data.append(df)
                    logger.info(f"Loaded {len(df)} bars for {ticker}")
                else:
                    logger.warning(f"No data found for {ticker}")

            except Exception as e:
                logger.error(f"Failed to load data for {ticker}: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        # Combine and clean
        combined = pd.concat(all_data, ignore_index=True)
        combined["price_date"] = pd.to_datetime(combined["price_date"])
        combined = combined.set_index("price_date")
        combined = combined.sort_index()

        # Ensure lowercase columns
        combined.columns = [c.lower() for c in combined.columns]

        return combined

    # -------------------------------------------------------------------------
    # Model Training and Loading
    # -------------------------------------------------------------------------

    def train_market_model(
        self,
        market_df: pd.DataFrame,
        save_path: Optional[str] = None,
        **kwargs
    ) -> "TwoLayerRegimeEngine":
        """
        Train the market classifier (Layer 1) on index data.

        Args:
            market_df: OHLCV data for the market index
            save_path: Optional path to save trained model
            **kwargs: Additional args for MarketRegimeClassifier

        Returns:
            Self for chaining
        """
        logger.info(f"Training market model on {self.market_ticker}")

        self.market_classifier = MarketRegimeClassifier(
            n_init=kwargs.get("n_init", 10),
            min_confidence=self.min_confidence,
            stability_bars=self.stability_bars,
        )

        self.market_classifier.fit(market_df, self.market_ticker)

        if save_path:
            self.market_classifier.save_model(save_path)
            logger.info(f"Model saved to {save_path}")

        self.is_trained = True
        return self

    def load_market_model(self, path: str) -> "TwoLayerRegimeEngine":
        """
        Load a pre-trained market model from JSON.

        Args:
            path: Path to saved model JSON

        Returns:
            Self for chaining
        """
        self.market_classifier = MarketRegimeClassifier()
        self.market_classifier.load_model(path)
        self.market_ticker = self.market_classifier.market_ticker or self.market_ticker
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
        return self

    # -------------------------------------------------------------------------
    # Ticker Management
    # -------------------------------------------------------------------------

    def register_ticker(self, ticker: str) -> None:
        """
        Register a ticker for Layer 2 tracking.

        Args:
            ticker: Stock symbol to track
        """
        if ticker not in self.stock_adjusters:
            self.stock_adjusters[ticker] = StockVolatilityAdjuster(
                ticker=ticker,
                vol_window=self.vol_window,
                beta_window=self.beta_window,
            )
            logger.info(f"Registered ticker: {ticker}")

    def deregister_ticker(self, ticker: str) -> None:
        """
        Remove a ticker from Layer 2 tracking.

        Args:
            ticker: Stock symbol to remove
        """
        if ticker in self.stock_adjusters:
            del self.stock_adjusters[ticker]
            logger.info(f"Deregistered ticker: {ticker}")

    def register_tickers(self, tickers: list[str]) -> None:
        """Register multiple tickers at once."""
        for ticker in tickers:
            self.register_ticker(ticker)

    # -------------------------------------------------------------------------
    # Step Functions
    # -------------------------------------------------------------------------

    def step_market(self, market_bar: dict) -> MarketRegimeState:
        """
        Advance Layer 1 by one market bar.

        Args:
            market_bar: Dict with 'close' and 'timestamp' keys

        Returns:
            Current MarketRegimeState
        """
        if not self.is_trained or self.market_classifier is None:
            raise ValueError("Market model not trained. Call train_market_model() first.")

        return self.market_classifier.step(market_bar)

    def get_allocation(
        self,
        ticker: str,
        stock_close: float,
        index_close: float,
    ) -> Optional[AllocationSignal]:
        """
        Get allocation signal for a single ticker.

        Advances Layer 2 for the ticker and combines with current
        Layer 1 state to produce final multiplier.

        Args:
            ticker: Stock symbol
            stock_close: Stock's closing price
            index_close: Market index closing price

        Returns:
            AllocationSignal or None if not warmed up
        """
        # Ensure ticker is registered
        if ticker not in self.stock_adjusters:
            self.register_ticker(ticker)

        # Step Layer 2
        adjuster = self.stock_adjusters[ticker]
        vol_profile = adjuster.step(stock_close, index_close)

        # Get Layer 1 state (must already be stepped)
        market_state = self.market_classifier.get_current_state()

        if market_state is None:
            # Layer 1 not initialized yet
            return None

        if vol_profile is None:
            # Layer 2 not warmed up yet
            return None

        # Combine multipliers
        final_multiplier = market_state.base_multiplier * vol_profile.vol_scalar

        # Generate reasoning
        reasoning = self._generate_reasoning(
            ticker, market_state, vol_profile, final_multiplier
        )

        return AllocationSignal(
            ticker=ticker,
            final_multiplier=final_multiplier,
            market_regime=market_state,
            base_multiplier=market_state.base_multiplier,
            vol_scalar=vol_profile.vol_scalar,
            regime_label=market_state.label,
            beta=vol_profile.beta,
            relative_vol=vol_profile.relative_vol,
            is_regime_confirmed=market_state.is_confirmed,
            is_flickering=market_state.is_flickering,
            timestamp=vol_profile.timestamp,
            reasoning=reasoning,
        )

    def step_all(
        self,
        market_bar: dict,
        stock_closes: dict[str, float],
        index_close: float,
    ) -> dict[str, AllocationSignal]:
        """
        Convenience method to update everything in one call.

        Args:
            market_bar: Dict with 'close' and 'timestamp' for market index
            stock_closes: Dict mapping ticker -> close price
            index_close: Market index close price

        Returns:
            Dict mapping ticker -> AllocationSignal
        """
        # Step Layer 1
        self.step_market(market_bar)

        # Step Layer 2 for each ticker
        signals = {}
        for ticker, stock_close in stock_closes.items():
            signal = self.get_allocation(ticker, stock_close, index_close)
            if signal is not None:
                signals[ticker] = signal

        return signals

    def _generate_reasoning(
        self,
        ticker: str,
        market_state: MarketRegimeState,
        vol_profile: StockVolatilityProfile,
        final_multiplier: float,
    ) -> str:
        """Generate human-readable reasoning for the signal."""
        parts = []

        # Market regime
        parts.append(f"Market: {market_state.label}")
        if market_state.is_flickering:
            parts.append("(flickering)")
        elif market_state.is_confirmed:
            parts.append("(confirmed)")

        # Volatility
        parts.append(f"Vol: {vol_profile.relative_vol:.2f}x market")
        parts.append(f"Beta: {vol_profile.beta:.2f}")

        # Multiplier breakdown
        parts.append(f"Base: {market_state.base_multiplier:.2f}")
        parts.append(f"Vol scalar: {vol_profile.vol_scalar:.2f}")
        parts.append(f"Final: {final_multiplier:.2f}")

        return " | ".join(parts)

    # -------------------------------------------------------------------------
    # Daily Update Orchestration
    # -------------------------------------------------------------------------

    def run_daily_update(
        self,
        market_tickers: list[str] = None,
        registered_tickers: list[str] = None,
        lookback_days: int = 504,
    ) -> dict:
        """
        Orchestrate a daily update cycle.

        1. Pull latest market index bars
        2. Pull latest bars for registered tickers
        3. Step both layers
        4. Write results to Supabase hmm_results

        Args:
            market_tickers: List of market indexes to try (first successful used)
            registered_tickers: Tickers to process (uses registered if None)
            lookback_days: Days of history to fetch

        Returns:
            Dict with 'successes' and 'failures' counts
        """
        if market_tickers is None:
            market_tickers = [self.market_ticker]
        if registered_tickers is None:
            registered_tickers = list(self.stock_adjusters.keys())

        logger.info(f"Running daily update for {len(registered_tickers)} tickers")

        # Determine date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=lookback_days)

        # Load market data (try each ticker until one works)
        market_df = None
        used_market_ticker = None
        for mt in market_tickers:
            try:
                market_df = self.load_stock_prices_from_supabase(mt, start_date, end_date)
                if not market_df.empty:
                    used_market_ticker = mt
                    break
            except Exception as e:
                logger.warning(f"Failed to load {mt}: {e}")
                continue

        if market_df is None or market_df.empty:
            raise ValueError(f"Could not load market data from any ticker: {market_tickers}")

        # Update market ticker if different
        if used_market_ticker and used_market_ticker != self.market_ticker:
            logger.info(f"Using market data from {used_market_ticker}")
            self.market_ticker = used_market_ticker

        # Train or step market model
        if not self.is_trained:
            logger.info("Training market model...")
            self.train_market_model(market_df)

        # Always step through each bar to initialize forward algorithm
        for idx in range(len(market_df)):
            bar = market_df.iloc[idx]
            self.step_market({
                "close": bar["close"],
                "timestamp": market_df.index[idx],
            })

        # Get current market state
        market_state = self.market_classifier.get_current_state()
        if market_state is None:
            raise ValueError("Market model not producing states")

        # Pre-warm Layer 2 adjusters with historical data
        logger.info("Warming up Layer 2 adjusters...")
        for ticker in registered_tickers:
            try:
                stock_df = self.load_stock_prices_from_supabase(ticker, start_date, end_date)
                if stock_df.empty:
                    logger.warning(f"No data for {ticker}, skipping warm-up")
                    continue

                # Step through historical bars to warm up adjuster
                min_len = min(len(stock_df), len(market_df))
                for i in range(min_len):
                    stock_close = stock_df.iloc[i]["close"]
                    index_close = market_df.iloc[i]["close"]
                    self.stock_adjusters[ticker].step(stock_close, index_close, stock_df.index[i])

                logger.info(f"Warmed up {ticker}: {len(self.stock_adjusters[ticker].stock_prices)} bars")
            except Exception as e:
                logger.warning(f"Failed to warm up {ticker}: {e}")

        # Load and process each ticker
        client = self._get_supabase_client()
        successes = 0
        failures = 0

        for ticker in registered_tickers:
            try:
                # Load stock data
                stock_df = self.load_stock_prices_from_supabase(ticker, start_date, end_date)
                if stock_df.empty:
                    logger.warning(f"No data for {ticker}, skipping")
                    failures += 1
                    continue

                # Get latest prices
                stock_close = float(stock_df.iloc[-1]["close"])
                index_close = float(market_df.iloc[-1]["close"])

                # Get allocation signal
                signal = self.get_allocation(ticker, stock_close, index_close)

                if signal is None:
                    logger.warning(f"Signal not ready for {ticker}")
                    failures += 1
                    continue

                # Write to Supabase
                self._write_allocation_signal(client, signal)
                successes += 1
                logger.info(f"Saved signal for {ticker}: multiplier={signal.final_multiplier:.2f}")

            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                failures += 1
                continue

        result = {
            "successes": successes,
            "failures": failures,
            "total": len(registered_tickers),
            "market_state": market_state.label if market_state else "UNKNOWN",
        }

        logger.info(f"Daily update complete: {successes} success, {failures} failures")
        return result

    def _write_allocation_signal(
        self,
        client: supabase.Client,
        signal: AllocationSignal,
    ) -> None:
        """Write allocation signal to Supabase hmm_results table."""
        from datetime import date

        data = {
            "ticker": signal.ticker,
            "result_date": signal.timestamp.date().isoformat(),
            "regime_label": signal.regime_label,
            "state_id": signal.market_regime.state_id,
            "probability": signal.market_regime.probability,
            "state_probabilities": signal.market_regime.state_probabilities,
            "volatility_level": signal.market_regime.volatility_bucket,
            "position_multiplier": signal.final_multiplier,
            # Extended fields for Two-Layer system
            "final_multiplier": signal.final_multiplier,
            "base_multiplier": signal.base_multiplier,
            "vol_scalar": signal.vol_scalar,
            "beta": signal.beta,
            "relative_vol": signal.relative_vol,
            "is_regime_confirmed": signal.is_regime_confirmed,
            "is_flickering": signal.is_flickering,
            "reasoning": signal.reasoning,
            "stability_bars": signal.market_regime.consecutive_bars,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Use upsert on (ticker, result_date)
        client.table("hmm_results").upsert(data, on_conflict="ticker,result_date").execute()