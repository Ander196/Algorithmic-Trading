"""
core/hmm_market.py — Layer 1: Market-Wide Volatility Regime Classifier
=======================================================================

WHAT THIS MODULE DOES
---------------------
Trains a single Gaussian HMM on a broad market index (SPY, QQQ, etc.) and
classifies the overall market environment into three coarse volatility buckets:

    CALM       → base_multiplier = 1.00  (fully invested)
    MODERATE   → base_multiplier = 0.75  (reduced exposure)
    TURBULENT  → base_multiplier = 0.50  (defensive)

This base_multiplier is then refined by Layer 2 (stock_adjuster.py) using
per-stock beta and realised-vol metrics to produce a final position size.

WHY ONE MODEL FOR THE ENTIRE MARKET?
--------------------------------------
Training a separate HMM per stock is statistically unreliable: with 2 years
of daily data you may only see 20–40 bars in a turbulent regime — far too few
to estimate a reliable covariance matrix. Training on SPY gives the same rich
history to every regime, produces economically interpretable transitions (2008
crash, 2017 calm, 2020 COVID spike), and results are consistent across tickers.

KEY DESIGN DECISIONS (with reasons)
-------------------------------------
1.  FULL covariance matrices — features like log-return and realised-vol are
    correlated; diagonal covariance ignores that and produces worse state
    separation. "full" is the professional standard for low-dimensional feature
    sets (≤10 features).

2.  BIC model selection over n_states ∈ {3,4,5,6,7} — penalises complexity
    so we don't overfit to noise in the training history.

3.  Genuinely random EM restarts — each restart uses np.random.randint() so
    the optimiser explores different basins. Fixed seeds (42+i) make every
    restart identical, defeating the purpose.

4.  Regime labels sorted by ACTUAL state volatility — we run Viterbi once at
    training time to assign bars to states, then compute std(raw log-returns)
    per state. The state with the lowest std gets label "CALM". This is stable
    across retraining runs because it uses real data, not scaled features.

5.  Forward algorithm (filtering) at inference — uses only past and present
    observations; never future ones. Viterbi (used by hmmlearn.predict()) uses
    the full sequence and would introduce look-ahead bias in backtesting.

6.  Single step() entry point — the ONLY method allowed to mutate internal
    state. All query methods are pure (no side effects), so calling them in
    any order or any number of times is safe.

7.  JSON persistence — stores raw numpy arrays as plain lists. No pickle,
    no version lock, no class-refactoring breakage.

PROFESSIONAL FEATURE SET
--------------------------
The feature vector fed to the HMM has 7 dimensions, matching what systematic
macro funds and risk-capital desks typically use:

    1. log_return        : Daily log-return — captures direction and magnitude.
    2. realised_vol_21   : 21-day rolling std × √252 — annualised daily vol.
    3. vol_of_vol        : Std of realised_vol over 21 days — "vol of vol",
                           the main driver of regime transitions.
    4. hl_range          : (high - low) / close — intraday range, a fast
                           stress indicator that reacts before close-to-close vol.
    5. overnight_gap     : open/prev_close − 1 — captures gap risk and
                           after-hours sentiment shock.
    6. momentum_20       : 20-day price momentum (close / close[−20] − 1).
                           Trend signal; helps separate crash regimes from
                           low-vol sideways markets.
    7. volume_ratio      : volume / 20-day avg volume. Elevated volume during
                           down-moves is a classic stress signal.

USAGE
-----
    from core.hmm_market import MarketRegimeClassifier

    # ── One-time training (offline, e.g. weekly) ─────────────────────────────
    clf = MarketRegimeClassifier()
    clf.fit(spy_df, market_ticker="SPY")
    clf.save_model("models/market_regime.json")

    # ── Live inference (once per bar, after market close) ────────────────────
    clf = MarketRegimeClassifier()
    clf.load_model("models/market_regime.json")

    # Warm up by replaying recent history (step() once per historical bar)
    for _, row in recent_spy_df.iterrows():
        clf.step(row)

    # On each new bar:
    state = clf.step(new_bar)          # MarketRegimeState
    print(state.volatility_bucket)    # 'CALM' | 'MODERATE' | 'TURBULENT'
    print(state.base_multiplier)      # 1.0 | 0.75 | 0.50 (+ penalties)
"""

from __future__ import annotations

import json
import logging
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from core.types import MarketRegimeState, RegimeInfo

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# How much to reduce the base_multiplier for each uncertainty condition.
# These are multiplicative — both can apply simultaneously.
_FLICKER_PENALTY    = 0.75   # regime changing too frequently
_UNCONFIRMED_PENALTY = 0.75  # regime not yet held for stability_bars

# Base multipliers per volatility bucket. The strategy layer receives these
# and further adjusts them via per-stock vol_scalar (Layer 2).
REGIME_MULTIPLIERS: dict[str, float] = {
    "CALM":      1.00,
    "MODERATE":  0.75,
    "TURBULENT": 0.50,
}

# Minimum number of training observations required.
# 504 ≈ 2 calendar years of daily bars.
MIN_TRAIN_BARS = 504

# Candidate numbers of HMM states to evaluate during BIC model selection.
BIC_CANDIDATE_STATES = [3, 4, 5, 6, 7]

# Feature column names in the order they appear in the feature matrix.
# Order matters: the scaler, means_, and covars_ arrays are indexed this way.
FEATURE_COLS = [
    "log_return",       # 1. Close-to-close log-return
    "realised_vol_21",  # 2. 21-day annualised realised volatility
    "vol_of_vol",       # 3. 21-day std of realised_vol (vol-of-vol)
    "hl_range",         # 4. Intraday high-low range / close
    "overnight_gap",    # 5. Open / prev_close − 1
    "momentum_20",      # 6. 20-day price momentum
    "volume_ratio",     # 7. Volume / 20-day mean volume
]

N_FEATURES = len(FEATURE_COLS)  # 7


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class MarketRegimeClassifier:
    """
    Gaussian HMM that classifies the broad market into volatility regimes.

    Trained once on SPY (or any liquid index), then used at inference time
    to classify each new daily bar into CALM / MODERATE / TURBULENT and
    produce a base_multiplier for position sizing.

    The classifier is stateful: call step() once per new bar to advance
    the forward algorithm and update internal counters. All other methods
    are read-only (pure).

    Typical lifecycle
    -----------------
    1.  fit(spy_df)                       ← train once offline
    2.  save_model("market_regime.json")  ← persist
    3.  load_model("market_regime.json")  ← restore on next run
    4.  for row in warmup_df: step(row)   ← warm up forward algorithm
    5.  state = step(live_row)            ← one call per new live bar
    """

    def __init__(
        self,
        n_init: int = 10,
        min_confidence: float = 0.60,
        stability_bars: int = 5,
        flicker_window: int = 20,
        flicker_threshold: int = 4,
    ):
        """
        Parameters
        ----------
        n_init            : Number of independent EM restarts per candidate
                            n_states during BIC model selection. Each restart
                            uses a fresh random seed. More restarts → better
                            chance of finding the global optimum, at the cost
                            of training time. 10 is a practical minimum.
        min_confidence    : Posterior probability P(state | obs_1..t) below
                            which we treat the current state as uncertain.
                            Exposed to the engine layer; not used internally.
        stability_bars    : Consecutive bars the same state must appear before
                            we call the regime "confirmed". Until confirmed,
                            the base_multiplier is penalised by _UNCONFIRMED_PENALTY.
        flicker_window    : Rolling window (bars) over which we count regime
                            changes for flicker detection.
        flicker_threshold : Maximum regime changes within flicker_window before
                            we declare the model "flickering" and apply
                            _FLICKER_PENALTY to the base_multiplier.
        """
        # ── Hyperparameters ──────────────────────────────────────────────────
        self.n_init = n_init
        self.min_confidence = min_confidence
        self.stability_bars = stability_bars
        self.flicker_window = flicker_window
        self.flicker_threshold = flicker_threshold

        # ── Model weights (populated by fit() or load_model()) ───────────────
        # We store raw numpy arrays instead of a GaussianHMM object so that
        # JSON serialisation is straightforward and there is no dependency on
        # a specific hmmlearn version at load time.
        self.n_states: int = 0
        self.startprob_: Optional[np.ndarray] = None    # shape (n_states,)
        self.transmat_: Optional[np.ndarray] = None     # shape (n_states, n_states)
        self.means_: Optional[np.ndarray] = None        # shape (n_states, N_FEATURES)
        self.covars_: Optional[np.ndarray] = None       # shape (n_states, N_FEATURES, N_FEATURES)

        # ── Scaler parameters (fitted on training data, applied at inference) ─
        # We store mean and scale as plain arrays rather than an sklearn object
        # so save/load stays trivial JSON with no sklearn version dependency.
        self.scaler_mean_: Optional[np.ndarray] = None   # shape (N_FEATURES,)
        self.scaler_scale_: Optional[np.ndarray] = None  # shape (N_FEATURES,)

        # ── Regime metadata (set by _assign_regime_labels after fit()) ────────
        # regime_info maps raw HMM state_id → RegimeInfo.
        # state_order is the list of state_ids sorted by volatility ascending.
        self.regime_info: dict[int, RegimeInfo] = {}
        self.state_order: list[int] = []

        # ── Training provenance ───────────────────────────────────────────────
        self.bic_score: float = 0.0
        self.training_date: Optional[datetime] = None
        self.market_ticker: Optional[str] = None

        # ── Live-inference state (mutated ONLY by step()) ─────────────────────
        # alpha_ is the forward variable: P(state | obs_1..t), shape (n_states,).
        # It is updated one observation at a time via the forward algorithm.
        self._alpha: Optional[np.ndarray] = None
        self._current_state: Optional[int] = None
        self._consecutive_bars: int = 0
        # Deque of (state_id) for the last flicker_window bars.
        self._state_history: deque[int] = deque(maxlen=flicker_window)
        self._last_confirmed_state: Optional[int] = None
        self._last_timestamp: Optional[datetime] = None

        # Flag set to True after a successful fit() or load_model().
        self.is_fitted: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Feature engineering
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive the 7-dimensional feature vector from raw OHLCV data.

        This method is static so it can be called without an instance —
        useful for pre-processing training and inference data consistently.

        Steps
        -----
        1.  Validate that all required OHLCV columns are present.
        2.  Compute each feature on a copy of the DataFrame so the original
            is never modified.
        3.  Drop rows with NaN values (present at the start due to rolling
            windows). The longest window is 21 bars, so we lose ~21 rows at
            the top of every dataset.
        4.  Return only the 7 feature columns in FEATURE_COLS order. Column
            order is critical because means_ and covars_ are indexed positionally.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close, volume.
            Index should be dates (any frequency, but daily is assumed).

        Returns
        -------
        pd.DataFrame
            Rows × 7 feature columns, NaN-free. Returns an empty DataFrame
            if fewer than 25 rows are available (cannot compute any rolling stat).
        """
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        if len(df) < 25:
            # Not enough rows to compute even the shortest rolling window.
            logger.warning("compute_features: fewer than 25 rows — returning empty.")
            return pd.DataFrame(columns=FEATURE_COLS)

        # Work on a copy; never modify the caller's DataFrame.
        d = df[["open", "high", "low", "close", "volume"]].copy().astype(float)

        # ── Feature 1: log_return ─────────────────────────────────────────────
        # log(close_t / close_{t-1}) is preferred over simple returns because
        # it is additive over time and symmetric around zero.
        d["log_return"] = np.log(d["close"] / d["close"].shift(1))

        # ── Feature 2: realised_vol_21 ────────────────────────────────────────
        # Annualised rolling standard deviation of log-returns over 21 trading
        # days (≈ 1 calendar month). Multiplied by √252 to annualise.
        # This is the primary volatility signal.
        d["realised_vol_21"] = (
            d["log_return"].rolling(21).std() * np.sqrt(252)
        )

        # ── Feature 3: vol_of_vol ─────────────────────────────────────────────
        # The standard deviation of realised_vol_21 over a 21-day window.
        # "Vol of vol" is one of the most reliable early-warning indicators of
        # regime transitions — it spikes before realised vol catches up.
        d["vol_of_vol"] = d["realised_vol_21"].rolling(21).std()

        # ── Feature 4: hl_range ───────────────────────────────────────────────
        # (High − Low) / Close normalises the intraday price range by the
        # closing level. This is a fast stress indicator — it reacts to
        # intraday panic before close-to-close vol does.
        d["hl_range"] = (d["high"] - d["low"]) / d["close"]

        # ── Feature 5: overnight_gap ──────────────────────────────────────────
        # Open / prev_close − 1 captures the gap between the previous session's
        # close and today's open. Large negative gaps signal after-hours stress
        # (earnings misses, macro shocks) that close-to-close returns miss.
        d["overnight_gap"] = (d["open"] / d["close"].shift(1)) - 1.0

        # ── Feature 6: momentum_20 ───────────────────────────────────────────
        # Close / close[−20] − 1 measures 20-day price momentum.
        # Helps the HMM distinguish crash regimes (negative momentum + high vol)
        # from low-vol sideways markets (low momentum + low vol).
        d["momentum_20"] = (d["close"] / d["close"].shift(20)) - 1.0

        # ── Feature 7: volume_ratio ───────────────────────────────────────────
        # Volume / 20-day mean volume. A ratio > 1 on a down day is a classic
        # signal of institutional selling; combined with the other features it
        # helps separate genuine stress from normal drawdowns.
        vol_avg = d["volume"].rolling(20).mean()
        d["volume_ratio"] = d["volume"] / vol_avg

        # Drop rows with any NaN (mainly the first ~21 rows due to rolling windows).
        d = d.dropna(subset=FEATURE_COLS)

        return d[FEATURE_COLS]

    # ─────────────────────────────────────────────────────────────────────────
    # Scaling helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _fit_scaler(self, features: np.ndarray) -> None:
        """
        Fit the StandardScaler on training features.

        Stores mean and std per feature as plain numpy arrays so they can be
        serialised to JSON without depending on sklearn's object format.

        Scaling is necessary because the HMM's EM algorithm initialises
        Gaussian parameters from the data range. Unscaled features with very
        different magnitudes (e.g. log_return ≈ 0.001, volume_ratio ≈ 1.0)
        cause the EM to converge slowly or to a poor local optimum.
        """
        self.scaler_mean_ = np.mean(features, axis=0)
        self.scaler_scale_ = np.std(features, axis=0, ddof=1)
        # If any feature has zero variance (constant column), set scale to 1
        # to avoid division by zero. This should not happen with real OHLCV data
        # but is a defensive guard.
        self.scaler_scale_[self.scaler_scale_ < 1e-10] = 1.0

    def _scale(self, features: np.ndarray) -> np.ndarray:
        """
        Apply fitted scaler: (x − mean) / std.

        Parameters
        ----------
        features : np.ndarray, shape (n_obs, N_FEATURES) or (N_FEATURES,)

        Returns
        -------
        np.ndarray, same shape as input, zero-mean/unit-variance per feature.

        Raises
        ------
        RuntimeError if the scaler has not been fitted yet.
        """
        if self.scaler_mean_ is None or self.scaler_scale_ is None:
            raise RuntimeError("Scaler not fitted. Call fit() or load_model() first.")
        return (features - self.scaler_mean_) / self.scaler_scale_

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def fit(
        self, df: pd.DataFrame, market_ticker: str = "UNKNOWN"
    ) -> "MarketRegimeClassifier":
        """
        Train the HMM on market index OHLCV data.

        Steps performed
        ---------------
        1.  Compute the 7-dimensional feature vector from raw OHLCV.
        2.  Fit the StandardScaler on training features (never on live data).
        3.  Run BIC model selection: for each candidate n_states ∈ {3,4,5,6,7},
            run n_init independent EM restarts with fresh random seeds and keep
            the restart with the lowest BIC. Then select the candidate with the
            globally lowest BIC.
        4.  Store the winning model's weights as plain numpy arrays.
        5.  Assign regime labels by sorting states on std(raw log-returns) of
            training bars assigned to each state via Viterbi. Using Viterbi here
            (training time only) is acceptable because we need a deterministic
            assignment of historical bars to states to compute state statistics.
        6.  Reset live-inference state so a newly trained model starts fresh.

        Parameters
        ----------
        df            : OHLCV DataFrame for the market index (SPY etc.).
                        Needs at least MIN_TRAIN_BARS rows after feature
                        computation (approx. MIN_TRAIN_BARS + 21 raw rows).
        market_ticker : Ticker symbol, stored in metadata only.

        Returns
        -------
        self — enables chaining: clf.fit(spy_df).save_model("m.json")
        """
        logger.info("Fitting MarketRegimeClassifier on %s", market_ticker)

        # ── Step 1: feature computation ───────────────────────────────────────
        features_df = self.compute_features(df)

        if len(features_df) < MIN_TRAIN_BARS:
            raise ValueError(
                f"Only {len(features_df)} feature rows available after dropping NaNs; "
                f"need at least {MIN_TRAIN_BARS} (≈ 2 years of daily bars). "
                f"Provide more historical data."
            )

        # Raw (unscaled) feature array — kept for label assignment in step 5.
        raw_features = features_df.values.astype(np.float64)

        # Raw log-returns (column 0) used for computing state volatility.
        # We preserve these BEFORE scaling so the std has a real economic meaning.
        raw_log_returns = raw_features[:, 0]

        # ── Step 2: fit scaler ────────────────────────────────────────────────
        self._fit_scaler(raw_features)
        scaled_features = self._scale(raw_features)

        # ── Step 3: BIC model selection ───────────────────────────────────────
        best_weights, best_n_states, best_bic = self._select_best_model(scaled_features)

        # ── Step 4: store model weights ───────────────────────────────────────
        self.n_states = best_n_states
        self.startprob_ = best_weights["startprob"]
        self.transmat_   = best_weights["transmat"]
        self.means_      = best_weights["means"]
        self.covars_     = best_weights["covars"]
        self.bic_score   = best_bic

        # ── Step 5: assign regime labels ──────────────────────────────────────
        self._assign_regime_labels(scaled_features, raw_log_returns)

        # ── Step 6: record metadata and reset inference state ─────────────────
        self.training_date = datetime.now(timezone.utc)
        self.market_ticker = market_ticker
        self.is_fitted = True
        self._reset_inference_state()

        logger.info(
            "Training complete: n_states=%d  BIC=%.2f  ticker=%s  date=%s",
            self.n_states, self.bic_score, self.market_ticker, self.training_date,
        )
        return self

    def _select_best_model(
        self, scaled_features: np.ndarray
    ) -> tuple[dict, int, float]:
        """
        Run BIC model selection over candidate state counts.

        For each candidate n_states:
          - Run n_init EM restarts, each with a fresh random seed.
          - Keep the restart achieving the lowest BIC.

        Then return the candidate with the globally lowest BIC.

        BIC = −2 × total_log_likelihood + k × log(n_samples)
        where k = number of free parameters (computed by _count_params).

        Why BIC and not AIC?
        BIC penalises model complexity more heavily than AIC, which is
        appropriate here because adding an extra HMM state has a large
        practical cost (more parameters, slower convergence, harder
        interpretation). BIC prevents us from over-fitting to noise in the
        training history.

        Parameters
        ----------
        scaled_features : np.ndarray, shape (n_obs, N_FEATURES)
            Pre-scaled training observations.

        Returns
        -------
        (best_weights_dict, best_n_states, best_bic)
        """
        from hmmlearn.hmm import GaussianHMM

        global_best_bic = np.inf
        global_best_weights: Optional[dict] = None
        global_best_n: int = 0

        for n_states in BIC_CANDIDATE_STATES:
            candidate_best_bic = np.inf
            candidate_best_weights: Optional[dict] = None

            for restart in range(self.n_init):
                # FIX: each restart MUST use a genuinely random seed.
                # Using fixed seeds like 42+restart makes every restart
                # identical — defeating the entire purpose of n_init.
                seed = np.random.randint(0, 100_000)

                try:
                    # FIX: covariance_type="full" captures correlations between
                    # features (e.g. log_return and realised_vol are correlated).
                    # "diag" assumes all features are independent, which is wrong
                    # for financial data and produces worse state separation.
                    model = GaussianHMM(
                        n_components=n_states,
                        covariance_type="full",   # ← full, not diag
                        n_iter=1000,              # ← 1000 iterations; 100 is too few
                        min_covar=1e-4,
                        random_state=seed,
                        params="stmc",            # optimise start, trans, means, covars
                        init_params="stmc",
                    )
                    model.fit(scaled_features)

                    # hmmlearn.score() is already the sequence total log likelihood.
                    total_ll = model.score(scaled_features)

                    k = self._count_params(n_states, N_FEATURES)
                    bic = -2.0 * total_ll + k * np.log(len(scaled_features))

                    logger.debug(
                        "n_states=%d  restart=%d  seed=%d  avg_ll=%.4f  BIC=%.2f",
                        n_states, restart, seed, total_ll / len(scaled_features), bic,
                    )

                    if bic < candidate_best_bic:
                        candidate_best_bic = bic
                        # FIX: deepcopy the arrays so the next restart's
                        # model.fit() does not overwrite the stored reference.
                        candidate_best_weights = {
                            "startprob": deepcopy(model.startprob_),
                            "transmat":  deepcopy(model.transmat_),
                            "means":     deepcopy(model.means_),
                            "covars":    self._regularize_covariances(model.covars_),
                        }

                except Exception as exc:
                    # A single restart failing is not fatal — log and continue.
                    logger.debug(
                        "Restart failed: n_states=%d restart=%d: %s",
                        n_states, restart, exc,
                    )

            if candidate_best_weights is not None:
                logger.info(
                    "n_states=%d  best BIC=%.2f", n_states, candidate_best_bic
                )
                if candidate_best_bic < global_best_bic:
                    global_best_bic = candidate_best_bic
                    global_best_weights = candidate_best_weights
                    global_best_n = n_states

        if global_best_weights is None:
            raise RuntimeError(
                "No HMM model could be trained successfully across all candidates. "
                "Check that the input data is valid and contains enough observations."
            )

        logger.info(
            "Selected n_states=%d  global BIC=%.2f", global_best_n, global_best_bic
        )
        return global_best_weights, global_best_n, global_best_bic

    @staticmethod
    def _regularize_covariances(covariances: np.ndarray) -> np.ndarray:
        """Make full covariance matrices symmetric positive-definite.

        Sparse HMM states can be numerically indefinite after EM.  Flooring
        eigenvalues retains the learned correlation structure while keeping
        Viterbi and forward-density calculations well-defined.
        """
        regularized = []
        for covariance in np.asarray(covariances, dtype=float):
            symmetric = (covariance + covariance.T) / 2.0
            values, vectors = np.linalg.eigh(symmetric)
            values = np.maximum(values, 1e-4)
            regularized.append((vectors * values) @ vectors.T)
        return np.asarray(regularized)

    @staticmethod
    def _count_params(n_states: int, n_features: int) -> int:
        """
        Count free parameters in a full-covariance Gaussian HMM.

        This is needed for the BIC formula: BIC penalises models with more
        parameters to prevent over-fitting.

        Breakdown
        ---------
        startprob : n_states − 1
            (probabilities sum to 1, so last is determined by the rest)
        transmat  : n_states × (n_states − 1)
            (each row sums to 1)
        means     : n_states × n_features
        covars    : n_states × n_features × (n_features + 1) / 2
            (full symmetric matrix — only upper triangle is free)

        Note: this corrects the original code which counted diagonal covariance
        parameters (n_states × n_features) but used covariance_type="diag".
        With covariance_type="full" the covariance parameter count is larger.
        """
        startprob = n_states - 1
        transmat  = n_states * (n_states - 1)
        means     = n_states * n_features
        covars    = n_states * n_features * (n_features + 1) // 2  # upper triangle
        return startprob + transmat + means + covars

    def _assign_regime_labels(
        self, scaled_features: np.ndarray, raw_log_returns: np.ndarray
    ) -> None:
        """
        Map each HMM state to a volatility label (CALM / MODERATE / TURBULENT)
        by ranking states on the actual standard deviation of raw log-returns
        of the training bars assigned to each state.

        Why raw log-returns, not scaled features?
        Scaled features have mean ≈ 0 and std ≈ 1 by construction, so their
        absolute magnitude is meaningless — sorting on them produces arbitrary
        orderings that change between retraining runs depending on which local
        optimum the EM converged to.

        Raw log-returns have a real economic scale (e.g. daily vol of 0.7% vs
        2.1%) that is stable and interpretable. A state with std(log_returns)
        of 2.1% is genuinely more turbulent than one with 0.7%, regardless of
        how the EM initialised.

        Why use Viterbi here (training time only)?
        We need a hard assignment of training bars to states to compute per-state
        statistics. Viterbi gives the single most-probable state sequence over
        the full training history. This look-ahead is acceptable at training time
        because it is only used to determine a human-readable label — it does NOT
        affect the forward algorithm used at inference time.

        Parameters
        ----------
        scaled_features : np.ndarray, shape (n_obs, N_FEATURES)
            Training features after scaling (used for Viterbi decoding).
        raw_log_returns : np.ndarray, shape (n_obs,)
            Unscaled log-returns for each training bar (column 0 of raw features).
        """
        from hmmlearn.hmm import GaussianHMM

        # Re-create a temporary GaussianHMM from stored weights just to call
        # predict() (Viterbi). We do not store this object — only the weights.
        temp_model = GaussianHMM(
            n_components=self.n_states, covariance_type="full"
        )
        temp_model.startprob_ = self.startprob_
        temp_model.transmat_  = self.transmat_
        temp_model.means_     = self.means_
        temp_model.covars_    = self.covars_

        # Viterbi decoding: returns the most likely state for each training bar.
        state_sequence = temp_model.predict(scaled_features)

        # Compute std(raw log-returns) per state using only bars assigned to
        # that state. This gives us the "true" volatility of each regime.
        state_vol: dict[int, float] = {}
        state_ret: dict[int, float] = {}

        for s in range(self.n_states):
            mask = state_sequence == s  # boolean mask for bars in state s
            count = mask.sum()
            if count > 1:
                # Use ddof=1 (sample std) for unbiased estimation.
                state_vol[s] = float(np.std(raw_log_returns[mask], ddof=1))
                state_ret[s] = float(np.mean(raw_log_returns[mask]))
            else:
                # Pathological edge case: a state has zero or one training bar.
                # Assign vol=0 so it sorts first (treated as the calmest state).
                state_vol[s] = 0.0
                state_ret[s] = 0.0
                logger.warning(
                    "State %d has only %d training bar(s) — label may be unreliable.",
                    s, count,
                )

        # Sort states by their volatility, ascending (calmest first).
        sorted_by_vol = sorted(state_vol.items(), key=lambda kv: kv[1])

        n = self.n_states
        self.state_order = [state_id for state_id, _ in sorted_by_vol]
        self.regime_info = {}

        for vol_rank, (state_id, vol) in enumerate(sorted_by_vol):
            # Map volatility rank to a coarse bucket based on tertiles.
            # Bottom third → CALM, middle third → MODERATE, top third → TURBULENT.
            if vol_rank < n // 3:
                bucket = "CALM"
            elif vol_rank < 2 * n // 3:
                bucket = "MODERATE"
            else:
                bucket = "TURBULENT"

            self.regime_info[state_id] = RegimeInfo(
                regime_id=state_id,
                regime_name=bucket,
                volatility_rank=vol_rank,
                # These are computed from REAL training data, not hardcoded.
                expected_return=state_ret[state_id],
                expected_volatility=vol,
            )

            logger.debug(
                "State %d → %s  vol_rank=%d  exp_vol=%.5f  exp_ret=%.6f  bars=%d",
                state_id, bucket, vol_rank, vol, state_ret[state_id],
                int(mask.sum()),  # mask is still in scope from last iteration
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Forward algorithm — pure helper (no side effects)
    # ─────────────────────────────────────────────────────────────────────────

    def _emission_probs(self, obs_scaled: np.ndarray) -> np.ndarray:
        """
        Compute P(obs | state_s) for every state s.

        Uses scipy.stats.multivariate_normal which internally uses a Cholesky
        decomposition — numerically stable even for near-singular covariance
        matrices, and faster than the manual inv(cov)/det(cov) approach.

        Parameters
        ----------
        obs_scaled : np.ndarray, shape (N_FEATURES,)
            A single scaled observation vector.

        Returns
        -------
        probs : np.ndarray, shape (n_states,)
            Un-normalised emission probabilities, clipped to 1e-300 so that
            downstream log operations never evaluate log(0).
        """
        probs = np.array([
            multivariate_normal.pdf(
                obs_scaled,
                mean=self.means_[s],
                cov=self.covars_[s],
                allow_singular=True,  # defensive guard for near-singular matrices
            )
            for s in range(self.n_states)
        ])
        # Clip to a small positive value to avoid numerical underflow.
        return np.clip(probs, 1e-300, None)

    def _forward_step(
        self, alpha: np.ndarray, obs_scaled: np.ndarray
    ) -> np.ndarray:
        """
        Advance the forward variable by one observation.

        The forward algorithm update rule is:
            α_{t+1}(j) = [ Σ_i α_t(i) × a_{ij} ] × b_j(obs_{t+1})

        where:
            α_t(i)   = P(state_t = i | obs_1..obs_t)   — current alpha
            a_{ij}   = transmat_[i, j]                 — transition probability
            b_j(obs) = P(obs | state = j)              — emission probability

        The result is then normalised to sum to 1 (scaling trick to prevent
        floating-point underflow on long sequences).

        This implementation is fully vectorised — no Python for-loop over
        states, which was a performance bottleneck in the original code.

        Parameters
        ----------
        alpha      : np.ndarray, shape (n_states,)
            Current posterior distribution P(state | obs_1..obs_t).
        obs_scaled : np.ndarray, shape (N_FEATURES,)
            Scaled observation for the new bar.

        Returns
        -------
        alpha_new : np.ndarray, shape (n_states,)
            Updated posterior after observing obs_{t+1}.
        """
        # Predict: propagate alpha through the transition matrix.
        # alpha_pred[j] = Σ_i alpha[i] × transmat_[i, j]
        # Vectorised as a dot product: alpha @ transmat_  (shape: n_states,)
        alpha_pred = alpha @ self.transmat_

        # Update: weight by emission probability.
        alpha_new = alpha_pred * self._emission_probs(obs_scaled)

        # Normalise to prevent underflow on long sequences.
        total = alpha_new.sum()
        if total > 0:
            alpha_new /= total
        else:
            # All emissions collapsed to zero — fall back to uniform.
            # This can happen with extreme observations far from all state means.
            logger.warning(
                "Forward alpha collapsed to zero — resetting to uniform. "
                "Consider retraining if this happens frequently."
            )
            alpha_new = np.ones(self.n_states) / self.n_states

        return alpha_new

    def _init_alpha(self, obs_scaled: np.ndarray) -> np.ndarray:
        """
        Initialise the forward variable for the very first observation.

        α_0(j) = startprob_[j] × b_j(obs_0)

        Parameters
        ----------
        obs_scaled : np.ndarray, shape (N_FEATURES,)

        Returns
        -------
        alpha : np.ndarray, shape (n_states,), normalised.
        """
        alpha = self.startprob_ * self._emission_probs(obs_scaled)
        total = alpha.sum()
        return alpha / total if total > 0 else np.ones(self.n_states) / self.n_states

    # ─────────────────────────────────────────────────────────────────────────
    # Primary inference entry point
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, bar: pd.Series | dict) -> MarketRegimeState:
        """
        Ingest ONE new market bar and return the updated regime state.

        THIS IS THE ONLY METHOD THAT MUTATES INTERNAL STATE.
        Call it exactly once per new bar (e.g. once per trading day after
        the market close). All other methods are read-only.

        IMPORTANT: This method needs a bar with OHLCV data AND enough recent
        history to compute rolling features. In practice, you should call
        step() by replaying a rolling window of the last ~25 bars through
        compute_features() and passing the most recent row. See the usage
        pattern in the module docstring.

        Parameters
        ----------
        bar : pd.Series or dict
            Must contain keys/index: open, high, low, close, volume, timestamp.
            The timestamp is used for logging and stored in MarketRegimeState.

        Returns
        -------
        MarketRegimeState
            Complete snapshot of the current macro regime, including the
            base_multiplier to pass to Layer 2 (StockVolatilityAdjuster).

        Raises
        ------
        RuntimeError if called before fit() or load_model().
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model not fitted. Call fit() or load_model() first."
            )

        # ── Extract timestamp ─────────────────────────────────────────────────
        if isinstance(bar, dict):
            timestamp = bar.get("timestamp", datetime.now(timezone.utc))
            bar_series = pd.Series(bar)
        else:
            timestamp = getattr(bar, "name", datetime.now(timezone.utc))
            bar_series = bar

        # ── Compute scaled observation from the bar's pre-computed features ───
        # The caller is expected to pass a bar that already has the feature
        # columns (log_return, realised_vol_21, etc.) — typically obtained by
        # calling compute_features() on a rolling window and passing iloc[-1].
        #
        # If the bar only has raw OHLCV columns, we cannot compute rolling
        # features from a single row; the caller must manage the rolling window.
        # We check which mode we're in by looking at the columns present.
        feature_keys = [c for c in FEATURE_COLS if c in bar_series.index]

        if len(feature_keys) == N_FEATURES:
            # Bar already has all feature columns — use them directly.
            obs_raw = bar_series[FEATURE_COLS].values.astype(np.float64)
        else:
            # FIX: The original code used a hardcoded placeholder:
            #   obs = np.array([0.0, 0.0, 0.0, 1.0, 0.01])
            # This meant the model was always evaluating a fake observation and
            # never actually reading the incoming bar — a critical bug.
            # We now raise a clear error instead of silently producing wrong output.
            raise ValueError(
                "bar does not contain the required feature columns. "
                f"Expected: {FEATURE_COLS}. "
                f"Got: {list(bar_series.index)}. "
                "Pass a row from compute_features() output, not raw OHLCV."
            )

        obs_scaled = self._scale(obs_raw)

        # ── Update forward variable ───────────────────────────────────────────
        # On the very first bar, initialise alpha from startprob × emission.
        # On subsequent bars, apply one forward step.
        if self._alpha is None:
            self._alpha = self._init_alpha(obs_scaled)
        else:
            self._alpha = self._forward_step(self._alpha, obs_scaled)

        # The most likely state at time t is the one with the highest posterior.
        new_state = int(np.argmax(self._alpha))

        # ── Update consecutive-bar counter ────────────────────────────────────
        # Counts how many bars in a row the model has been in new_state.
        # Resets to 1 on every state change.
        if new_state == self._current_state:
            self._consecutive_bars += 1
        else:
            self._consecutive_bars = 1
            self._current_state = new_state

        # ── Update flicker-detection history ──────────────────────────────────
        # Append every bar's state to the rolling window.
        # Flicker = too many state changes within flicker_window bars.
        self._state_history.append(new_state)

        # ── Check confirmation ────────────────────────────────────────────────
        is_confirmed = self._consecutive_bars >= self.stability_bars
        if is_confirmed and new_state != self._last_confirmed_state:
            self._last_confirmed_state = new_state
            logger.info(
                "Regime confirmed: %s  state_id=%d  bars=%d  ticker=%s",
                self.regime_info.get(new_state, RegimeInfo(0,"?",0,0,0)).regime_name,
                new_state, self._consecutive_bars, self.market_ticker,
            )

        self._last_timestamp = timestamp

        # ── Compute base_multiplier ───────────────────────────────────────────
        bucket, base_mult = self._compute_allocation(new_state, is_confirmed)

        return MarketRegimeState(
            label=bucket,
            state_id=new_state,
            volatility_bucket=bucket,
            base_multiplier=base_mult,
            probability=float(self._alpha[new_state]),
            state_probabilities=self._alpha.tolist(),
            timestamp=timestamp,
            is_confirmed=is_confirmed,
            consecutive_bars=self._consecutive_bars,
            is_flickering=self._is_flickering(),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Allocation computation — pure helper
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_allocation(
        self, state_id: int, is_confirmed: bool
    ) -> tuple[str, float]:
        """
        Translate a HMM state_id into a volatility bucket and base_multiplier.

        Uses volatility_rank from regime_info (derived from actual training data)
        rather than string-matching on label names. This means the bucket logic
        works correctly even if labels are renamed or n_states changes between
        retraining runs.

        Uncertainty penalties
        ---------------------
        - Flickering (too many recent regime changes): × _FLICKER_PENALTY
        - Not yet confirmed (< stability_bars consecutive): × _UNCONFIRMED_PENALTY

        Returns ('UNKNOWN', 0.0) if state_id has no regime_info entry.
        A 0.0 multiplier means "do not trade" — never a silent default position.
        """
        info = self.regime_info.get(state_id)
        if info is None:
            logger.error(
                "No regime_info for state_id=%d — returning UNKNOWN/0.0.", state_id
            )
            return ("UNKNOWN", 0.0)

        bucket = info.regime_name
        multiplier = REGIME_MULTIPLIERS.get(bucket, 0.0)

        # Apply penalties for model uncertainty.
        if self._is_flickering():
            logger.warning(
                "Regime flickering detected (%d changes in last %d bars) — "
                "applying %.0f%% penalty.",
                self._flicker_rate(), self.flicker_window,
                (1 - _FLICKER_PENALTY) * 100,
            )
            multiplier *= _FLICKER_PENALTY

        if not is_confirmed:
            multiplier *= _UNCONFIRMED_PENALTY

        return (bucket, round(multiplier, 4))

    def _flicker_rate(self) -> int:
        """
        Count regime changes in the current flicker_window.

        A "change" is defined as consecutive bars with different state_ids.
        Returns 0 if fewer than 2 bars have been observed.
        """
        history = list(self._state_history)
        if len(history) < 2:
            return 0
        return sum(1 for i in range(1, len(history)) if history[i] != history[i - 1])

    def _is_flickering(self) -> bool:
        """
        True if the number of state changes in flicker_window exceeds
        flicker_threshold.

        FIX: The original code defined flickering as
            consecutive_bars < flicker_threshold
        which is backwards — a low consecutive count is caused by frequent
        changes, but consecutive_bars is not the same as the change rate.
        This method correctly counts the number of transitions in the rolling
        window and compares to the threshold.
        """
        return self._flicker_rate() > self.flicker_threshold

    # ─────────────────────────────────────────────────────────────────────────
    # Stateless query methods (pure — no side effects)
    # ─────────────────────────────────────────────────────────────────────────

    def get_current_state(self) -> Optional[MarketRegimeState]:
        """
        Return the current regime state WITHOUT advancing internal state.

        Safe to call any number of times between step() calls.
        Returns None if step() has never been called.
        """
        if self._alpha is None or self._current_state is None:
            return None

        is_confirmed = self._consecutive_bars >= self.stability_bars
        bucket, multiplier = self._compute_allocation(self._current_state, is_confirmed)

        return MarketRegimeState(
            label=bucket,
            state_id=self._current_state,
            volatility_bucket=bucket,
            base_multiplier=multiplier,
            probability=float(self._alpha[self._current_state]),
            state_probabilities=self._alpha.tolist(),
            timestamp=self._last_timestamp or datetime.now(timezone.utc),
            is_confirmed=is_confirmed,
            consecutive_bars=self._consecutive_bars,
            is_flickering=self._is_flickering(),
        )

    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """
        Return the learned transition probability matrix.

        Shape: (n_states, n_states).
        Entry [i, j] = probability of moving from state i to state j.

        Useful for risk analysis: a high off-diagonal entry in the CALM →
        TURBULENT cell means the model has learned that calm regimes can
        transition quickly to turbulent ones (e.g. in 2020 COVID markets).
        """
        return self.transmat_.copy() if self.transmat_ is not None else None

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence — JSON only (no pickle)
    # ─────────────────────────────────────────────────────────────────────────

    def save_model(self, path: str) -> None:
        """
        Save all model weights and metadata to a JSON file.

        Stores raw numpy arrays as nested lists so the file is:
        - Human-readable (can be inspected in any text editor).
        - Version-portable (loads correctly across Python/hmmlearn versions).
        - Refactor-safe (no dependency on the class structure at load time).

        The file contains everything needed to reconstruct inference:
        model weights, scaler parameters, regime metadata, and hyperparameters.

        Parameters
        ----------
        path : str
            Destination file path. Parent directories are created if they
            do not exist. Recommended extension: .json
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model. Call fit() first.")

        payload = {
            "metadata": {
                "schema_version": 1,
                "model_type": "MarketRegimeClassifier",
                "n_states": self.n_states,
                "bic_score": self.bic_score,
                "training_date": self.training_date.isoformat() if self.training_date else None,
                "market_ticker": self.market_ticker,
                "n_init": self.n_init,
                "min_confidence": self.min_confidence,
                "stability_bars": self.stability_bars,
                "flicker_window": self.flicker_window,
                "flicker_threshold": self.flicker_threshold,
                "feature_cols": FEATURE_COLS,
            },
            "model_weights": {
                # numpy arrays → plain lists for JSON compatibility.
                "startprob": self.startprob_.tolist(),
                "transmat":  self.transmat_.tolist(),
                "means":     self.means_.tolist(),
                "covars":    self.covars_.tolist(),
            },
            "scaler": {
                "mean":  self.scaler_mean_.tolist(),
                "scale": self.scaler_scale_.tolist(),
            },
            "regime_info": {
                # Keys are strings because JSON requires string keys.
                str(k): {
                    "regime_id":           v.regime_id,
                    "regime_name":         v.regime_name,
                    "volatility_rank":     v.volatility_rank,
                    "expected_return":     v.expected_return,
                    "expected_volatility": v.expected_volatility,
                }
                for k, v in self.regime_info.items()
            },
            "state_order": self.state_order,
        }

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))

        logger.info("Model saved to %s", path)

    def load_model(self, path: str) -> "MarketRegimeClassifier":
        """
        Load a model previously saved with save_model().

        Reconstructs all arrays from JSON lists and resets live-inference
        state so the loaded model starts fresh (no stale alpha/history).

        Parameters
        ----------
        path : str
            Path to the JSON file produced by save_model().

        Returns
        -------
        self — enables chaining: clf.load_model("m.json").step(bar)
        """
        payload = json.loads(Path(path).read_text())

        meta    = payload["metadata"]
        weights = payload["model_weights"]
        scaler  = payload["scaler"]

        # ── Restore hyperparameters ───────────────────────────────────────────
        self.n_states          = meta["n_states"]
        self.bic_score         = meta.get("bic_score", 0.0)
        self.market_ticker     = meta.get("market_ticker")
        self.n_init            = meta.get("n_init", self.n_init)
        self.min_confidence    = meta.get("min_confidence", self.min_confidence)
        self.stability_bars    = meta.get("stability_bars", self.stability_bars)
        self.flicker_window    = meta.get("flicker_window", self.flicker_window)
        self.flicker_threshold = meta.get("flicker_threshold", self.flicker_threshold)

        self.training_date = (
            datetime.fromisoformat(meta["training_date"])
            if meta.get("training_date") else None
        )

        # ── Restore model weights ─────────────────────────────────────────────
        self.startprob_ = np.array(weights["startprob"])
        self.transmat_  = np.array(weights["transmat"])
        self.means_     = np.array(weights["means"])
        self.covars_    = np.array(weights["covars"])

        # ── Restore scaler ────────────────────────────────────────────────────
        self.scaler_mean_  = np.array(scaler["mean"])
        self.scaler_scale_ = np.array(scaler["scale"])

        # ── Restore regime metadata ───────────────────────────────────────────
        self.regime_info = {
            int(k): RegimeInfo(
                regime_id=v["regime_id"],
                regime_name=v["regime_name"],
                volatility_rank=v["volatility_rank"],
                expected_return=v["expected_return"],
                expected_volatility=v["expected_volatility"],
            )
            for k, v in payload["regime_info"].items()
        }
        self.state_order = payload["state_order"]

        # ── Reset live inference state ────────────────────────────────────────
        self._reset_inference_state()
        self.is_fitted = True

        logger.info(
            "Model loaded from %s  (n_states=%d  ticker=%s  trained=%s)",
            path, self.n_states, self.market_ticker, self.training_date,
        )
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Internal utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_inference_state(self) -> None:
        """
        Reset all mutable inference counters to their initial values.

        Called automatically after fit() and load_model() to ensure a
        freshly trained or loaded model starts with a clean slate.
        """
        self._alpha = None
        self._current_state = None
        self._consecutive_bars = 0
        self._state_history = deque(maxlen=self.flicker_window)
        self._last_confirmed_state = None
        self._last_timestamp = None

    def __repr__(self) -> str:
        if not self.is_fitted:
            return "MarketRegimeClassifier(unfitted)"
        return (
            f"MarketRegimeClassifier("
            f"n_states={self.n_states}, "
            f"ticker={self.market_ticker!r}, "
            f"BIC={self.bic_score:.2f}, "
            f"trained={self.training_date})"
        )
