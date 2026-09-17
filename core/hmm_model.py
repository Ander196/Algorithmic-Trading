"""
HMM Model Module - Hidden Markov Model for Volatility Regime Detection
=======================================================================

WHAT THIS MODULE DOES
---------------------
Implements a Gaussian Hidden Markov Model (HMM) that classifies the current
market environment into one of N volatility regimes (e.g. CALM, MODERATE,
TURBULENT). It does NOT predict price direction.

The strategy layer reads the volatility classification and adjusts portfolio
allocation accordingly:
  - CALM      → full position size (multiplier = 1.0)
  - MODERATE  → reduced position  (multiplier = 0.75)
  - TURBULENT → minimal position  (multiplier = 0.50)

FIXES APPLIED OVER PREVIOUS VERSION
-------------------------------------
1. predict_regime_proba()  — forward loop was iterating backwards over
                             observations; fixed to run correctly forward.
2. fit() random restarts   — n_init used fixed seed offsets, making all
                             restarts identical; now uses random seeds.
3. Regime label ordering   — sorted on SCALED features (meaningless); now
                             sorts on raw returns from the original DataFrame.
4. Emission computation    — manual O(n³) inv/det replaced with
                             scipy.stats.multivariate_normal (faster, stable).
5. Mutable state safety    — predict_*() methods no longer mutate instance
                             state; a single step() method advances state
                             exactly once per bar.
6. Unknown-state exposure  — returning 0.5 multiplier for unknown state;
                             now returns 0.0 (flat) and raises a warning.
7. Strategy params source  — strategy types derived from label strings
                             (fragile); now derived from volatility quantile.
8. Pickle serialisation    — model weights saved as plain numpy arrays +
                             JSON metadata instead of a monolithic pickle blob.

USAGE
-----
    from data.indicators import prepareFeaturesForHMM

    # ── Training ────────────────────────────────────────────────────────────
    clf = HMMVolatilityClassifier()
    clf.fit(df_train)                  # df must have OHLCV columns
    clf.save_model("hmm_weights.json")

    # ── Live inference ───────────────────────────────────────────────────────
    clf2 = HMMVolatilityClassifier()
    clf2.load_model("hmm_weights.json")

    # Call step() exactly ONCE per new bar — it returns a RegimeState and
    # updates all internal counters atomically.
    state = clf2.step(new_bar_features)   # new_bar_features: 1-row DataFrame
    vol_level, multiplier = clf2.get_regime_for_allocation()
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

# ── Optional Supabase storage ────────────────────────────────────────────────
try:
    from storage import storeHMMResult
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Compatibility export.  The former two-layer orchestrator was referenced by
# package callers but never shipped; the supported runner lives in train_only.


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeInfo:
    """
    Static metadata for a single HMM state, assigned once after training.

    Fields
    ------
    regime_id           : HMM state index (0-based, as returned by hmmlearn).
    regime_name         : Human-readable label (e.g. 'BEAR', 'NEUTRAL', 'BULL').
    volatility_rank     : 0 = lowest-volatility state, n_regimes-1 = highest.
    expected_return     : Mean raw return of training bars assigned to this state.
    expected_volatility : Std-dev of raw returns for this state.
    """
    regime_id: int
    regime_name: str
    volatility_rank: int
    expected_return: float
    expected_volatility: float


@dataclass
class RegimeState:
    """
    Snapshot of regime detection at a single point in time.

    Returned by step() and get_current_regime_state().

    Fields
    ------
    label              : Human-readable name of the current state.
    state_id           : HMM state index.
    probability        : P(current_state | all observations so far).
    state_probabilities: Full distribution over all N states.
    timestamp          : Wall-clock time of this observation.
    is_confirmed       : True if the same state has persisted for
                         >= confirmation_bars consecutive bars.
    consecutive_bars   : Number of bars in a row with this state.
    """
    label: str
    state_id: int
    probability: float
    state_probabilities: np.ndarray
    timestamp: Optional[datetime] = None
    is_confirmed: bool = False
    consecutive_bars: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class HMMVolatilityClassifier:
    """
    Gaussian HMM for market volatility regime classification.

    Key design decisions
    --------------------
    * BIC-based automatic model selection tests n_components ∈ {3,4,5,6,7}.
    * Forward algorithm (filtering) is used for inference — never the Viterbi
      smoother — so no future data leaks into current predictions.
    * All state mutation happens inside step(), which must be called exactly
      once per new bar. Probability query methods are pure (no side effects).
    * Regime labels are sorted by VOLATILITY of raw returns in training data,
      not by scaled-feature means, so labels are stable across retraining.
    """

    # Ordered label sets for each possible n_regimes value.
    # Labels run from lowest-volatility (index 0) to highest-volatility (last).
    REGIME_LABELS: dict[int, list[str]] = {
        3: ['LOW_VOL',    'MID_VOL',     'HIGH_VOL'],
        4: ['LOW_VOL',    'MID_LOW_VOL', 'MID_HIGH_VOL', 'HIGH_VOL'],
        5: ['LOW_VOL',    'MID_LOW_VOL', 'MID_VOL',      'MID_HIGH_VOL', 'HIGH_VOL'],
        6: ['VOL_1',      'VOL_2',       'VOL_3',        'VOL_4',        'VOL_5', 'VOL_6'],
        7: ['VOL_1',      'VOL_2',       'VOL_3',        'VOL_4',        'VOL_5', 'VOL_6', 'VOL_7'],
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
        Parameters
        ----------
        min_periods         : Minimum trading days required before training.
                              504 ≈ 2 calendar years of daily bars.
        n_init              : Independent random restarts per candidate model.
                              Each restart uses a different random seed so the
                              EM optimiser explores different local optima.
        confidence_threshold: P(state) must exceed this to act on a signal.
        confirmation_bars   : Consecutive bars required to confirm a new regime.
        flicker_threshold   : Max regime changes within flicker_window before
                              the model enters "uncertainty" mode.
        flicker_window      : Rolling window size for flicker detection.
        """
        # ── Hyperparameters (set at construction, never changed) ─────────────
        self.min_periods = min_periods
        self.n_init = n_init
        self.confidence_threshold = confidence_threshold
        self.confirmation_bars = confirmation_bars
        self.flicker_threshold = flicker_threshold
        self.flicker_window = flicker_window

        # ── Model artefacts (populated by fit() or load_model()) ─────────────
        self.model: Optional[GaussianHMM] = None
        self.scaler: Optional[StandardScaler] = None
        self.n_regimes: int = 0
        self.bic_score: float = 0.0
        self.training_date: Optional[datetime] = None
        self.regime_labels: list[str] = []

        # Maps HMM state_id → RegimeInfo (assigned after training).
        self.regime_info: dict[int, RegimeInfo] = {}

        # Maps HMM state_id → human-readable label string.
        self._state_to_label: dict[int, str] = {}

        # ── Live-inference state (advanced by step()) ────────────────────────
        # Current alpha vector P(state | obs_1..t) — shape (n_regimes,).
        self._alpha: Optional[np.ndarray] = None
        # Most recent confirmed state_id.
        self._current_state: Optional[int] = None
        # Rolling history of state_ids, length ≤ flicker_window.
        self._state_history: list[int] = []
        # How many consecutive bars have shown _current_state.
        self._consecutive_bars: int = 0
        # Last state_id that was "confirmed" (persisted ≥ confirmation_bars).
        self._last_confirmed_state: Optional[int] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "HMMVolatilityClassifier":
        """
        Train the HMM on historical OHLCV data.

        Steps
        -----
        1. Extract features from the raw OHLCV DataFrame via
           prepareFeaturesForHMM().  This function (defined in
           data/indicators.py) is expected to return a numeric DataFrame
           with columns such as log-returns, ATR-normalised volatility,
           RSI, etc.
        2. Scale features to zero-mean / unit-variance using StandardScaler
           fitted on the training data only.
        3. Test GaussianHMM for n_components ∈ {3,4,5,6,7}.  For each
           candidate, run n_init independent restarts with different random
           seeds to escape local optima.  Select the restart with the
           lowest BIC within each candidate.
        4. Pick the candidate n_components with the globally lowest BIC
           (Bayesian Information Criterion balances fit vs. model complexity).
        5. Assign human-readable regime labels sorted by the VOLATILITY of
           raw returns in training data (not on scaled features, which have
           no meaningful absolute magnitude).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at minimum columns: open, high, low, close, volume.

        Returns
        -------
        self  (enables method chaining: clf.fit(df).save_model("w.json"))
        """
        # ── Step 1: feature extraction ───────────────────────────────────────
        # prepareFeaturesForHMM is imported from data.indicators.
        # It returns a DataFrame of numeric features aligned to df's index,
        # potentially shorter than df due to lookback windows.
        from data.indicators import prepareFeaturesForHMM
        features_df = prepareFeaturesForHMM(df, self.min_periods)

        if len(features_df) < self.min_periods:
            raise ValueError(
                f"Only {len(features_df)} rows after feature preparation; "
                f"need at least {self.min_periods}."
            )

        # ── Step 2: scaling ──────────────────────────────────────────────────
        # StandardScaler transforms each feature to mean=0, std=1.
        # We fit the scaler HERE (on training data only) and reuse it at
        # inference time — never refit on test/live data.
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features_df.values)

        # Keep raw returns for label assignment (see step 5).
        # Convention: first column of features_df must be the log-return.
        raw_returns = features_df.iloc[:, 0].values

        # ── Step 3: model selection loop ─────────────────────────────────────
        candidate_components = [3, 4, 5, 6, 7]
        selection_results: list[dict] = []

        logger.info("Starting HMM model selection over %s", candidate_components)

        for n_comp in candidate_components:
            best_bic = np.inf
            best_model: Optional[GaussianHMM] = None
            best_ll: Optional[float] = None

            for init_idx in range(self.n_init):
                # FIX: use a genuinely random seed each restart (not 42+init).
                # Previously all restarts produced identical models because
                # 42+0, 42+1, … are deterministic.  Now each restart explores
                # a different starting point in parameter space.
                rng_seed = np.random.randint(0, 100_000)

                model = GaussianHMM(
                    n_components=n_comp,
                    covariance_type="full",
                    n_iter=1000,
                    random_state=rng_seed,
                )

                try:
                    model.fit(features_scaled)

                    # hmmlearn.score() returns AVERAGE log-likelihood per
                    # sample.  Total LL = score * n_samples.
                    avg_ll = model.score(features_scaled)
                    total_ll = avg_ll * len(features_scaled)

                    n_params = self._count_parameters(n_comp, features_scaled.shape[1])
                    bic = -2 * total_ll + n_params * np.log(len(features_scaled))

                    logger.debug(
                        "n_comp=%d init=%d seed=%d  avg_ll=%.3f  BIC=%.2f",
                        n_comp, init_idx + 1, rng_seed, avg_ll, bic,
                    )

                    if bic < best_bic:
                        best_bic = bic
                        best_model = deepcopy(model)   # isolate from next iter
                        best_ll = avg_ll

                except Exception as exc:
                    logger.warning(
                        "Training failed for n_comp=%d init=%d: %s",
                        n_comp, init_idx + 1, exc,
                    )

            if best_model is not None:
                selection_results.append({
                    "n_components": n_comp,
                    "bic": best_bic,
                    "log_likelihood": best_ll,
                    "model": best_model,
                })
                logger.info("n_comp=%d  best BIC=%.2f  best_ll=%.3f", n_comp, best_bic, best_ll)

        if not selection_results:
            raise RuntimeError("No HMM model could be trained successfully.")

        # ── Step 4: select best model by BIC ─────────────────────────────────
        selected = min(selection_results, key=lambda r: r["bic"])
        self.n_regimes = selected["n_components"]
        self.model = selected["model"]
        self.bic_score = selected["bic"]
        logger.info("Selected n_regimes=%d  BIC=%.2f", self.n_regimes, self.bic_score)

        # ── Step 5: assign regime labels sorted by raw-return volatility ──────
        # We decode the training data once (using Viterbi — only during training
        # is this acceptable since we have the full history) to find which
        # training bars belong to each state, then compute the volatility of
        # raw returns for each state.
        self._assign_regime_labels(features_scaled, raw_returns)
        self.training_date = datetime.now()

        # Reset live-inference state so a retrained classifier starts fresh.
        self._reset_live_state()

        return self

    def _count_parameters(self, n_components: int, n_features: int) -> int:
        """
        Count the number of free parameters in a full-covariance GaussianHMM.

        Breakdown
        ---------
        startprob : n_states - 1   (sums to 1, so one is determined)
        transmat  : n_states × (n_states - 1)   (each row sums to 1)
        means     : n_states × n_features
        covars    : n_states × n_features × (n_features + 1) / 2
                    (symmetric matrix, upper triangle only)
        """
        startprob = n_components - 1
        transmat = n_components * (n_components - 1)
        means = n_components * n_features
        covars = n_components * n_features * (n_features + 1) // 2
        return startprob + transmat + means + covars

    def _assign_regime_labels(
        self, features_scaled: np.ndarray, raw_returns: np.ndarray
    ) -> None:
        """
        Map HMM state indices to human-readable labels sorted by VOLATILITY.

        Why sort by volatility, not by return?
        This is a VOLATILITY classifier.  The labels CALM / TURBULENT refer
        to how much the market is moving, not which direction.  Sorting by
        return would mix high-return / low-vol states with low-return /
        high-vol states and give inconsistent labels across retraining runs.

        Steps
        -----
        1. Run Viterbi on the training data to get the most-likely state
           sequence (using full history is fine here — training-time only).
        2. For each state, compute the std-dev of raw log-returns of all
           bars assigned to that state.
        3. Sort states by that std-dev (ascending = calmer first).
        4. Map sorted position → label from REGIME_LABELS[n_regimes].
        5. Record volatility_rank in RegimeInfo for downstream use.
        """
        hidden_states = self.model.predict(features_scaled)

        state_volatility: dict[int, float] = {}
        state_mean_return: dict[int, float] = {}

        for state in range(self.n_regimes):
            mask = hidden_states == state
            if mask.sum() > 0:
                state_volatility[state] = float(np.std(raw_returns[mask]))
                state_mean_return[state] = float(np.mean(raw_returns[mask]))
            else:
                # Edge case: a state was never visited in training data.
                state_volatility[state] = 0.0
                state_mean_return[state] = 0.0
                logger.warning("State %d had no training observations.", state)

        # Sort states by volatility ascending: index 0 is the calmest.
        sorted_by_vol = sorted(state_volatility.items(), key=lambda kv: kv[1])

        label_list = self.REGIME_LABELS[self.n_regimes]
        self.regime_labels = label_list

        self.regime_info = {}
        self._state_to_label = {}

        for vol_rank, (state_id, vol) in enumerate(sorted_by_vol):
            label = label_list[vol_rank]
            self.regime_info[state_id] = RegimeInfo(
                regime_id=state_id,
                regime_name=label,
                volatility_rank=vol_rank,
                expected_return=state_mean_return[state_id],
                expected_volatility=vol,
            )
            self._state_to_label[state_id] = label
            logger.debug(
                "State %d → %s  vol_rank=%d  exp_vol=%.5f  exp_ret=%.5f",
                state_id, label, vol_rank, vol, state_mean_return[state_id],
            )

    def _reset_live_state(self) -> None:
        """Reset all mutable inference state to defaults (call after fit/load)."""
        self._alpha = None
        self._current_state = None
        self._state_history = []
        self._consecutive_bars = 0
        self._last_confirmed_state = None

    # ─────────────────────────────────────────────────────────────────────────
    # Pure helper: emission probabilities
    # ─────────────────────────────────────────────────────────────────────────

    def _emission_probs(self, obs: np.ndarray) -> np.ndarray:
        """
        Return P(obs | state) for every state — shape (n_regimes,).

        Uses scipy.stats.multivariate_normal for each state's Gaussian
        distribution.  This replaces the previous manual implementation that
        called np.linalg.inv() and np.linalg.det() per state per timestep,
        which was both slow (O(d³) per state) and numerically fragile (det
        underflows to 0 for high-dimensional features).

        Parameters
        ----------
        obs : np.ndarray, shape (n_features,)
            Single SCALED observation vector.

        Returns
        -------
        probs : np.ndarray, shape (n_regimes,)
            Raw (un-normalised) emission probabilities.  Clipped to 1e-300
            so that downstream log operations never hit -inf.
        """
        probs = np.array([
            multivariate_normal.pdf(
                obs,
                mean=self.model.means_[s],
                cov=self.model.covars_[s],
                allow_singular=True,   # avoids crashes on near-singular matrices
            )
            for s in range(self.n_regimes)
        ])
        return np.clip(probs, 1e-300, None)

    # ─────────────────────────────────────────────────────────────────────────
    # Pure helper: forward algorithm (no side effects)
    # ─────────────────────────────────────────────────────────────────────────

    def _forward_pass(
        self,
        features_scaled: np.ndarray,
        initial_alpha: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run the HMM forward (filtering) algorithm and return the final alpha.

        The forward algorithm computes, for each time step t:
            α_t(j) = P(state_t = j | obs_1 .. obs_t)

        This uses ONLY past and present observations — no future data leaks
        in — making it safe for live inference and backtesting.

        The Viterbi algorithm (used by hmmlearn.predict() internally) uses
        the FULL sequence and therefore introduces look-ahead bias; we
        deliberately avoid it here.

        Parameters
        ----------
        features_scaled : np.ndarray, shape (T, n_features)
            Pre-scaled observations from the oldest to the most recent.
        initial_alpha   : np.ndarray or None, shape (n_regimes,)
            If provided, start the recursion from this alpha vector instead
            of the model's startprob.  Used by step() to continue from the
            previous bar's alpha rather than reprocessing the whole history.

        Returns
        -------
        alpha : np.ndarray, shape (n_regimes,)
            Normalised posterior distribution over states given all
            observations in features_scaled (plus the prior if initial_alpha
            was provided).
        """
        transmat = self.model.transmat_    # shape (n_regimes, n_regimes)

        if initial_alpha is not None:
            # Continue from a previous alpha: compute predicted alpha for
            # the new observation, then update with emission.
            alpha = initial_alpha @ transmat
            alpha = alpha * self._emission_probs(features_scaled[0])
            start_idx = 1
        else:
            # Cold start: initialise with start probabilities × first emission.
            alpha = self.model.startprob_ * self._emission_probs(features_scaled[0])
            start_idx = 1

        # Normalise to prevent underflow on long sequences.
        alpha /= alpha.sum()

        # FIX: the original predict_regime_proba() initialised alpha on
        # features_scaled[-1] and then iterated over features_scaled[0..n-2],
        # effectively running the recursion backwards.  Corrected here to
        # always iterate forward from start_idx to T-1.
        for t in range(start_idx, len(features_scaled)):
            alpha = (alpha @ transmat) * self._emission_probs(features_scaled[t])
            # Normalise at every step (scale the probability to prevent underflow
            # in very long sequences).
            alpha_sum = alpha.sum()
            if alpha_sum == 0:
                logger.warning("Alpha collapsed to zero at t=%d; resetting to uniform.", t)
                alpha = np.ones(self.n_regimes) / self.n_regimes
            else:
                alpha /= alpha_sum

        return alpha

    # ─────────────────────────────────────────────────────────────────────────
    # Primary inference entry point
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, new_bar: pd.DataFrame | np.ndarray) -> RegimeState:
        """
        Ingest ONE new bar and advance all internal state.

        This is the ONLY method that should mutate instance state during live
        inference.  Call it exactly once per new bar.  Do NOT call any of the
        predict_*() methods in a live loop — they are stateless helpers for
        batch analysis.

        Parameters
        ----------
        new_bar : pd.DataFrame or np.ndarray, shape (1, n_features) or (n_features,)
            Features for the single new bar.  Must be in the same raw
            (unscaled) feature space that was used to train the scaler.

        Returns
        -------
        RegimeState
            Comprehensive snapshot of the current regime.

        Raises
        ------
        RuntimeError
            If called before fit() or load_model().
        """
        self._assert_trained()

        obs = self._to_scaled_array(new_bar)  # shape (1, n_features)

        # ── 1. Update alpha via forward algorithm ─────────────────────────────
        # Pass the existing _alpha as the warm-start prior.  This way we never
        # reprocess the full history — only the single new observation.
        self._alpha = self._forward_pass(obs, initial_alpha=self._alpha)
        new_state = int(np.argmax(self._alpha))

        # ── 2. Update consecutive-bar counter ────────────────────────────────
        if new_state == self._current_state:
            self._consecutive_bars += 1
        else:
            self._consecutive_bars = 1
            self._current_state = new_state

        # ── 3. Update rolling history (for flicker detection) ────────────────
        self._state_history.append(new_state)
        if len(self._state_history) > self.flicker_window:
            self._state_history.pop(0)

        # ── 4. Check confirmation ─────────────────────────────────────────────
        is_confirmed = self._consecutive_bars >= self.confirmation_bars
        if is_confirmed and new_state != self._last_confirmed_state:
            self._last_confirmed_state = new_state
            logger.info(
                "Regime confirmed: %s (state_id=%d, bars=%d)",
                self._state_to_label.get(new_state, "UNKNOWN"),
                new_state,
                self._consecutive_bars,
            )

        # ── 5. Build and return snapshot ─────────────────────────────────────
        return RegimeState(
            label=self._state_to_label.get(new_state, "UNKNOWN"),
            state_id=new_state,
            probability=float(self._alpha[new_state]),
            state_probabilities=self._alpha.copy(),
            timestamp=datetime.now(),
            is_confirmed=is_confirmed,
            consecutive_bars=self._consecutive_bars,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Stateless query methods (no side effects — safe to call any time)
    # ─────────────────────────────────────────────────────────────────────────

    def predict_proba_batch(
        self, features: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """
        Compute the posterior alpha vector after seeing a full batch of bars.

        Runs a complete forward pass from scratch on the provided sequence.
        Does NOT update _alpha or any other instance state.

        Useful for backtesting or offline analysis where you want the
        filtered probability at the END of a window without disturbing the
        live-inference state.

        Parameters
        ----------
        features : pd.DataFrame or np.ndarray, shape (T, n_features)
            Sequence of raw (unscaled) feature observations.

        Returns
        -------
        alpha : np.ndarray, shape (n_regimes,)
            Posterior probability distribution over regimes after bar T.
        """
        self._assert_trained()
        obs_scaled = self._to_scaled_array(features)
        return self._forward_pass(obs_scaled)

    def get_current_regime_state(self) -> RegimeState:
        """
        Return the current regime as a RegimeState without advancing state.

        Requires at least one prior call to step().

        Returns
        -------
        RegimeState
            Snapshot based on the last step() call.

        Raises
        ------
        RuntimeError
            If step() has never been called.
        """
        if self._alpha is None or self._current_state is None:
            raise RuntimeError(
                "No regime state available.  Call step() at least once before "
                "querying get_current_regime_state()."
            )

        state_id = self._current_state
        is_confirmed = self._consecutive_bars >= self.confirmation_bars

        return RegimeState(
            label=self._state_to_label.get(state_id, "UNKNOWN"),
            state_id=state_id,
            probability=float(self._alpha[state_id]),
            state_probabilities=self._alpha.copy(),
            timestamp=datetime.now(),
            is_confirmed=is_confirmed,
            consecutive_bars=self._consecutive_bars,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Flicker & stability metrics
    # ─────────────────────────────────────────────────────────────────────────

    def get_regime_flicker_rate(self) -> float:
        """
        Count regime changes in the last flicker_window bars.

        A high flicker rate indicates the model is oscillating between states,
        which typically signals a transitional / noisy market environment.

        Returns
        -------
        float
            Number of state changes in the rolling window.
        """
        history = self._state_history
        if len(history) < 2:
            return 0.0

        return float(
            sum(1 for i in range(1, len(history)) if history[i] != history[i - 1])
        )

    def is_flickering(self) -> bool:
        """Return True if the flicker rate exceeds flicker_threshold."""
        return self.get_regime_flicker_rate() > self.flicker_threshold

    def get_regime_stability(self) -> int:
        """Return how many consecutive bars the current state has been active."""
        return self._consecutive_bars

    def get_transition_matrix(self) -> np.ndarray:
        """
        Return the learned transition probability matrix.

        Shape (n_regimes, n_regimes).  Entry [i, j] is the probability of
        transitioning from state i to state j on the next bar.
        """
        self._assert_trained()
        return self.model.transmat_.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # Strategy-layer interface
    # ─────────────────────────────────────────────────────────────────────────

    def get_regime_for_allocation(self) -> tuple[str, float]:
        """
        Translate the current regime into a volatility bucket and a position
        size multiplier for the strategy layer.

        Bucketing is based on VOLATILITY RANK (not on label strings), so the
        result is consistent even if labels are renamed or the n_regimes
        changes between training runs.

        Volatility buckets
        ------------------
        Bottom third of regimes by vol  → 'CALM'       multiplier = 1.00
        Middle third                    → 'MODERATE'   multiplier = 0.75
        Top third                       → 'TURBULENT'  multiplier = 0.50

        Adjustments
        -----------
        * Flickering detected              → multiplier × 0.75
        * Not yet confirmed (< conf_bars)  → multiplier × 0.75
        * Unknown / unwarmed state         → 'UNKNOWN'  multiplier = 0.00
          (FIX: was 0.5 previously, which silently took a position when
          the model had no opinion)

        Returns
        -------
        (volatility_level, position_multiplier) : (str, float)
        """
        if self._current_state is None:
            # Model hasn't been warmed up — do not trade.
            logger.warning(
                "get_regime_for_allocation() called before any step(). "
                "Returning UNKNOWN with 0.0 multiplier."
            )
            return ("UNKNOWN", 0.0)

        info = self.regime_info.get(self._current_state)
        if info is None:
            logger.error("No RegimeInfo for state_id=%d.", self._current_state)
            return ("UNKNOWN", 0.0)

        vol_rank = info.volatility_rank
        n = self.n_regimes

        # Assign bucket by volatility rank (independent of label strings).
        if vol_rank < n // 3:
            level = "CALM"
            multiplier = 1.00
        elif vol_rank < 2 * n // 3:
            level = "MODERATE"
            multiplier = 0.75
        else:
            level = "TURBULENT"
            multiplier = 0.50

        # Uncertainty adjustments — each halves the multiplier by 25 %.
        if self.is_flickering():
            logger.warning("Flickering detected — reducing position multiplier.")
            multiplier *= 0.75

        if self._consecutive_bars < self.confirmation_bars:
            multiplier *= 0.75

        return (level, round(multiplier, 4))

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save_model(self, path: str | Path) -> None:
        """
        Save the trained model to a JSON file.

        Unlike pickle, JSON is human-readable, version-agnostic, and immune
        to class-refactoring breakage.  The model weights (means, covariances,
        transition matrix, start probabilities) are stored as nested lists;
        the scaler parameters are stored alongside them.

        The file can be loaded with load_model() in any future Python/hmmlearn
        version without compatibility issues.

        Parameters
        ----------
        path : str or Path
            Destination file path (typically *.json).
        """
        self._assert_trained()
        path = Path(path)

        payload = {
            "metadata": {
                "n_regimes": self.n_regimes,
                "bic_score": self.bic_score,
                "training_date": self.training_date.isoformat() if self.training_date else None,
                "regime_labels": self.regime_labels,
                "confidence_threshold": self.confidence_threshold,
                "confirmation_bars": self.confirmation_bars,
                "flicker_threshold": self.flicker_threshold,
                "flicker_window": self.flicker_window,
                "regime_info": {
                    str(k): {
                        "regime_id": v.regime_id,
                        "regime_name": v.regime_name,
                        "volatility_rank": v.volatility_rank,
                        "expected_return": v.expected_return,
                        "expected_volatility": v.expected_volatility,
                    }
                    for k, v in self.regime_info.items()
                },
                "state_to_label": {str(k): v for k, v in self._state_to_label.items()},
            },
            "model_weights": {
                "startprob": self.model.startprob_.tolist(),
                "transmat": self.model.transmat_.tolist(),
                "means": self.model.means_.tolist(),
                "covars": self.model.covars_.tolist(),
                "covariance_type": self.model.covariance_type,
            },
            "scaler": {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
                "var": self.scaler.var_.tolist(),
                "n_features_in": int(self.scaler.n_features_in_),
            },
        }

        path.write_text(json.dumps(payload, indent=2))
        logger.info("Model saved to %s", path)

    def load_model(self, path: str | Path) -> None:
        """
        Load a model previously saved with save_model().

        Reconstructs the GaussianHMM by directly setting the weight arrays
        rather than calling fit() — this is the correct way to restore a
        trained hmmlearn model without re-training.

        Parameters
        ----------
        path : str or Path
            Path to the JSON file produced by save_model().
        """
        path = Path(path)
        payload = json.loads(path.read_text())

        meta = payload["metadata"]
        weights = payload["model_weights"]
        scaler_data = payload["scaler"]

        # ── Restore metadata ─────────────────────────────────────────────────
        self.n_regimes = meta["n_regimes"]
        self.bic_score = meta["bic_score"]
        self.training_date = (
            datetime.fromisoformat(meta["training_date"])
            if meta.get("training_date")
            else None
        )
        self.regime_labels = meta["regime_labels"]
        self.confidence_threshold = meta["confidence_threshold"]
        self.confirmation_bars = meta["confirmation_bars"]
        self.flicker_threshold = meta["flicker_threshold"]
        self.flicker_window = meta["flicker_window"]

        self.regime_info = {
            int(k): RegimeInfo(**v) for k, v in meta["regime_info"].items()
        }
        self._state_to_label = {int(k): v for k, v in meta["state_to_label"].items()}

        # ── Restore GaussianHMM weights ──────────────────────────────────────
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=weights["covariance_type"],
        )
        self.model.startprob_ = np.array(weights["startprob"])
        self.model.transmat_ = np.array(weights["transmat"])
        self.model.means_ = np.array(weights["means"])
        self.model.covars_ = np.array(weights["covars"])

        # ── Restore StandardScaler ───────────────────────────────────────────
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(scaler_data["mean"])
        self.scaler.scale_ = np.array(scaler_data["scale"])
        self.scaler.var_ = np.array(scaler_data["var"])
        self.scaler.n_features_in_ = scaler_data["n_features_in"]

        # Reset live-inference state so the loaded model starts fresh.
        self._reset_live_state()

        logger.info(
            "Model loaded from %s  (n_regimes=%d, trained=%s)",
            path, self.n_regimes, self.training_date,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Optional Supabase storage
    # ─────────────────────────────────────────────────────────────────────────

    def store_to_supabase(self, ticker: str) -> dict | None:
        """
        Store the current regime state to Supabase.

        Only operates when the `storage` module is importable.  If Supabase
        is unavailable (e.g. in local dev), logs a warning and returns None
        without raising.

        Parameters
        ----------
        ticker : str
            Equity ticker symbol (e.g. 'AAPL').

        Returns
        -------
        dict or None
            The inserted Supabase record, or None if unavailable.
        """
        if not SUPABASE_AVAILABLE:
            logger.warning("Supabase not configured — skipping storage.")
            return None

        state = self.get_current_regime_state()
        vol_level, multiplier = self.get_regime_for_allocation()

        result = storeHMMResult(
            ticker=ticker,
            result_date=state.timestamp or datetime.now(),
            regime_label=state.label,
            state_id=state.state_id,
            probability=state.probability,
            state_probabilities=state.state_probabilities.tolist(),
            volatility_level=vol_level,
            position_multiplier=multiplier,
            is_confirmed=state.is_confirmed,
            is_flickering=self.is_flickering(),
            stability_bars=state.consecutive_bars,
            bic_score=self.bic_score,
            n_regimes=self.n_regimes,
            training_date=self.training_date,
        )

        logger.info("Stored HMM result for %s: %s", ticker, state.label)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Internal utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _assert_trained(self) -> None:
        """Raise RuntimeError if the model has not been trained or loaded."""
        if self.model is None or self.scaler is None:
            raise RuntimeError(
                "Model is not trained.  Call fit() or load_model() first."
            )

    def _to_scaled_array(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Convert raw features to a scaled numpy array.

        Handles DataFrames, 2-D arrays, and 1-D row vectors.

        Parameters
        ----------
        features : pd.DataFrame or np.ndarray

        Returns
        -------
        np.ndarray, shape (T, n_features), dtype float64
        """
        if isinstance(features, pd.DataFrame):
            arr = features.values.astype(np.float64)
        else:
            arr = np.asarray(features, dtype=np.float64)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        return self.scaler.transform(arr)

    def __repr__(self) -> str:
        status = (
            f"n_regimes={self.n_regimes}, BIC={self.bic_score:.2f}, "
            f"trained={self.training_date}"
            if self.model is not None
            else "untrained"
        )
        return f"HMMVolatilityClassifier({status})"


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def train_hmm(df: pd.DataFrame, **kwargs) -> HMMVolatilityClassifier:
    """
    Convenience wrapper: construct and train an HMMVolatilityClassifier.

    Parameters
    ----------
    df     : pd.DataFrame with OHLCV columns.
    kwargs : Forwarded to HMMVolatilityClassifier.__init__().

    Returns
    -------
    HMMVolatilityClassifier (trained, ready for step() calls)
    """
    return HMMVolatilityClassifier(**kwargs).fit(df)


# ---------------------------------------------------------------------------
# Train-only orchestration.  It lives here because Layer 2 is the individual
# HMM implemented above; no separate runner is needed.

_PRICE_COLUMNS = "ticker, price_date, open, high, low, close, volume"


def _fetch_prices(client: Any, ticker: str, limit: int = 3000) -> pd.DataFrame:
    rows = client.table("stock_prices").select(_PRICE_COLUMNS).eq(
        "ticker", ticker.upper()
    ).order("price_date", desc=True).limit(limit).execute().data
    frame = pd.DataFrame(rows or [])
    if frame.empty:
        raise ValueError(f"No OHLCV rows available for {ticker}")
    frame["price_date"] = pd.to_datetime(frame["price_date"], utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["open", "high", "low", "close"]).sort_values("price_date")


def latest_price_date_tickers(client: Any) -> tuple[str, list[str]]:
    """Return every ticker present on the latest available price date.

    This intentionally queries ``stock_prices`` rather than ``stocks``: the
    latter is a catalogue/load batch, whereas the former tells us which assets
    have data suitable for the current HMM run.
    """
    latest = client.table("stock_prices").select("price_date").order(
        "price_date", desc=True
    ).limit(1).execute().data
    if not latest:
        raise ValueError("stock_prices is empty")
    result_date = latest[0]["price_date"]
    rows = client.table("stock_prices").select("ticker").eq(
        "price_date", result_date
    ).execute().data
    return result_date, sorted({row["ticker"].upper() for row in rows})


def _write_result_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _market_result(classifier: Any, market_ticker: str, state: Any) -> dict[str, Any]:
    return {
        "market_ticker": market_ticker,
        "result_date": state.timestamp.date().isoformat(),
        "regime_label": state.label,
        "state_id": state.state_id,
        "probability": state.probability,
        "state_probabilities": state.state_probabilities,
        "base_multiplier": state.base_multiplier,
        "is_regime_confirmed": state.is_confirmed,
        "is_flickering": state.is_flickering,
        "stability_bars": state.consecutive_bars,
        "bic_score": classifier.bic_score,
        "n_states": classifier.n_states,
        "trained_at": classifier.training_date.isoformat(),
    }


def run_train_only(
    client: Any,
    market_ticker: str = "SPY",
    *,
    market_json: str = "hmm_market_results.json",
    results_json: str = "hmm_results.json",
    market_model_path: str = "models/hmm_market_model.json",
) -> dict[str, Any]:
    """Update macro HMM first, then train one individual HMM per price ticker."""
    from core.hmm_market import MarketRegimeClassifier
    from data.indicators import prepareFeaturesForHMM

    market_ticker = market_ticker.upper()
    market_prices = _fetch_prices(client, market_ticker)
    market_model = MarketRegimeClassifier()
    market_model.fit(market_prices, market_ticker=market_ticker)
    market_model.save_model(market_model_path)
    market_features = market_model.compute_features(
        market_prices.set_index("price_date", drop=False)
    )
    for timestamp, row in market_features.iterrows():
        row = row.copy()
        row.name = timestamp
        market_state = market_model.step(row)
    market = _market_result(market_model, market_ticker, market_state)
    _write_result_json(market_json, market)
    client.table("hmm_market_results").upsert(
        market, on_conflict="market_ticker,result_date"
    ).execute()

    result_date, tickers = latest_price_date_tickers(client)
    if not tickers:
        raise ValueError(f"No tickers found for latest stock_prices date {result_date}")

    results: list[dict[str, Any]] = []
    for ticker in tickers:
        try:
            prices = _fetch_prices(client, ticker)
            model = HMMVolatilityClassifier()
            model.fit(prices)
            features = prepareFeaturesForHMM(prices, model.min_periods)
            for index in range(len(features)):
                model.step(features.iloc[[index]])
            state = model.get_current_regime_state()
            volatility_level, stock_multiplier = model.get_regime_for_allocation()
            # Layer 1 reduces the allocation supplied by the individual HMM;
            # it never increases it beyond the stock model's risk decision.
            position_multiplier = round(stock_multiplier * market_state.base_multiplier, 4)
            results.append({
                "ticker": ticker,
                "result_date": result_date,
                "regime_label": state.label,
                "state_id": state.state_id,
                "probability": state.probability,
                "state_probabilities": state.state_probabilities.tolist(),
                "volatility_level": volatility_level,
                "position_multiplier": position_multiplier,
                "is_confirmed": state.is_confirmed,
                "is_flickering": model.is_flickering(),
                "stability_bars": state.consecutive_bars,
                "bic_score": model.bic_score,
                "n_regimes": model.n_regimes,
                "training_date": model.training_date.isoformat(),
            })
        except Exception as exc:
            logger.warning("Skipping %s: %s", ticker, exc)

    if not results:
        raise RuntimeError("No individual HMM result could be calculated")
    payload = {"market": market, "result_date": result_date, "results": results}
    _write_result_json(results_json, payload)
    client.table("hmm_results").upsert(
        results, on_conflict="ticker,result_date"
    ).execute()
    return payload


class TwoLayerRegimeEngine:
    """Compatibility facade around the one supported train-only workflow."""

    def __init__(self, market_ticker: str = "SPY", client: Any | None = None):
        self.market_ticker = market_ticker.upper()
        self.client = client

    def run(self, client: Any | None = None, **kwargs: Any) -> dict[str, Any]:
        client = client or self.client
        if client is None:
            raise ValueError("A Supabase client is required")
        return run_train_only(client, self.market_ticker, **kwargs)
