"""
HMM Model Module - Hidden Markov Model for Market Regime Detection
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    """Market regime detector using Gaussian HMM."""

    def __init__(self, n_components: int = 7, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.state_returns = None
        self.bull_state = None
        self.bear_state = None
        self.neutral_states = None

    def fit(self, df: pd.DataFrame) -> 'RegimeDetector':
        """
        Fit HMM model on historical data.

        Args:
            df: DataFrame with return, range, volumeVolatility columns
        """
        features = df[['return', 'range', 'volumeVolatility']].values

        features_scaled = self.scaler.fit_transform(features)

        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type='full',
            n_iter=1000,
            random_state=self.random_state
        )
        self.model.fit(features_scaled)

        self._identifyRegimeStates(df)

        return self

    def _identifyRegimeStates(self, df: pd.DataFrame):
        """Identify bull/bear states based on mean returns."""
        features = df[['return', 'range', 'volumeVolatility']].values
        features_scaled = self.scaler.transform(features)
        hidden_states = self.model.predict(features_scaled)

        state_returns = {}
        for state in range(self.n_components):
            mask = hidden_states == state
            if mask.sum() > 0:
                state_returns[state] = df.loc[mask, 'return'].mean()

        self.state_returns = state_returns

        sorted_states = sorted(state_returns.items(), key=lambda x: x[1])
        self.bear_state = sorted_states[0][0]
        self.bull_state = sorted_states[-1][0]
        self.neutral_states = [s[0] for s in sorted_states[1:-1]]

    def predictRegimes(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict regime for each observation.

        Returns:
            Series with regime labels: 'Bull', 'Bear', 'Crash', 'Neutral'
        """
        features = df[['return', 'range', 'volumeVolatility']].values
        features_scaled = self.scaler.transform(features)
        hidden_states = self.model.predict(features_scaled)

        regime_labels = []
        for state in hidden_states:
            if state == self.bull_state:
                regime_labels.append('Bull')
            elif state == self.bear_state:
                if self.state_returns[state] < -0.002:
                    regime_labels.append('Crash')
                else:
                    regime_labels.append('Bear')
            else:
                regime_labels.append('Neutral')

        return pd.Series(regime_labels, index=df.index)

    def getCurrentRegime(self, df: pd.DataFrame) -> str:
        """Get the most recent regime."""
        regimes = self.predictRegimes(df)
        return regimes.iloc[-1]
