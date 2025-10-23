import numpy as np
import pandas as pd
from .features import (
    calculate_market_regime_features,
    calculate_onchain_features
)

def calculate_zscore_momentum(series: pd.Series, period: int = 20) -> float:
    """
    Calcule le Z-score du momentum sur une période donnée.
    """
    if series.empty or len(series) < period:
        return 0.0

    returns = series.pct_change().dropna()

    if len(returns) < period:
        return 0.0

    rolling_mean = returns.rolling(window=period).mean()
    rolling_std = returns.rolling(window=period).std()

    if rolling_std.iloc[-1] == 0:
        return 0.0

    zscore = (returns.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
    return zscore

class ConfidenceScoreCalculator:
    """
    Calcule un score de confiance multi-factoriel pour les signaux de trading.
    Ceci est l'implémentation de Phase 1 (basée sur des règles pondérées)
    telle que définie dans le 'Crypto Algo Trading Confidence Score'.
    """
    def __init__(self, db_engine=None):
        self.weights = {
            'momentum_zscore': 0.4,
            'market_regime': 0.3,
            'on_chain_score': 0.3
        }
        self.db_engine = db_engine

    def _normalize_score(self, score, min_val=-1, max_val=1) -> float:
        """Normalise un score entre 0 et 10."""
        normalized = (score - min_val) / (max_val - min_val)
        return max(0, min(10, normalized * 10))

    def calculate(self, symbol: str, timeframe: str, price_series: pd.Series) -> float:
        """
        Méthode principale pour calculer le score de confiance final.
        """
        try:
            momentum_score = calculate_zscore_momentum(price_series)

            regime_features = calculate_market_regime_features(self.db_engine, symbol, timeframe)
            regime_score = regime_features.get('regime_score', 0)

            onchain_features = calculate_onchain_features(self.db_engine, symbol, timeframe)
            onchain_score = onchain_features.get('cvd_divergence_score', 0)

            final_weighted_score = (
                (momentum_score * self.weights['momentum_zscore']) +
                (regime_score * self.weights['market_regime']) +
                (onchain_score * self.weights['on_chain_score'])
            )

            final_score = self._normalize_score(final_weighted_score)

            return final_score

        except Exception as e:
            print(f"Erreur lors du calcul du Confidence Score pour {symbol}: {e}")
            return 0.0