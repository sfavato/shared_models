import pandas as pd
import numpy as np
from .features import calculate_divergence_score, oi_weighted_funding_momentum, trapped_trader_score
from .pipeline import PreprocessingPipeline

def generate_confidence_scores(
    price: pd.Series,
    cvd: pd.Series,
    open_interest: pd.Series,
    funding_rate: pd.Series,
    long_liquidations: pd.Series = None,
    short_liquidations: pd.Series = None,
    lookback_period: int = 20
) -> np.ndarray:
    """
    Orchestre le calcul complet des scores de confiance.
    Calcule uniquement les facteurs pour lesquels les données sont fournies.
    """
    # ÉTAPE 1: Initialiser un dictionnaire pour stocker les features calculées.
    features = {}

    # ÉTAPE 2: Calculer les facteurs de base (toujours présents).
    features['divergence_score'] = calculate_divergence_score(price, cvd, lookback_period)
    features['oi_funding_momentum'] = oi_weighted_funding_momentum(funding_rate, open_interest, lookback_period)

    # ÉTAPE 3: Calculer les facteurs optionnels seulement si les données sont fournies.
    if long_liquidations is not None and short_liquidations is not None:
        features['trapped_trader_score'] = trapped_trader_score(
            price, cvd, long_liquidations, short_liquidations, lookback_period
        )

    # ÉTAPE 4: Combiner les facteurs en un seul DataFrame.
    features_df = pd.DataFrame(features)

    # ... (The rest of the function: NaN handling and pipeline application) ...
    features_df.bfill(inplace=True)
    features_df.ffill(inplace=True)

    if features_df.empty or features_df.isnull().values.any():
        return np.array([])

    # Instantiate and use the preprocessing pipeline
    pipeline = PreprocessingPipeline()
    processed_features_df = pipeline.fit_transform(features_df)

    return processed_features_df.to_numpy()