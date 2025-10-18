import pandas as pd
import numpy as np
from .features import calculate_divergence_score, oi_weighted_funding_momentum, trapped_trader_score
from .pipeline import PreprocessingPipeline

def generate_confidence_scores(
    merged_df: pd.DataFrame,
    lookback_period: int = 20
) -> np.ndarray:
    """
    Orchestre le calcul complet des scores de confiance à partir d'un DataFrame consolidé.

    Args:
        merged_df (pd.DataFrame): DataFrame contenant toutes les séries de données nécessaires,
                                  y compris 'close', 'CVD', 'open_interest', 'funding_rate',
                                  'long_liquidations_usd', et 'short_liquidations_usd'.
        lookback_period (int): La fenêtre glissante pour les calculs de features.

    Returns:
        np.ndarray: Un tableau numpy des scores de confiance traités.
    """
    # ÉTAPE 1: Initialiser un dictionnaire pour stocker les features calculées.
    features = {}

    # Extraire les séries de données du DataFrame
    price = merged_df['close']
    cvd = merged_df['CVD']
    open_interest = merged_df['open_interest']
    funding_rate = merged_df['funding_rate']
    long_liquidations = merged_df.get('long_liquidations_usd')  # Utiliser .get() pour la flexibilité
    short_liquidations = merged_df.get('short_liquidations_usd')

    # ÉTAPE 2: Calculer les facteurs de base (toujours présents).
    features['divergence_score'] = calculate_divergence_score(price, cvd, lookback_period)
    features['oi_funding_momentum'] = oi_weighted_funding_momentum(funding_rate, open_interest, lookback_period)

    # ÉTAPE 3: Calculer les facteurs optionnels seulement si les données sont fournies.
    if long_liquidations is not None and short_liquidations is not None:
        features['trapped_trader_score'] = trapped_trader_score(
            price_close=price,
            long_liquidations=long_liquidations,
            short_liquidations=short_liquidations,
            window=lookback_period
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