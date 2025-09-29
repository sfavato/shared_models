import pandas as pd
import numpy as np
from .features import calculate_divergence_score, oi_weighted_funding_momentum, trapped_trader_score
from .pipeline import preprocessing_pipeline

def generate_confidence_scores(
    price: pd.Series,
    cvd: pd.Series,
    open_interest: pd.Series,
    funding_rate: pd.Series,
    long_liquidations: pd.Series,
    short_liquidations: pd.Series,
    lookback_period: int = 20
) -> np.ndarray:
    """
    Orchestre le calcul complet des scores de confiance.

    Cette fonction prend les séries temporelles brutes, calcule tous les facteurs,
    les combine, puis les traite à travers le pipeline de prétraitement.

    Args:
        price (pd.Series): Série des prix.
        cvd (pd.Series): Série du Cumulative Volume Delta.
        open_interest (pd.Series): Série de l'Open Interest.
        funding_rate (pd.Series): Série des taux de financement.
        long_liquidations (pd.Series): Série des liquidations 'long'.
        short_liquidations (pd.Series): Série des liquidations 'short'.
        lookback_period (int, optional): Fenêtre de calcul pour les facteurs. Defaults to 20.

    Returns:
        np.ndarray: Un tableau NumPy contenant les caractéristiques finales,
                    normalisées et dé-corrélées, prêtes pour un modèle ML.
    """
    # ÉTAPE 1: Calculer chaque facteur brut en utilisant les fonctions importées.
    divergence = calculate_divergence_score(price, cvd, lookback_period)
    momentum = oi_weighted_funding_momentum(funding_rate, open_interest, lookback_period)
    trapped_traders = trapped_trader_score(price, cvd, long_liquidations, short_liquidations, lookback_period)

    # ÉTAPE 2: Combiner les facteurs en un seul DataFrame.
    features_df = pd.DataFrame({
        'divergence_score': divergence,
        'oi_funding_momentum': momentum,
        'trapped_trader_score': trapped_traders
    })

    # ÉTAPE 3: Gérer les valeurs manquantes.
    # Remplir les NaN qui peuvent résulter des fenêtres glissantes avant de traiter.
    features_df.bfill(inplace=True) # Rétro-propagation
    features_df.ffill(inplace=True) # Propagation avant

    if features_df.empty or features_df.isnull().values.any():
        # Si le DataFrame est toujours vide ou contient des NaN, retourner un tableau vide.
        return np.array([])

    # ÉTAPE 4: Appliquer le pipeline de prétraitement.
    # Le pipeline s'occupe de la normalisation (Quantile) et de la dé-corrélation (PCA).
    processed_features = preprocessing_pipeline.fit_transform(features_df)

    return processed_features