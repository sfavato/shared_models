import pandas as pd
import numpy as np
import os
from typing import Optional
from .features import (
    calculate_divergence_score,
    oi_weighted_funding_momentum,
    trapped_trader_score,
    calculate_mvrv_score,
    calculate_exchange_netflow_score,
    calculate_whale_accumulation_score
)
from .pipeline import PreprocessingPipeline

def generate_confidence_scores(
    merged_df: pd.DataFrame,
    lookback_period: int = 20,
    pipeline_path: Optional[str] = None
) -> np.ndarray:
    """
    Orchestre le calcul complet des scores de confiance à partir d'un DataFrame consolidé.

    Cette fonction agit comme le point d'entrée principal du moteur de scoring. Elle transforme
    les données brutes de marché (OHLCV, On-Chain) en un vecteur de probabilités utilisable
    par le système de décision. Elle assure l'extraction des features (Feature Engineering)
    puis leur normalisation via le pipeline ML pré-entraîné.

    Args:
        merged_df (pd.DataFrame): DataFrame contenant toutes les séries de données nécessaires synchronisées par timestamp.
        lookback_period (int): La fenêtre glissante pour les calculs de features (ex: divergences).
        pipeline_path (Optional[str]): Le chemin vers le fichier .pkl du pipeline scikit-learn entraîné.

    Returns:
        np.ndarray: Un tableau numpy des scores de confiance/features traités, prêt pour l'inférence ou le stockage.
    """
    # ÉTAPE 1: Initialiser un dictionnaire pour stocker les features calculées.
    features = {}

    # Extraction sécurisée des séries de données du DataFrame.
    # L'utilisation de .get() permet une robustesse si certaines colonnes optionnelles manquent.
    price = merged_df['close']
    cvd = merged_df['CVD']
    open_interest = merged_df['open_interest']
    funding_rate = merged_df['funding_rate']
    long_liquidations = merged_df.get('long_liquidations_usd')
    short_liquidations = merged_df.get('short_liquidations_usd')
    mvrv = merged_df.get('mvrv_usd')
    netflow = merged_df.get('exchange_netflow_usd')
    whale_accumulation = merged_df.get('whale_accumulation_delta')

    # ÉTAPE 2: Calculer les facteurs de base (Order Flow & Sentiment).
    features['divergence_score'] = calculate_divergence_score(price, cvd, lookback_period)
    features['oi_funding_momentum'] = oi_weighted_funding_momentum(funding_rate, open_interest, lookback_period)

    # ÉTAPE 3: Calculer les facteurs optionnels (Liquidations, On-Chain) si les données sont présentes.
    if long_liquidations is not None and short_liquidations is not None:
        features['trapped_trader_score'] = trapped_trader_score(
            price_close=price,
            long_liquidations=long_liquidations,
            short_liquidations=short_liquidations,
            window=lookback_period
        )

    if mvrv is not None:
        features['mvrv_score'] = calculate_mvrv_score(mvrv)
    if netflow is not None:
        features['netflow_score'] = calculate_exchange_netflow_score(netflow)
    if whale_accumulation is not None:
        features['whale_accumulation_score'] = calculate_whale_accumulation_score(whale_accumulation)

    # ÉTAPE 4: Combiner les facteurs en un seul DataFrame aligné temporellement.
    features_df = pd.DataFrame(features)

    # Gestion des valeurs manquantes (Backfill puis Forwardfill) pour garantir la continuité des données.
    features_df.bfill(inplace=True)
    features_df.ffill(inplace=True)

    # Vérification de sécurité : si le DataFrame est vide ou corrompu, on échoue silencieusement.
    if features_df.empty or features_df.isnull().values.any():
        return np.array([])

    # ÉTAPE 5: Application du Pipeline ML (Normalisation + PCA).
    # On ne transforme les données que si un pipeline entraîné est fourni.
    if pipeline_path and os.path.exists(pipeline_path):
        pipeline = PreprocessingPipeline.load(pipeline_path)
        # IMPORTANT : Utiliser transform(), jamais fit(), pour éviter la fuite de données (data leakage) en production.
        processed_features_df = pipeline.transform(features_df)
    else:
        # Si pas de pipeline, on retourne un tableau vide ou on pourrait retourner les raw features selon le besoin.
        # Ici, la convention est de retourner vide pour signaler l'absence de traitement ML valide.
        return np.array([])

    return processed_features_df.to_numpy()
