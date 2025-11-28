import pandas as pd
import numpy as np
import os
import logging
from .features import (
    calculate_divergence_score,
    oi_weighted_funding_momentum,
    trapped_trader_score,
    calculate_mvrv_score,
    calculate_exchange_netflow_score,
    calculate_whale_accumulation_score
)
from .pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

def generate_confidence_scores(
    merged_df: pd.DataFrame,
    lookback_period: int = 20,
    pipeline_path: str = None
) -> np.ndarray:
    """
    Orchestre le calcul complet et renvoie [c1, c2, c3] pour chaque ligne.
    c1: Score IA (Pipeline)
    c2: Score Divergence (Structure)
    c3: Score Trapped/Momentum (Sentiment)
    """
    # ÉTAPE 1: Initialiser un dictionnaire pour stocker les features calculées.
    features = {}

    # Sécurisation des colonnes requises
    required_cols = ['close', 'CVD', 'open_interest', 'funding_rate']
    for col in required_cols:
        if col not in merged_df.columns:
            logger.error(f"Missing column: {col}")
            return np.array([])

    # Extraire les séries
    price = merged_df['close']
    cvd = merged_df['CVD']
    volume = merged_df.get('volume') # On récupère le volume (présent dans les klines)
    open_interest = merged_df['open_interest']
    funding_rate = merged_df['funding_rate']
    
    # Gestion des optionnels avec get()
    long_liquidations = merged_df.get('long_liquidations_usd')
    short_liquidations = merged_df.get('short_liquidations_usd')
    mvrv = merged_df.get('mvrv_usd')
    netflow = merged_df.get('exchange_netflow_usd')
    whale_accumulation = merged_df.get('whale_accumulation_delta')

    # ÉTAPE 2: Calculer les facteurs de base.
    # PATCH : On passe 'volume' pour activer le fallback OBV
    features['divergence_score'] = calculate_divergence_score(price, cvd, lookback_period, volume=volume)
    features['oi_funding_momentum'] = oi_weighted_funding_momentum(funding_rate, open_interest, lookback_period)

    # ÉTAPE 3: Calculer les facteurs optionnels.
    if long_liquidations is not None and short_liquidations is not None:
        features['trapped_trader_score'] = trapped_trader_score(
            price_close=price,
            long_liquidations=long_liquidations,
            short_liquidations=short_liquidations,
            window=lookback_period
        )
    else:
        # Fallback si pas de liquidations
        features['trapped_trader_score'] = pd.Series(0, index=merged_df.index)

    if mvrv is not None:
        features['mvrv_score'] = calculate_mvrv_score(mvrv)
    if netflow is not None:
        features['netflow_score'] = calculate_exchange_netflow_score(netflow)
    if whale_accumulation is not None:
        features['whale_accumulation_score'] = calculate_whale_accumulation_score(whale_accumulation)

    # ÉTAPE 4: Combiner les facteurs en un seul DataFrame.
    features_df = pd.DataFrame(features, index=merged_df.index)

    # Nettoyage des NaNs
    features_df.bfill(inplace=True)
    features_df.ffill(inplace=True)
    features_df.fillna(0, inplace=True) # Sécurité ultime

    if features_df.empty:
        return np.array([])

    # ÉTAPE 5: Génération des 3 Composantes (Le Patch "User Request")
    
    # --- C1: Le Score ML (Pipeline) ---
    if pipeline_path and os.path.exists(pipeline_path):
        try:
            pipeline = PreprocessingPipeline.load(pipeline_path)
            c1_values = pipeline.transform(features_df)
            # Si le pipeline renvoie un DataFrame, on prend les valeurs
            if hasattr(c1_values, 'values'):
                c1_values = c1_values.values
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            c1_values = np.zeros((len(features_df), 1))
    else:
        # Fallback simple: moyenne des scores normalisés si pas de modèle
        c1_values = np.mean(features_df.values, axis=1).reshape(-1, 1)

    # Assurons-nous que c1 est bien 2D (N, 1)
    if c1_values.ndim == 1:
        c1_values = c1_values.reshape(-1, 1)

    # --- C2: La Structure (Divergence) ---
    c2_values = features_df['divergence_score'].values.reshape(-1, 1)

    # --- C3: Le Sentiment (Trapped Trader ou Momentum) ---
    # On privilégie Trapped Trader, sinon OI Momentum
    if 'trapped_trader_score' in features_df.columns and features_df['trapped_trader_score'].abs().sum() > 0:
        c3_values = features_df['trapped_trader_score'].values.reshape(-1, 1)
    else:
        c3_values = features_df['oi_funding_momentum'].values.reshape(-1, 1)

    # Assemblage final : [C1, C2, C3]
    final_scores = np.hstack([c1_values, c2_values, c3_values])

    return final_scores
