import pandas as pd
import numpy as np
import os
import logging
import joblib
from .features import (
    calculate_divergence_score,
    oi_weighted_funding_momentum,
    trapped_trader_score,
    calculate_mvrv_score,
    calculate_exchange_netflow_score,
    calculate_whale_accumulation_score
)

logger = logging.getLogger(__name__)

def generate_confidence_scores(
    merged_df: pd.DataFrame,
    lookback_period: int = 20,
    model_path: str = None,
    preprocessor_path: str = None
) -> np.ndarray:
    """
    Orchestre le calcul complet.
    Version G.E.M. : Préserve les features externes pour le modèle ML.
    """
    # 1. Extraction et calculs internes
    features = {}
    
    # Sécurisation
    required_cols = ['close', 'CVD', 'open_interest', 'funding_rate']
    for col in required_cols:
        if col not in merged_df.columns:
            logger.error(f"Missing base column: {col}")
            return np.array([])

    price = merged_df['close']
    cvd = merged_df['CVD']
    volume = merged_df.get('volume')
    open_interest = merged_df['open_interest']
    funding_rate = merged_df['funding_rate']
    
    long_liquidations = merged_df.get('long_liquidations_usd')
    short_liquidations = merged_df.get('short_liquidations_usd')
    mvrv = merged_df.get('mvrv_usd')
    netflow = merged_df.get('exchange_netflow_usd')
    whale_accumulation = merged_df.get('whale_accumulation_delta')

    # Calcul des features INTERNES
    features['divergence_score'] = calculate_divergence_score(price, cvd, lookback_period, volume=volume)
    features['oi_weighted_funding_momentum'] = oi_weighted_funding_momentum(funding_rate, open_interest, lookback_period)

    if long_liquidations is not None and short_liquidations is not None:
        features['trapped_trader_score'] = trapped_trader_score(price, long_liquidations, short_liquidations, lookback_period)
    else:
        features['trapped_trader_score'] = pd.Series(0, index=merged_df.index)

    if mvrv is not None: features['mvrv_score'] = calculate_mvrv_score(mvrv)
    if netflow is not None: features['netflow_score'] = calculate_exchange_netflow_score(netflow)
    if whale_accumulation is not None: features['whale_accumulation_score'] = calculate_whale_accumulation_score(whale_accumulation)

    # 2. Création du DataFrame de features
    features_df = pd.DataFrame(features, index=merged_df.index)

    # 3. --- FUSION CRITIQUE DES FEATURES EXTERNES ---
    # Le modèle ML a besoin de ces colonnes qui ont été calculées en amont (dans update_confidence_score.py)
    # On les copie de merged_df vers features_df
    
    external_cols = [
        'price_vs_poc', 'price_vs_vah', 'price_vs_val', 'is_in_value_area',
        'btc_trend_ma_200', 'btc_dominance', 'long_short_ratio', 'cvd'
    ]
    
    for col in external_cols:
        if col in merged_df.columns:
            features_df[col] = merged_df[col]
        else:
            # Fallback silencieux (0.0) si vraiment manquant, pour éviter le crash
            features_df[col] = 0.0

    # Nettoyage
    features_df.bfill(inplace=True)
    features_df.ffill(inplace=True)
    features_df.fillna(0, inplace=True)

    if features_df.empty: return np.array([])

    # 4. Prédiction ML
    c1_values = None
    
    if model_path and os.path.exists(model_path) and preprocessor_path and os.path.exists(preprocessor_path):
        try:
            preprocessor = joblib.load(preprocessor_path)
            model = joblib.load(model_path)
            
            # Le DataFrame contient maintenant TOUTES les colonnes requises
            X_processed = preprocessor.transform(features_df)
            
            try:
                c1_values = model.predict_proba(X_processed)[:, 1].reshape(-1, 1)
            except AttributeError:
                c1_values = model.predict(X_processed).reshape(-1, 1)
            
            c1_values = c1_values * 10
            
        except Exception as e:
            logger.error(f"Erreur chargement ML: {e}. Passage en mode fallback.")
            c1_values = None

    # Fallback
    if c1_values is None:
        tech_cols = ['divergence_score', 'oi_weighted_funding_momentum', 'trapped_trader_score']
        valid_cols = [c for c in tech_cols if c in features_df.columns]
        if valid_cols:
            c1_values = np.mean(features_df[valid_cols].values, axis=1).reshape(-1, 1)
            c1_values = (np.tanh(c1_values) + 1) * 5
        else:
            c1_values = np.zeros((len(features_df), 1))

    c2_values = features_df['divergence_score'].values.reshape(-1, 1)
    
    if 'trapped_trader_score' in features_df.columns and features_df['trapped_trader_score'].abs().sum() > 0:
        c3_values = features_df['trapped_trader_score'].values.reshape(-1, 1)
    else:
        c3_values = features_df['oi_weighted_funding_momentum'].values.reshape(-1, 1)

    return np.hstack([c1_values, c2_values, c3_values])
