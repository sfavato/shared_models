import pandas as pd
from .features import calculate_geometric_purity_score, get_historical_performance_score

def calculate_zscore_momentum(series: pd.Series, window: int = 24) -> pd.Series:
    """
    Calcule le Z-score d'une série pandas sur une fenêtre glissante.

    Formule : (valeur_actuelle - moyenne_mobile) / ecart_type_mobile
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    z_score = (series - rolling_mean) / rolling_std
    return z_score

def calculate_derivatives_score(derivatives_data: pd.DataFrame) -> int:
    """
    Calcule un score basé sur les métriques de dérivés (open interest, funding rate).

    Input:
        derivatives_data (pd.DataFrame): Doit contenir 'open_interest' et 'funding_rate'.

    Output:
        int: Score représentant la pression haussière/baissière.
    """
    if derivatives_data.empty:
        return 0

    # Calcul des Z-scores pour l'open interest et le funding rate
    oi_zscore = calculate_zscore_momentum(derivatives_data['open_interest'])
    fr_zscore = calculate_zscore_momentum(derivatives_data['funding_rate'])

    # Récupération des dernières valeurs de Z-score
    latest_oi_zscore = oi_zscore.iloc[-1]
    latest_fr_zscore = fr_zscore.iloc[-1]

    score = 0

    # Logique de scoring, en s'assurant que les z-scores ne sont pas NaN
    if pd.notna(latest_oi_zscore) and latest_oi_zscore > 1:
        score += 1

    if pd.notna(latest_fr_zscore):
        if latest_fr_zscore < -1.5:
            score += 2
        elif latest_fr_zscore > 1.5:
            score -= 2

    return score


def calculate_onchain_score(onchain_data: pd.DataFrame) -> int:
    """
    Calcule un score basé sur les métriques on-chain (MVRV, Netflow).
    """
    if onchain_data.empty:
        return 0

    mvrv_zscore = calculate_zscore_momentum(onchain_data.get('mvrv_usd', pd.Series(dtype='float64')))
    netflow_zscore = calculate_zscore_momentum(onchain_data.get('exchange_netflow_usd', pd.Series(dtype='float64')))

    latest_mvrv_zscore = mvrv_zscore.iloc[-1] if not mvrv_zscore.empty else None
    latest_netflow_zscore = netflow_zscore.iloc[-1] if not netflow_zscore.empty else None

    score = 0

    # MVRV > 1 (Z-score > 1) est baissier
    if pd.notna(latest_mvrv_zscore) and latest_mvrv_zscore > 1.5:
        score -= 2
    # MVRV < 1 (Z-score < -1) est haussier
    elif pd.notna(latest_mvrv_zscore) and latest_mvrv_zscore < -1.5:
        score += 2

    # Netflow négatif (sorties, Z-score < -1) est haussier
    if pd.notna(latest_netflow_zscore) and latest_netflow_zscore < -1.0:
        score += 1
    # Netflow positif (entrées, Z-score > 1) est baissier
    elif pd.notna(latest_netflow_zscore) and latest_netflow_zscore > 1.0:
        score -= 1

    return score


def calculate_final_score(
    pattern_details: dict,
    derivatives_data: pd.DataFrame,
    onchain_data: pd.DataFrame,
    db_connection
) -> float:
    """
    Calcule le score de confiance final en agrégeant les scores des différentes features.

    Args:
        pattern_details (dict): Détails du pattern harmonique.
        derivatives_data (pd.DataFrame): Données sur les dérivés (OI, funding rate).
        onchain_data (pd.DataFrame): Données on-chain (MVRV, netflow).
        db_connection: Connexion à la base de données pour les données historiques.

    Returns:
        float: Le score de confiance final.
    """
    # Poids pour chaque composant du score
    WEIGHTS = {
        'geometric_purity': 0.5,
        'derivatives': 0.3,
        'onchain': 0.2
    }

    # Calculer le score de pureté géométrique (Base score: 0-10)
    geometric_score = calculate_geometric_purity_score(pattern_details)

    # Calculer le score des dérivés (Score: -2 à +3 typiquement)
    derivatives_score = calculate_derivatives_score(derivatives_data)

    # Calculer le score on-chain
    onchain_score = calculate_onchain_score(onchain_data)

    # Calculer le score de performance historique (Bonus/Malus: -1.0, 0.5, 1.5)
    historical_bonus = get_historical_performance_score(pattern_details, db_connection)

    # Calculer le score pondéré
    weighted_score = (geometric_score * WEIGHTS['geometric_purity']) + \
                     (derivatives_score * WEIGHTS['derivatives']) + \
                     (onchain_score * WEIGHTS['onchain'])

    # Ajouter le bonus/malus historique
    final_score = weighted_score + historical_bonus

    # S'assurer que le score final reste dans une plage raisonnable (ex: 0-10)
    return max(0, min(10, final_score))
