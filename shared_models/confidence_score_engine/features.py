import pandas as pd

def calculate_divergence_score(price: pd.Series, cvd: pd.Series, lookback_period: int) -> pd.Series:
    """
    Calcule un score basé sur les divergences entre le prix et le CVD.

    Une divergence haussière (+1) est détectée si le prix atteint un nouveau plus bas
    sur la période de référence (lookback) alors que le CVD est plus haut que son plus bas précédent.
    Une divergence baissière (-1) est détectée si le prix atteint un nouveau plus haut
    alors que le CVD est plus bas que son plus haut précédent.

    Args:
        price (pd.Series): Série temporelle des prix.
        cvd (pd.Series): Série temporelle du Cumulative Volume Delta.
        lookback_period (int): La fenêtre glissante (en nombre de périodes) pour
                               identifier les plus hauts/bas.

    Returns:
        pd.Series: Une série temporelle contenant le score de divergence lissé.
    """
    # ÉTAPE 1: Trouver les extrêmes du prix et du CVD sur la fenêtre glissante.
    price_high = price.rolling(window=lookback_period).max()
    cvd_high = cvd.rolling(window=lookback_period).max()
    price_low = price.rolling(window=lookback_period).min()
    cvd_low = cvd.rolling(window=lookback_period).min()

    # ÉTAPE 2: Identifier les moments où le prix actuel atteint un nouveau sommet/creux.
    is_new_price_high = (price == price_high)
    is_new_price_low = (price == price_low)

    # ÉTAPE 3: Définir les conditions de divergence.
    # Note: On compare le CVD actuel aux extrêmes précédents (.shift(1)) pour détecter la non-confirmation.
    bearish_divergence = (is_new_price_high) & (cvd < cvd_high.shift(1))
    bullish_divergence = (is_new_price_low) & (cvd > cvd_low.shift(1))

    # ÉTAPE 4: Créer le score brut.
    # Le score est +1 pour une divergence haussière, -1 pour une baissière.
    divergence_score = bullish_divergence.astype(int) - bearish_divergence.astype(int)

    # ÉTAPE 5: Lisser le signal pour capturer un état persistant.
    # Utiliser une somme glissante sur une petite fenêtre (ex: 5 périodes).
    smoothed_score = divergence_score.rolling(window=5, min_periods=1).sum()

    return smoothed_score


def oi_weighted_funding_momentum(funding_rate: pd.Series, open_interest: pd.Series, lookback: int) -> pd.Series:
    """
    Crée un facteur de sentiment en pondérant la tendance du taux de financement
    par le momentum de l'Open Interest (OI).

    Un score positif élevé suggère un fort momentum haussier engagé, tandis qu'un
    score négatif élevé suggère un fort momentum baissier, potentiellement
    sur-endetté et propice à un retournement.

    Args:
        funding_rate (pd.Series): Série temporelle des taux de financement.
        open_interest (pd.Series): Série temporelle de l'Open Interest.
        lookback (int): La fenêtre glissante pour calculer la tendance et le momentum.

    Returns:
        pd.Series: Une série temporelle représentant le facteur de sentiment.
    """
    # ÉTAPE 1: Calculer le momentum de l'Open Interest.
    # Utiliser le taux de changement en pourcentage (Rate of Change) sur la période de référence.
    # fill_method=None est spécifié pour se conformer aux futures versions de pandas.
    oi_roc = open_interest.pct_change(periods=lookback, fill_method=None)

    # ÉTAPE 2: Calculer la tendance du taux de financement.
    # Utiliser une moyenne mobile simple sur la même période.
    funding_ma = funding_rate.rolling(window=lookback).mean()

    # ÉTAPE 3: Calculer le facteur de sentiment.
    # La formule est la tendance du financement, amplifiée par la croissance de l'OI.
    # Remplir les valeurs NaN initiales du RoC avec 0 pour ne pas perdre de points de données.
    sentiment_factor = funding_ma * (1 + oi_roc.fillna(0))

    return sentiment_factor


def trapped_trader_score(price_close: pd.Series, long_liquidations: pd.Series, short_liquidations: pd.Series, window: int = 24) -> pd.Series:
    """
    Calcule un score indiquant la probabilité d'une activité récente de traders piégés,
    basé sur les pics de liquidations coïncidant avec des mouvements de prix significatifs.

    Args:
        price_close (pd.Series): Série temporelle des prix de clôture.
        long_liquidations (pd.Series): Série temporelle des liquidations 'long'.
        short_liquidations (pd.Series): Série temporelle des liquidations 'short'.
        window (int): La fenêtre glissante pour détecter les pics de liquidations.

    Returns:
        pd.Series: Un score de traders piégés.
    """
    # ÉTAPE 1: Calculer le volume total des liquidations.
    total_liquidations = long_liquidations + short_liquidations

    # ÉTAPE 2: Identifier les pics de liquidations.
    # Un pic est un point de données qui dépasse la moyenne mobile + 2 écarts-types.
    liq_mean = total_liquidations.rolling(window=window).mean()
    liq_std = total_liquidations.rolling(window=window).std()
    liquidation_spikes = (total_liquidations > (liq_mean + 2 * liq_std)).astype(int)

    # ÉTAPE 3: Évaluer la direction du mouvement des prix pendant les pics.
    # Un changement de prix négatif pendant un pic de liquidations 'long' -> traders piégés.
    # Un changement de prix positif pendant un pic de liquidations 'short' -> traders piégés.
    price_change = price_close.diff()

    # Score pour les longs piégés (liquidations long + baisse de prix)
    trapped_longs_score = (liquidation_spikes * (price_change < 0) * long_liquidations).rank(pct=True)

    # Score pour les shorts piégés (liquidations short + hausse de prix)
    trapped_shorts_score = (liquidation_spikes * (price_change > 0) * short_liquidations).rank(pct=True)

    # ÉTAPE 4: Combiner les scores.
    # Le score final est la somme des deux, indiquant l'intensité globale de l'activité piégée.
    # On ajoute les deux scores pour capturer l'effet combiné.
    final_score = (trapped_longs_score + trapped_shorts_score).fillna(0)

    # ÉTAPE 5: Lisser le score final pour plus de stabilité.
    return final_score.rolling(window=3).mean()


def calculate_geometric_purity_score(pattern_details: dict) -> float:
    """
    Calcule un score de "pureté géométrique" pour un pattern harmonique.

    Le score est basé sur la proximité des ratios réels du pattern par rapport
    aux ratios de Fibonacci idéaux.

    Args:
        pattern_details (dict): Un dictionnaire contenant:
            - 'name' (str): Le nom du pattern (ex: 'Gartley').
            - 'ratios' (dict): Un dictionnaire des ratios mesurés (ex: {'B': 0.618, 'C': 0.786, ...}).

    Returns:
        float: Un score normalisé entre 0 et 10, où 10 est un pattern parfait.
    """
    IDEAL_RATIOS = {
        'Gartley': {'B': 0.618, 'C': 0.786, 'D': 0.786, 'XA': 0.786},
        'Butterfly': {'B': 0.786, 'C': 0.886, 'D': 1.272, 'XA': 1.272},
        'Bat': {'B': 0.5, 'C': 0.886, 'D': 0.886, 'XA': 0.886},
        'Crab': {'B': 0.618, 'C': 0.886, 'D': 1.618, 'XA': 1.618}
    }

    pattern_name = pattern_details.get('name')
    actual_ratios = pattern_details.get('ratios', {})

    if pattern_name not in IDEAL_RATIOS:
        return 0.0  # Pattern non reconnu

    ideal = IDEAL_RATIOS[pattern_name]
    errors = []

    for point, ideal_ratio in ideal.items():
        actual_ratio = actual_ratios.get(point)
        if actual_ratio is not None:
            error = (actual_ratio - ideal_ratio) ** 2
            errors.append(error)

    if not errors:
        return 0.0

    # Calcul de l'erreur quadratique moyenne (MSE)
    mse = sum(errors) / len(errors)

    # Normalisation du score: 1 / (1 + MSE) pour borner entre 0 et 1, puis mise à l'échelle sur 10.
    # Un MSE de 0 donne un score de 10 (parfait).
    # Un MSE de 0.1 donne un score de ~9.09.
    # Un MSE de 1.0 donne un score de 5.0.
    normalized_score = 1 / (1 + mse)
    final_score = normalized_score * 10

    return final_score


def get_historical_performance_score(pattern_details: dict, db_connection) -> float:
    """
    Calcule un score bonus/malus basé sur la performance historique d'un pattern.

    Args:
        pattern_details (dict): Dictionnaire contenant les détails du pattern
                                (name, symbol, timeframe).
        db_connection: Une connexion à la base de données pour exécuter des requêtes.

    Returns:
        float: Un score de type bonus/malus.
    """
    pattern_name = pattern_details.get('name')
    symbol = pattern_details.get('symbol')
    timeframe = pattern_details.get('timeframe')

    if not all([pattern_name, symbol, timeframe, db_connection]):
        return 0.0

    try:
        cursor = db_connection.cursor()
        query = """
            SELECT win_rate
            FROM trades_log
            WHERE pattern_name = %s AND symbol = %s AND timeframe = %s
            ORDER BY trade_date DESC
            LIMIT 100;
        """
        cursor.execute(query, (pattern_name, symbol, timeframe))
        results = cursor.fetchall()

        if not results:
            return 0.0  # Pas d'historique, score neutre

        # Calculer le win rate moyen des trades historiques
        win_rates = [row[0] for row in results]
        average_win_rate = sum(win_rates) / len(win_rates)

        # Appliquer le barème de bonus/malus
        if average_win_rate > 0.7:
            return 1.5
        elif average_win_rate > 0.5:
            return 0.5
        elif average_win_rate < 0.4:
            return -1.0
        else:
            return 0.0  # Score neutre pour une performance moyenne

    except Exception as e:
        print(f"Database error in get_historical_performance_score: {e}")
        return 0.0  # En cas d'erreur, retourner un score neutre