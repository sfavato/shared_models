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
    oi_roc = open_interest.pct_change(periods=lookback)

    # ÉTAPE 2: Calculer la tendance du taux de financement.
    # Utiliser une moyenne mobile simple sur la même période.
    funding_ma = funding_rate.rolling(window=lookback).mean()

    # ÉTAPE 3: Calculer le facteur de sentiment.
    # La formule est la tendance du financement, amplifiée par la croissance de l'OI.
    # Remplir les valeurs NaN initiales du RoC avec 0 pour ne pas perdre de points de données.
    sentiment_factor = funding_ma * (1 + oi_roc.fillna(0))

    return sentiment_factor


def trapped_trader_score(price: pd.Series, cvd: pd.Series, long_liquidations: pd.Series, short_liquidations: pd.Series, lookback: int) -> pd.Series:
    """
    Calcule un score indiquant la probabilité d'une activité récente de traders piégés.

    Le score est élevé lorsqu'un retournement brusque du prix et du CVD se produit
    simultanément, surtout s'il est confirmé par un pic de liquidations.

    Args:
        price (pd.Series): Série temporelle des prix.
        cvd (pd.Series): Série temporelle du Cumulative Volume Delta.
        long_liquidations (pd.Series): Série temporelle des liquidations 'long'.
        short_liquidations (pd.Series): Série temporelle des liquidations 'short'.
        lookback (int): La fenêtre glissante pour les calculs d'accélération et de pics.

    Returns:
        pd.Series: Une série temporelle contenant le score final lissé des traders piégés.
    """
    # ÉTAPE 1: Identifier les retournements brusques de prix via l'accélération.
    # L'accélération est la dérivée seconde (différence de la différence).
    price_acceleration = price.diff().diff().abs()
    price_reversal_signal = price_acceleration.rolling(window=lookback).mean()

    # ÉTAPE 2: Identifier les retournements brusques de flux (CVD) via l'accélération.
    cvd_acceleration = cvd.diff().diff().abs()
    cvd_reversal_signal = cvd_acceleration.rolling(window=lookback).mean()

    # ÉTAPE 3: Identifier les pics de liquidations.
    # Un pic est défini comme un volume total de liquidations dépassant la moyenne
    # plus 2 écarts-types sur une fenêtre plus longue.
    total_liquidations = long_liquidations + short_liquidations
    liq_baseline = total_liquidations.rolling(window=lookback * 5).mean()
    liq_std = total_liquidations.rolling(window=lookback * 5).std()
    is_liquidation_spike = (total_liquidations > (liq_baseline + 2 * liq_std)).astype(int)

    # ÉTAPE 4: Combiner les signaux pour créer le score.
    # On multiplie les signaux de retournement et on normalise le résultat
    # en utilisant le rang en percentile pour le rendre comparable.
    base_score = (price_reversal_signal * cvd_reversal_signal).rank(pct=True)

    # Le pic de liquidation agit comme un "amplificateur" du score de base.
    trapped_score = base_score + is_liquidation_spike

    # ÉTAPE 5: Lisser le score final pour plus de stabilité.
    return trapped_score.rolling(window=3).mean()