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