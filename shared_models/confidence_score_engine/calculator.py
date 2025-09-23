import pandas as pd

def calculate_zscore_momentum(series: pd.Series, window: int = 24) -> pd.Series:
    """
    Calcule le Z-score d'une série pandas sur une fenêtre glissante.

    Formule : (valeur_actuelle - moyenne_mobile) / ecart_type_mobile
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    z_score = (series - rolling_mean) / rolling_std
    return z_score
