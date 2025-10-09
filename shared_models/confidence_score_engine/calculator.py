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
