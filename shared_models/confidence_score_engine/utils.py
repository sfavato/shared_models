import pandas as pd

def normalize_score(series: pd.Series) -> pd.Series:
    """
    Prend en entrée les Z-scores bruts et les re-calibrera sur une échelle de -1 à +1.

    Méthode : "min-max scaling" adaptée pour une échelle de -1 à 1.
    """
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0] * len(series), index=series.index)

    normalized_series = 2 * (series - min_val) / (max_val - min_val) - 1
    return normalized_series
