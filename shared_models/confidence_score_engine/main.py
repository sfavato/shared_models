import pandas as pd
from .calculator import calculate_zscore_momentum
from .utils import normalize_score

def compute_confidence_score(derivatives_df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestre le processus de calcul du Score de Confiance.

    a. Appelle calculate_zscore_momentum sur les colonnes open_interest et funding_rate.
    b. Appelle normalize_score sur chacun des Z-scores obtenus.
    c. Calcule le Score de Confiance Composite en faisant la moyenne simple des deux scores normalisés.
    d. Retourne le DataFrame original enrichi avec les colonnes des scores intermédiaires et du score final.
    """
    result_df = derivatives_df.copy()

    # Étape 1: Calculer les Z-Scores
    oi_zscore = calculate_zscore_momentum(result_df['open_interest'])
    fr_zscore = calculate_zscore_momentum(result_df['funding_rate'])

    result_df['oi_zscore'] = oi_zscore
    result_df['fr_zscore'] = fr_zscore

    # Étape 2: Normaliser les Scores
    # La normalisation est calculée sur la partie non-NaN des z-scores.
    # Le résultat sera une série avec des NaNs pour la fenêtre initiale, ce qui est correct.
    oi_normalized = normalize_score(oi_zscore.dropna())
    fr_normalized = normalize_score(fr_zscore.dropna())

    result_df['oi_normalized_score'] = oi_normalized
    result_df['fr_normalized_score'] = fr_normalized

    # Étape 3: Calculer le Score Composite
    # La moyenne propagera correctement les NaNs.
    result_df['confidence_score'] = (result_df['oi_normalized_score'] + result_df['fr_normalized_score']) / 2

    return result_df
