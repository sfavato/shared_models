import unittest
import pandas as pd
import numpy as np
from shared_models.confidence_score_engine import compute_confidence_score

class TestMain(unittest.TestCase):

    def test_compute_confidence_score(self):
        """
        Teste la fonction principale d'orchestration.
        """
        # Crée un DataFrame d'exemple
        data = {
            'open_interest': np.linspace(1000, 2000, 50),
            'funding_rate': np.sin(np.linspace(0, 2 * np.pi, 50)) * 0.01
        }
        df = pd.DataFrame(data)

        # La fenêtre par défaut est de 24, donc les 23 premières lignes de scores doivent être NaN
        window = 24
        result_df = compute_confidence_score(df)

        # 1. Vérifie si la sortie est un DataFrame
        self.assertIsInstance(result_df, pd.DataFrame)

        # 2. Vérifie la présence des colonnes requises
        expected_cols = [
            'open_interest', 'funding_rate', 'oi_zscore', 'fr_zscore',
            'oi_normalized_score', 'fr_normalized_score', 'confidence_score'
        ]
        for col in expected_cols:
            self.assertIn(col, result_df.columns)

        # 3. Vérifie les NaNs initiaux dus à la fenêtre glissante
        self.assertTrue(result_df['confidence_score'].head(window - 1).isna().all())
        self.assertFalse(pd.isna(result_df['confidence_score'].iloc[window - 1]))

        # 4. Vérifie que le score de confiance est dans l'intervalle [-1, 1]
        valid_scores = result_df['confidence_score'].dropna()
        self.assertTrue((valid_scores >= -1).all() and (valid_scores <= 1).all())

        # 5. Vérifie que les données originales ne sont pas modifiées
        self.assertNotIn('confidence_score', df.columns)


if __name__ == '__main__':
    unittest.main()
