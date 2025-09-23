import unittest
import pandas as pd
import numpy as np
from shared_models.confidence_score_engine.calculator import calculate_zscore_momentum

class TestCalculator(unittest.TestCase):

    def test_calculate_zscore_momentum(self):
        """
        Teste la fonction de calcul du Z-score.
        """
        # Crée une série simple pour laquelle le z-score est facile à calculer
        data = pd.Series([10.0, 12.0, 11.0, 15.0, 16.0, 14.0])
        window = 3

        # Calcul manuel pour le 3ème élément (index 2)
        # Fenêtre: [10, 12, 11], moyenne = 11, std = 1.0
        # z-score = (11 - 11) / 1 = 0

        # Calcul pour le 4ème élément (index 3)
        # Fenêtre: [12, 11, 15], moyenne = 12.666..., std = 2.0816...
        # z-score = (15 - 12.666...) / 2.0816... = 1.12090

        result = calculate_zscore_momentum(data, window=window)

        # Vérifie la présence de NaNs dans la fenêtre initiale
        self.assertTrue(pd.isna(result[0]))
        self.assertTrue(pd.isna(result[1]))

        # Vérifie les valeurs calculées
        self.assertAlmostEqual(result[2], 0.0)
        self.assertAlmostEqual(result[3], 1.12090, places=5)

        # Vérifie le type de la sortie
        self.assertIsInstance(result, pd.Series)

if __name__ == '__main__':
    unittest.main()
