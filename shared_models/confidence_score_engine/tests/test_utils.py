import unittest
import pandas as pd
import numpy as np
from shared_models.confidence_score_engine.utils import normalize_score

class TestUtils(unittest.TestCase):

    def test_normalize_score(self):
        """
        Teste la fonction de normalisation sur une série standard.
        """
        data = pd.Series([-10.0, -5.0, 0.0, 5.0, 10.0])
        result = normalize_score(data)

        self.assertAlmostEqual(result.min(), -1.0)
        self.assertAlmostEqual(result.max(), 1.0)
        self.assertAlmostEqual(result[2], 0.0) # La valeur centrale 0 doit être mappée à 0
        self.assertIsInstance(result, pd.Series)

    def test_normalize_score_constant_series(self):
        """
        Teste la fonction de normalisation avec une série constante.
        """
        data = pd.Series([5.0, 5.0, 5.0, 5.0])
        result = normalize_score(data)
        self.assertTrue((result == 0).all())

    def test_normalize_score_with_nan(self):
        """
        Teste que les NaNs sont gérés correctement.
        La normalisation doit s'opérer sur les valeurs non-NaN et le NaN doit être préservé.
        """
        data = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        result = normalize_score(data)

        # Le min et le max des valeurs non-NaN doivent être mappés à -1 et 1
        self.assertAlmostEqual(result.dropna().min(), -1.0)
        self.assertAlmostEqual(result.dropna().max(), 1.0)

        # Le NaN doit être préservé à sa position originale
        self.assertTrue(pd.isna(result[2]))

if __name__ == '__main__':
    unittest.main()
