import logging

logger = logging.getLogger(__name__)

class DynamicScoreCalculator:
    @staticmethod
    def calculate_dynamic_score(ml_proba: float, purity_score: float, market_regime_score: float, win_rate_score: float) -> float:
        """
        Calcule un score dynamique pondéré à partir de quatre piliers.

        Args:
            ml_proba (float): Probabilité du modèle ML (0.0 - 1.0).
            purity_score (float): Score de pureté géométrique du pattern (0.0 - 1.0).
            market_regime_score (float): Score du régime de marché (tendance, volatilité) (0.0 - 1.0).
            win_rate_score (float): Score basé sur le taux de victoire historique (0.0 - 1.0).

        Returns:
            float: Le score final, normalisé sur 10 et arrondi à une décimale.
        """
        # Pondération recommandée pour Démarrage Phase II
        w_ml = 0.40       # L'IA reste le maître
        w_purity = 0.20   # La qualité du pattern (HarmoFinder)
        w_regime = 0.25   # Ne pas trader contre le vent (Update_Indices)
        w_history = 0.15  # La preuve par l'historique (DB)

        final_score = (ml_proba * w_ml) + \
                      (purity_score * w_purity) + \
                      (market_regime_score * w_regime) + \
                      (win_rate_score * w_history)

        final_score_normalized = round(final_score * 10, 1)

        logger.info(
            f"Score Dynamique Calculé : {final_score_normalized}/10 "
            f"(ML: {ml_proba:.2f}, Pureté: {purity_score:.2f}, "
            f"Régime: {market_regime_score:.2f}, Historique: {win_rate_score:.2f})"
        )

        return final_score_normalized
