import logging
from typing import Union

logger = logging.getLogger(__name__)

class DynamicScoreCalculator:
    """
    Moteur de calcul pour l'évaluation de la confiance des trades.

    Cette classe agrège plusieurs signaux hétérogènes (IA, Analyse Technique, Données Fondamentales)
    pour produire un score unifié. L'objectif est de réduire la complexité décisionnelle pour
    le système de trading en fournissant une métrique unique et pondérée.
    """

    @staticmethod
    def calculate_dynamic_score(ml_proba: float, purity_score: float, market_regime_score: float, win_rate_score: float) -> float:
        """
        Calcule un score dynamique pondéré à partir de quatre piliers fondamentaux.

        Cette méthode applique une pondération spécifique à chaque composante pour refléter
        sa fiabilité relative et son importance stratégique dans la prise de décision actuelle.
        L'objectif est de ne pas se fier à une seule source de signal, mais de chercher la
        confluence entre l'IA, la géométrie du marché, le contexte macro (régime) et l'historique.

        Args:
            ml_proba (float): Probabilité issue du modèle ML (0.0 - 1.0), représentant la prédiction statistique pure.
            purity_score (float): Score de pureté géométrique du pattern (0.0 - 1.0), validant la qualité technique du setup.
            market_regime_score (float): Score du régime de marché (0.0 - 1.0), pour éviter de trader à contre-tendance ou dans une volatilité dangereuse.
            win_rate_score (float): Score basé sur le taux de victoire historique (0.0 - 1.0), apportant une preuve empirique de succès passé.

        Returns:
            float: Le score final, normalisé sur une échelle de 0 à 10 et arrondi à une décimale, prêt pour le filtrage décisionnel.
        """
        # Pondération recommandée pour Démarrage Phase II
        # Ces poids sont ajustés pour donner la priorité à l'IA tout en gardant des garde-fous techniques et fondamentaux.
        w_ml = 0.40       # L'IA reste le signal primaire (40%).
        w_purity = 0.20   # La qualité visuelle/technique du pattern (20%).
        w_regime = 0.25   # Le contexte de marché (25%), crucial pour la gestion du risque.
        w_history = 0.15  # La performance passée (15%), comme validateur supplémentaire.

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
