import logging
import os
from typing import Union
from decimal import Decimal, getcontext

logger = logging.getLogger(__name__)

# Définir la précision pour les calculs décimaux
getcontext().prec = 10

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
        # Vérification du mode Premium
        premium_data_active = os.getenv("PREMIUM_DATA_ACTIVE", "True").lower() == "true"

        if premium_data_active:
            # Pondération standard (Phase II) : Priorité à l'IA et aux données complexes
            w_ml = Decimal('0.40')
            w_purity = Decimal('0.20')
            w_regime = Decimal('0.25')
            w_history = Decimal('0.15')
        else:
            # Mode "Dégradé" / Non-Premium :
            # On coupe le poids du ML (qui dépend des features Premium manquantes)
            # On redistribue sur l'Analyse Technique pure (Purity) et l'Historique
            w_ml = Decimal('0.00')
            w_purity = Decimal('0.80')  # Forte emphase sur la qualité géométrique
            w_regime = Decimal('0.00')  # On ignore le régime s'il dépend de données externes complexes
            w_history = Decimal('0.20') # On garde un peu d'historique

        ml_proba_d = Decimal(str(ml_proba))
        purity_score_d = Decimal(str(purity_score))
        market_regime_score_d = Decimal(str(market_regime_score))
        win_rate_score_d = Decimal(str(win_rate_score))

        final_score = (ml_proba_d * w_ml) + \
                      (purity_score_d * w_purity) + \
                      (market_regime_score_d * w_regime) + \
                      (win_rate_score_d * w_history)

        final_score_normalized = float(
            (final_score * Decimal('10')).quantize(Decimal('0.1'))
        )

        logger.info(
            f"Score Dynamique Calculé : {final_score_normalized}/10 "
            f"(ML: {ml_proba:.2f}, Pureté: {purity_score:.2f}, "
            f"Régime: {market_regime_score:.2f}, Historique: {win_rate_score:.2f})"
        )

        return final_score_normalized

    def calculate_score(self, features_dict: dict) -> float:
        """
        Wrapper pour compatibilité avec l'ancienne interface.
        """
        ml_proba = features_dict.get('ml_proba', 0.5)
        purity_score = features_dict.get('purity', 0.5)
        market_regime_score = features_dict.get('regime', 0.5)
        win_rate_score = features_dict.get('win_rate', 0.5)

        return self.calculate_dynamic_score(
            ml_proba=ml_proba,
            purity_score=purity_score,
            market_regime_score=market_regime_score,
            win_rate_score=win_rate_score
        )

# Alias pour la compatibilité ascendante
ConfidenceScoreCalculator = DynamicScoreCalculator
