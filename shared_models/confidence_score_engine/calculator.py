import joblib
import pandas as pd
import logging
import os
from google.cloud import storage
from io import BytesIO

# Configuration
MODEL_BUCKET_NAME = "adept-coda-420809.appspot.com"
MODEL_FILE_NAME = "confidence_model.pkl"
PREPROCESSOR_FILE_NAME = "preprocessor.pkl"
# Colonnes attendues par le modèle, dans le bon ordre
EXPECTED_FEATURES = ['open_interest', 'funding_rate', 'long_short_ratio']

logger = logging.getLogger(__name__)

class ConfidenceScoreCalculator:
    """
    Charge les modèles IA (XGBoost et Preprocessor) depuis GCS
    et calcule le score de confiance prédictif.
    """
    def __init__(self, bucket_name=MODEL_BUCKET_NAME, model_path=MODEL_FILE_NAME, prep_path=PREPROCESSOR_FILE_NAME):
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.prep_path = prep_path
        self.storage_client = None
        self.model = None
        self.preprocessor = None

    def _initialize_client(self):
        """Initialise le client GCS."""
        if self.storage_client is None:
            try:
                self.storage_client = storage.Client()
                logger.info("Client Google Cloud Storage initialisé.")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du client GCS: {e}")
                raise

    def load_models(self):
        """
        Charge le modèle et le préprocesseur depuis GCS.
        À appeler une fois au démarrage du service.
        """
        self._initialize_client()
        try:
            bucket = self.storage_client.bucket(self.bucket_name)

            # Télécharger et charger le préprocesseur
            prep_blob = bucket.blob(self.prep_path)
            prep_file_obj = BytesIO()
            prep_blob.download_to_file(prep_file_obj)
            prep_file_obj.seek(0)
            self.preprocessor = joblib.load(prep_file_obj)
            logger.info(f"Préprocesseur '{self.prep_path}' chargé depuis GCS.")

            # Télécharger et charger le modèle
            model_blob = bucket.blob(self.model_path)
            model_file_obj = BytesIO()
            model_blob.download_to_file(model_file_obj)
            model_file_obj.seek(0)
            self.model = joblib.load(model_file_obj)
            logger.info(f"Modèle '{self.model_path}' chargé depuis GCS.")

            logger.info("✅ Modèles de score de confiance prêts.")
            return True

        except Exception as e:
            logger.error(f"Échec critique du chargement des modèles depuis GCS: {e}", exc_info=True)
            return False

    def calculate_score(self, live_features: dict) -> float:
        """
        Calcule le score de confiance à partir des features live.

        Args:
            live_features (dict): Un dict contenant les 3 features.
                                 Ex: {'open_interest': 12345, 'funding_rate': 0.01, 'long_short_ratio': 1.5}

        Returns:
            float: Le score de confiance (probabilité de succès de 0.0 à 1.0).
        """
        if not self.model or not self.preprocessor:
            logger.warning("Modèles non chargés. Tentative de chargement...")
            if not self.load_models():
                logger.error("Calcul du score impossible: Modèles non disponibles.")
                return 0.0  # Score par défaut en cas d'échec

        try:
            # 1. Créer un DataFrame dans le bon ordre
            # Cela garantit que le préprocesseur et le modèle voient les colonnes comme prévu
            df = pd.DataFrame([live_features], columns=EXPECTED_FEATURES)

            # 2. Transformer les données
            data_transformed = self.preprocessor.transform(df)

            # 3. Prédire la probabilité
            # model.predict_proba() renvoie [[prob_classe_0, prob_classe_1]]
            proba = self.model.predict_proba(data_transformed)

            # 4. Extraire la probabilité de succès (classe 1)
            confidence_score = float(proba[0][1])

            logger.info(f"Score de confiance calculé : {confidence_score:.4f}")
            return confidence_score

        except Exception as e:
            logger.error(f"Erreur lors du calcul du score : {e}. Données reçues : {live_features}", exc_info=True)
            return 0.0 # Score par défaut en cas d'échec