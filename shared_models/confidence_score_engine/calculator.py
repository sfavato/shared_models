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
EXPECTED_FEATURES = ['open_interest', 'funding_rate', 'long_short_ratio']

logger = logging.getLogger(__name__)

class ConfidenceScoreCalculator:
    def __init__(self, bucket_name=MODEL_BUCKET_NAME, model_path=MODEL_FILE_NAME, prep_path=PREPROCESSOR_FILE_NAME):
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.prep_path = prep_path
        self.storage_client = None
        self.model = None
        self.preprocessor = None

    def _initialize_client(self):
        if self.storage_client is None:
            try:
                self.storage_client = storage.Client()
                logger.info("Client Google Cloud Storage initialisé.")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du client GCS: {e}")
                raise

    def load_models(self):
        self._initialize_client()
        try:
            bucket = self.storage_client.bucket(self.bucket_name)

            # Charger le préprocesseur
            prep_blob = bucket.blob(self.prep_path)
            prep_file_obj = BytesIO()
            prep_blob.download_to_file(prep_file_obj)
            prep_file_obj.seek(0)
            self.preprocessor = joblib.load(prep_file_obj)
            logger.info(f"Préprocesseur chargé.")

            # Charger le modèle
            model_blob = bucket.blob(self.model_path)
            model_file_obj = BytesIO()
            model_blob.download_to_file(model_file_obj)
            model_file_obj.seek(0)
            self.model = joblib.load(model_file_obj)
            logger.info(f"Modèle chargé.")

            return True
        except Exception as e:
            logger.error(f"Échec critique du chargement des modèles: {e}", exc_info=True)
            return False

    def calculate_score(self, live_features: dict) -> float:
        if not self.model or not self.preprocessor:
            logger.warning("Modèles non chargés. Tentative de chargement...")
            if not self.load_models():
                return 0.0

        try:
            # Création du DataFrame avec les features dans l'ordre exact
            df = pd.DataFrame([live_features], columns=EXPECTED_FEATURES)

            # Transformation et Prédiction
            data_transformed = self.preprocessor.transform(df)
            proba = self.model.predict_proba(data_transformed)

            # Retourne la probabilité de la classe 1 (Trade Gagnant)
            return float(proba[0][1])

        except Exception as e:
            logger.error(f"Erreur calcul score ML: {e}", exc_info=True)
            return 0.0