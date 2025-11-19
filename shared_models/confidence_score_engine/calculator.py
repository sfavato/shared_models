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
EXPECTED_FEATURES = [
    'open_interest', 
    'funding_rate', 
    'long_short_ratio',
    'cvd',
    'btc_dominance',
    'btc_trend_ma_200',
    'mvrv_score',
    'netflow_score',
    'trapped_trader_score',
    'divergence_score',
    'oi_weighted_funding_momentum',
    'price_vs_poc',
    'price_vs_vah',
    'price_vs_val',
    'is_in_value_area'
]

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

    def calculate_score(self, live_features: dict, geometric_purity: float = 0.5, market_confluence_score: float = 0.5, historical_win_rate: float = 0.5) -> float:
        """
        Calcule un Score Hybride pondéré.

        Formule : Score_Final = (Model_ML_Prob * 0.40) + (Confluence * 0.30) + (Purity * 0.15) + (History * 0.15)

        Args:
            live_features (dict): Features pour le modèle ML (OI, Funding, etc.)
            geometric_purity (float): Score de pureté géométrique (0.0 - 1.0)
            market_confluence_score (float): Score de confluence de marché (0.0 - 1.0)
            historical_win_rate (float): Taux de victoire historique (0.0 - 1.0)
        """
        # --- 1. Obtenir le score ML (Probabilité 0.0 - 1.0) ---
        ml_probability = 0.0

        if not self.model or not self.preprocessor:
            logger.warning("Modèles non chargés. Tentative de chargement...")
            if not self.load_models():
                logger.error("Calcul ML impossible : Modèles non disponibles. Fallback sur les autres scores.")
                # Si le ML échoue, on met sa probabilité à 0.0 pour ne pas influencer le score
                ml_probability = 0.0

        try:
            df = pd.DataFrame([live_features], columns=EXPECTED_FEATURES)
            data_transformed = self.preprocessor.transform(df)
            proba = self.model.predict_proba(data_transformed)
            ml_probability = float(proba[0][1])
            logger.info(f"Score ML Brut : {ml_probability:.4f}")

        except Exception as e:
            logger.error(f"Erreur calcul ML : {e}. Fallback sur les autres scores.", exc_info=True)
            ml_probability = 0.0

        # --- 2. Calculer le Score Composite ---
        WEIGHT_ML = 0.40
        WEIGHT_CONFLUENCE = 0.30
        WEIGHT_PURITY = 0.15
        WEIGHT_HISTORY = 0.15

        final_score_raw = (ml_probability * WEIGHT_ML) + \
                          (market_confluence_score * WEIGHT_CONFLUENCE) + \
                          (geometric_purity * WEIGHT_PURITY) + \
                          (historical_win_rate * WEIGHT_HISTORY)

        # --- 3. Normaliser le score final sur 10 ---
        final_score_normalized = final_score_raw * 10

        logger.info(f"Score Final Calculé : {final_score_normalized:.2f}/10 (ML: {ml_probability:.2f}, Confluence: {market_confluence_score:.2f}, Pureté: {geometric_purity:.2f}, Historique: {historical_win_rate:.2f})")

        return final_score_normalized
