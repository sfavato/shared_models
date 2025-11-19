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

    def calculate_score(self, live_features: dict, purity_score: float = 5.0) -> float:
        """
        Calcule un Score Hybride pondéré.

        Formule : Score Final = (Probabilité ML * 70%) + (Pureté Géométrique * 30%)

        Args:
            live_features (dict): Features pour le modèle ML (OI, Funding, etc.)
            purity_score (float): Score de pureté technique sur 10 (défaut: 5.0)
        """
        # --- 1. Obtenir le score ML (Probabilité 0.0 - 1.0) ---
        ml_probability = 0.0

        if not self.model or not self.preprocessor:
            logger.warning("Modèles non chargés. Tentative de chargement...")
            if not self.load_models():
                logger.error("Calcul ML impossible : Modèles non disponibles. Fallback sur Pureté.")
                # Si le ML échoue, on se base uniquement sur la pureté normalisée (0-1)
                return purity_score / 10.0

        try:
            df = pd.DataFrame([live_features], columns=EXPECTED_FEATURES)
            data_transformed = self.preprocessor.transform(df)
            proba = self.model.predict_proba(data_transformed)
            ml_probability = float(proba[0][1])
            logger.info(f"Score ML Brut : {ml_probability:.4f}")

        except Exception as e:
            logger.error(f"Erreur calcul ML : {e}. Fallback sur Pureté.", exc_info=True)
            return purity_score / 10.0

        # --- 2. Normaliser la Pureté (0-10 -> 0.0-1.0) ---
        # On s'assure que le score est borné entre 0 et 10 avant de diviser
        clean_purity = max(0.0, min(10.0, float(purity_score)))
        purity_factor = clean_purity / 10.0

        # --- 3. Calculer le Score Composite ---
        # Poids : 70% ML (Context), 30% Technique (Geometry)
        # Ce réglage permet au ML de rester dominant tout en laissant la qualité du pattern
        # faire la différence sur les cas limites (ex: ML à 0.45 mais Pureté à 9/10 -> Score final ~0.58)
        WEIGHT_ML = 0.7
        WEIGHT_PURITY = 0.3

        final_score = (ml_probability * WEIGHT_ML) + (purity_factor * WEIGHT_PURITY)

        logger.info(f"Score Hybride Calculé : {final_score:.4f} (ML: {ml_probability:.2f}, Pureté: {clean_purity:.1f})")

        return final_score
