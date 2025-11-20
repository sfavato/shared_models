import pandas as pd
import joblib
from typing import Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def apply_quantile_transform(data: pd.DataFrame, n_quantiles: int = 1000, random_state: int = 42) -> Tuple[pd.DataFrame, QuantileTransformer]:
    """
    Applique une transformation quantile aux données d'entrée (Gaussian Rank Transform).

    Cette fonction force les données à suivre une distribution normale (Gaussienne), ce qui est
    crucial pour de nombreux modèles ML et pour réduire l'impact des valeurs aberrantes (outliers).
    Elle préserve l'ordre des données (rang) mais déforme les distances.

    Args:
        data (pd.DataFrame): DataFrame contenant les alpha factors bruts.
        n_quantiles (int): Nombre de quantiles utilisés pour discrétiser la fonction de distribution cumulative.
        random_state (int): Graine aléatoire pour assurer la reproductibilité des transformations.

    Returns:
        Tuple[pd.DataFrame, QuantileTransformer]: Les données transformées et l'instance du transformateur ajusté.
    """
    transformer = QuantileTransformer(
        output_distribution='normal',
        n_quantiles=n_quantiles,
        random_state=random_state
    )
    transformed_data = pd.DataFrame(transformer.fit_transform(data), columns=data.columns, index=data.index)
    return transformed_data, transformer

def apply_pca(data: pd.DataFrame, variance_threshold: float = 0.95) -> Tuple[pd.DataFrame, PCA]:
    """
    Applique une Analyse en Composantes Principales (PCA) pour réduire la dimensionnalité.

    L'objectif est de supprimer la colinéarité entre les features corrélés (ex: RSI et MACD peuvent porter la même info)
    et de concentrer l'information ("signal") dans un nombre réduit de composantes orthogonales.
    Cela simplifie le modèle final et réduit le bruit.

    Args:
        data (pd.DataFrame): DataFrame des features déjà normalisés (ex: via QuantileTransform).
        variance_threshold (float): Le pourcentage de variance expliquée à conserver (ex: 0.95 pour 95%).

    Returns:
        Tuple[pd.DataFrame, PCA]: DataFrame contenant les Composantes Principales et l'objet PCA ajusté.
    """
    pca = PCA(n_components=variance_threshold)
    principal_components = pd.DataFrame(pca.fit_transform(data), index=data.index)
    principal_components.columns = [f"PC_{i+1}" for i in range(principal_components.shape[1])]
    return principal_components, pca


class DtypeCoercer(BaseEstimator, TransformerMixin):
    """
    Un transformateur personnalisé pour forcer le typage float64.

    Assure que toutes les données entrant dans le pipeline sont numériques, évitant les erreurs
    silencieuses ou les exceptions sklearn liées aux types d'objets ou entiers mixtes.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.astype('float64')


class PreprocessingPipeline:
    """
    Un pipeline de prétraitement réutilisable enchaînant coercition de type, QuantileTransformer et PCA.

    Cette classe encapsule toute la logique de préparation des données pour garantir que
    les données d'inférence (production) subissent EXACTEMENT les mêmes transformations
    que les données d'entraînement. Elle gère la persistance (sauvegarde/chargement) de l'état du pipeline.
    """
    def __init__(self, n_quantiles: int = 1000, variance_threshold: float = 0.95, random_state: int = 42):
        """
        Initialise la structure du pipeline.

        Args:
            n_quantiles (int): Paramètre pour QuantileTransformer.
            variance_threshold (float): Seuil de variance pour PCA.
            random_state (int): Seed pour la reproductibilité.
        """
        self.pipeline = Pipeline([
            (
                'dtype_coercer',
                DtypeCoercer()
            ),
            (
                'quantile_transformer',
                QuantileTransformer(
                    output_distribution='normal',
                    n_quantiles=n_quantiles,
                    random_state=random_state
                )
            ),
            (
                'pca',
                PCA(n_components=variance_threshold)
            )
        ])
        self.is_fitted = False

    def fit(self, data: pd.DataFrame) -> 'PreprocessingPipeline':
        """
        Ajuste (entraîne) le pipeline sur les données historiques.

        Calcule les quantiles et les vecteurs propres de la PCA basés sur le jeu de données fourni.

        Args:
            data (pd.DataFrame): Le dataset d'entraînement (facteurs bruts).

        Returns:
            PreprocessingPipeline: L'instance elle-même (fluent interface).
        """
        self.pipeline.fit(data)
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applique le pipeline ajusté pour transformer de nouvelles données (Inférence).

        ATTENTION : Ne doit être appelé que si le pipeline est déjà 'fitted'.

        Args:
            data (pd.DataFrame): Les nouvelles données brutes.

        Returns:
            pd.DataFrame: Les données transformées (Composantes Principales).
        """
        if not self.is_fitted:
            raise RuntimeError("The pipeline must be fitted before transforming data.")

        principal_components = self.pipeline.transform(data)
        num_components = principal_components.shape[1]

        # Création de noms de colonnes explicites pour la traçabilité
        pc_columns = [f"PC_{i+1}" for i in range(num_components)]

        return pd.DataFrame(principal_components, columns=pc_columns, index=data.index)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ajuste le pipeline et transforme les données en une seule étape.

        Utile lors de la phase d'entraînement initial.

        Args:
            data (pd.DataFrame): Le dataset d'entraînement.

        Returns:
            pd.DataFrame: Les données transformées.
        """
        self.is_fitted = True
        principal_components = self.pipeline.fit_transform(data)
        num_components = principal_components.shape[1]

        # Création de noms de colonnes explicites
        pc_columns = [f"PC_{i+1}" for i in range(num_components)]

        return pd.DataFrame(principal_components, columns=pc_columns, index=data.index)

    def save(self, filepath: str) -> None:
        """
        Sérialise et sauvegarde le pipeline ajusté dans un fichier (.pkl).

        Permet de persister l'intelligence apprise (quantiles, vecteurs PCA) pour l'utiliser
        dans d'autres microservices sans ré-entraînement.

        Args:
            filepath (str): Le chemin de destination du fichier.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save a pipeline that has not been fitted.")
        joblib.dump(self.pipeline, filepath)

    @staticmethod
    def load(filepath: str) -> 'PreprocessingPipeline':
        """
        Charge un pipeline ajusté depuis un fichier.

        Reconstruit l'objet PreprocessingPipeline complet à partir de l'artefact scikit-learn sauvegardé.

        Args:
            filepath (str): Le chemin du fichier .pkl.

        Returns:
            PreprocessingPipeline: Une instance prête à l'emploi (is_fitted=True).
        """
        loaded_pipeline = joblib.load(filepath)

        # Extraction des hyperparamètres pour recréer une instance cohérente
        n_quantiles = loaded_pipeline.named_steps['quantile_transformer'].n_quantiles
        variance_threshold = loaded_pipeline.named_steps['pca'].n_components
        random_state = loaded_pipeline.named_steps['quantile_transformer'].random_state

        # Création de la nouvelle instance
        new_pipeline_instance = PreprocessingPipeline(
            n_quantiles=n_quantiles,
            variance_threshold=variance_threshold,
            random_state=random_state
        )

        # Injection de l'objet pipeline chargé et du flag d'état
        new_pipeline_instance.pipeline = loaded_pipeline
        new_pipeline_instance.is_fitted = True

        return new_pipeline_instance
