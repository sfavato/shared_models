import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def apply_quantile_transform(data: pd.DataFrame, n_quantiles: int = 1000, random_state: int = 42) -> (pd.DataFrame, QuantileTransformer):
    """
    Applies a quantile transformation to the input data.

    This function fits a QuantileTransformer on the data, making it robust to outliers
    and transforming the distribution to be normal.

    Args:
        data (pd.DataFrame): DataFrame containing the engineered alpha factors.
        n_quantiles (int): Number of quantiles to be used by the transformer.
        random_state (int): Seed for the random number generator for reproducibility.

    Returns:
        pd.DataFrame: The transformed data.
        QuantileTransformer: The fitted transformer instance.
    """
    transformer = QuantileTransformer(
        output_distribution='normal',
        n_quantiles=n_quantiles,
        random_state=random_state
    )
    transformed_data = pd.DataFrame(transformer.fit_transform(data), columns=data.columns, index=data.index)
    return transformed_data, transformer

def apply_pca(data: pd.DataFrame, variance_threshold: float = 0.95) -> (pd.DataFrame, PCA):
    """
    Applies Principal Component Analysis (PCA) to the transformed data.

    This function reduces redundancy by creating a set of linearly uncorrelated features
    that explain a specified percentage of the variance.

    Args:
        data (pd.DataFrame): DataFrame of the quantile-transformed features.
        variance_threshold (float): The percentage of variance to be explained by the components.

    Returns:
        pd.DataFrame: DataFrame containing the final orthogonal principal components.
        PCA: The fitted PCA instance.
    """
    pca = PCA(n_components=variance_threshold)
    principal_components = pd.DataFrame(pca.fit_transform(data), index=data.index)
    principal_components.columns = [f"PC_{i+1}" for i in range(principal_components.shape[1])]
    return principal_components, pca


class DtypeCoercer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to coerce all columns in a DataFrame to float64.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.astype('float64')


class PreprocessingPipeline:
    """
    A reusable preprocessing pipeline that chains QuantileTransformer and PCA.

    This class encapsulates the entire preprocessing logic, allowing it to be fitted
    on historical data and then used to transform new data consistently.
    """
    def __init__(self, n_quantiles: int = 1000, variance_threshold: float = 0.95, random_state: int = 42):
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

    def fit(self, data: pd.DataFrame):
        """
        Fits the pipeline on the training data.

        Args:
            data (pd.DataFrame): The training dataset with alpha factors.
        """
        self.pipeline.fit(data)
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the fitted pipeline to transform new data.

        Args:
            data (pd.DataFrame): The new data to be transformed.

        Returns:
            pd.DataFrame: The preprocessed data with principal components.
        """
        if not self.is_fitted:
            raise RuntimeError("The pipeline must be fitted before transforming data.")

        principal_components = self.pipeline.transform(data)
        num_components = principal_components.shape[1]

        # Create meaningful column names for the principal components
        pc_columns = [f"PC_{i+1}" for i in range(num_components)]

        return pd.DataFrame(principal_components, columns=pc_columns, index=data.index)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the pipeline and transforms the data in a single step.

        Args:
            data (pd.DataFrame): The training dataset.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        self.is_fitted = True
        principal_components = self.pipeline.fit_transform(data)
        num_components = principal_components.shape[1]

        # Create meaningful column names
        pc_columns = [f"PC_{i+1}" for i in range(num_components)]

        return pd.DataFrame(principal_components, columns=pc_columns, index=data.index)

    def save(self, filepath: str):
        """
        Saves the fitted pipeline to a file.

        Args:
            filepath (str): The path where the pipeline will be saved.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save a pipeline that has not been fitted.")
        joblib.dump(self.pipeline, filepath)

    @staticmethod
    def load(filepath: str):
        """
        Loads a fitted pipeline from a file.

        Args:
            filepath (str): The path to the saved pipeline file.

        Returns:
            PreprocessingPipeline: An instance of this class with the loaded pipeline.
        """
        loaded_pipeline = joblib.load(filepath)

        # Extract parameters to create a new instance
        n_quantiles = loaded_pipeline.named_steps['quantile_transformer'].n_quantiles
        variance_threshold = loaded_pipeline.named_steps['pca'].n_components
        random_state = loaded_pipeline.named_steps['quantile_transformer'].random_state

        # Create a new PreprocessingPipeline instance
        new_pipeline_instance = PreprocessingPipeline(
            n_quantiles=n_quantiles,
            variance_threshold=variance_threshold,
            random_state=random_state
        )

        # Assign the loaded sklearn pipeline and mark as fitted
        new_pipeline_instance.pipeline = loaded_pipeline
        new_pipeline_instance.is_fitted = True

        return new_pipeline_instance