import pytest
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from shared_models.confidence_score_engine.pipeline import PreprocessingPipeline, apply_quantile_transform, apply_pca

@pytest.fixture
def sample_feature_data():
    """Provides a sample DataFrame of engineered features."""
    np.random.seed(42)
    data = np.random.rand(100, 3) * np.array([1, 10, 100]) # Create features with different scales
    return pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])

def test_apply_quantile_transform(sample_feature_data):
    """
    Tests the standalone quantile transformation function.
    """
    transformed_data, transformer = apply_quantile_transform(sample_feature_data)

    assert isinstance(transformed_data, pd.DataFrame)
    assert isinstance(transformer, QuantileTransformer)
    assert transformed_data.shape == sample_feature_data.shape
    assert not transformed_data.isnull().values.any()
    # After a normal distribution transform, mean should be ~0 and std dev ~1
    assert np.allclose(transformed_data.mean(), 0, atol=0.1)
    # The std dev may not be exactly 1, especially with few samples, so we use a wider tolerance.
    assert np.allclose(transformed_data.std(), 1, atol=0.3)

def test_apply_pca(sample_feature_data):
    """
    Tests the standalone PCA function.
    """
    # PCA is applied to already scaled data
    scaled_data, _ = apply_quantile_transform(sample_feature_data)

    principal_components, pca_model = apply_pca(scaled_data, variance_threshold=0.95)

    assert isinstance(principal_components, pd.DataFrame)
    assert isinstance(pca_model, PCA)
    assert principal_components.shape[0] == scaled_data.shape[0]
    assert principal_components.shape[1] <= scaled_data.shape[1]

    # Check that variance explained is >= threshold
    assert pca_model.explained_variance_ratio_.sum() >= 0.95

def test_pipeline_fit_transform(sample_feature_data):
    """
    Tests the fit_transform method of the PreprocessingPipeline.
    """
    pipeline = PreprocessingPipeline(variance_threshold=0.95)
    processed_data = pipeline.fit_transform(sample_feature_data)

    assert isinstance(processed_data, pd.DataFrame)
    assert pipeline.is_fitted
    assert processed_data.shape[0] == sample_feature_data.shape[0]
    # The number of components should be less than or equal to the original number of features
    assert processed_data.shape[1] <= sample_feature_data.shape[1]

def test_pipeline_transform_unfitted_raises_error(sample_feature_data):
    """
    Tests that calling transform on an unfitted pipeline raises a RuntimeError.
    """
    pipeline = PreprocessingPipeline()
    with pytest.raises(RuntimeError):
        pipeline.transform(sample_feature_data)

def test_pipeline_save_and_load(sample_feature_data, tmp_path):
    """
    Tests that a pipeline can be saved, loaded, and produce identical results.
    """
    # Create and fit the original pipeline
    original_pipeline = PreprocessingPipeline(n_quantiles=100, variance_threshold=0.9)
    original_pipeline.fit(sample_feature_data)

    # Define file path for saving
    save_path = tmp_path / "test_pipeline.joblib"

    # Save the pipeline
    original_pipeline.save(save_path)
    assert os.path.exists(save_path)

    # Load the pipeline into a new instance
    loaded_pipeline = PreprocessingPipeline.load(save_path)

    assert loaded_pipeline.is_fitted
    # Check that loaded params are correct
    assert loaded_pipeline.pipeline.named_steps['quantile_transformer'].n_quantiles == 100
    assert loaded_pipeline.pipeline.named_steps['pca'].n_components == 0.9

    # Create new data to transform
    np.random.seed(0) # Use a different seed for new data
    new_data = pd.DataFrame(np.random.rand(50, 3), columns=['feature1', 'feature2', 'feature3'])

    # Transform new data with both pipelines
    original_result = original_pipeline.transform(new_data)
    loaded_result = loaded_pipeline.transform(new_data)

    # Assert that the results are identical
    pd.testing.assert_frame_equal(original_result, loaded_result)

def test_pipeline_save_unfitted_raises_error(tmp_path):
    """
    Tests that calling save on an unfitted pipeline raises a RuntimeError.
    """
    pipeline = PreprocessingPipeline()
    save_path = tmp_path / "unfitted_pipeline.joblib"
    with pytest.raises(RuntimeError):
        pipeline.save(save_path)