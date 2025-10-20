import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from shared_models.confidence_score_engine.calculator import ConfidenceScoreCalculator

@pytest.fixture
def mock_gcs():
    """Fixture to mock Google Cloud Storage client and blob."""
    with patch('google.cloud.storage.Client') as mock_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        yield mock_client, mock_blob

@pytest.fixture
def mock_joblib():
    """Fixture to mock joblib.load."""
    with patch('joblib.load') as mock_load:
        mock_model = MagicMock()
        mock_preprocessor = MagicMock()

        # Configure mocks to return a model and a preprocessor
        mock_load.side_effect = [mock_preprocessor, mock_model]

        # Mock the prediction logic
        mock_model.predict_proba.return_value = [[0.2, 0.8]] # Example probability

        # Mock the transform logic
        mock_preprocessor.transform.return_value = pd.DataFrame({'feature': [1]})

        yield mock_load, mock_model, mock_preprocessor

def test_load_models_success(mock_gcs, mock_joblib):
    """Test successful loading of models from GCS."""
    mock_client, mock_blob = mock_gcs
    mock_load, _, _ = mock_joblib

    calculator = ConfidenceScoreCalculator()
    assert calculator.load_models() is True
    assert calculator.model is not None
    assert calculator.preprocessor is not None
    mock_client.return_value.bucket.assert_called_with("adept-coda-420809.appspot.com")
    assert mock_blob.download_to_file.call_count == 2
    assert mock_load.call_count == 2

def test_load_models_failure(mock_gcs):
    """Test failure of loading models from GCS."""
    mock_client, mock_blob = mock_gcs
    mock_blob.download_to_file.side_effect = Exception("GCS error")

    calculator = ConfidenceScoreCalculator()
    assert calculator.load_models() is False
    assert calculator.model is None
    assert calculator.preprocessor is None

def test_calculate_score_success(mock_gcs, mock_joblib):
    """Test successful score calculation."""
    live_features = {'open_interest': 12345, 'funding_rate': 0.01, 'long_short_ratio': 1.5}

    calculator = ConfidenceScoreCalculator()

    # Pre-load models
    calculator.load_models()

    score = calculator.calculate_score(live_features)

    assert score == 0.8

    _, mock_model, mock_preprocessor = mock_joblib

    mock_preprocessor.transform.assert_called_once()
    mock_model.predict_proba.assert_called_once()

def test_calculate_score_models_not_loaded(mock_gcs, mock_joblib):
    """Test that models are loaded if not already present."""
    live_features = {'open_interest': 12345, 'funding_rate': 0.01, 'long_short_ratio': 1.5}

    calculator = ConfidenceScoreCalculator()

    # Do not pre-load models
    score = calculator.calculate_score(live_features)

    assert score == 0.8
    assert calculator.model is not None
    assert calculator.preprocessor is not None

def test_calculate_score_loading_fails(mock_gcs):
    """Test score calculation when model loading fails."""
    mock_client, mock_blob = mock_gcs
    mock_blob.download_to_file.side_effect = Exception("GCS error")

    live_features = {'open_interest': 12345, 'funding_rate': 0.01, 'long_short_ratio': 1.5}

    calculator = ConfidenceScoreCalculator()

    score = calculator.calculate_score(live_features)

    assert score == 0.0

def test_calculate_score_prediction_fails(mock_gcs, mock_joblib):
    """Test score calculation when prediction fails."""
    _, mock_model, _ = mock_joblib
    mock_model.predict_proba.side_effect = Exception("Prediction error")

    live_features = {'open_interest': 12345, 'funding_rate': 0.01, 'long_short_ratio': 1.5}

    calculator = ConfidenceScoreCalculator()
    calculator.load_models()

    score = calculator.calculate_score(live_features)

    assert score == 0.0