import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
import joblib

# Import the class to be tested
from shared_models.confidence_score_engine.calculator import ConfidenceScoreCalculator

# Define simple, pickleable mock classes
class MockPreprocessor:
    def transform(self, data):
        # Simulate the data transformation
        return [[0.1, 0.2, 0.3]]

class MockModel:
    def predict_proba(self, data):
        # Simulate probability prediction
        return [[0.2, 0.8]]  # [prob_class_0, prob_class_1]

@pytest.fixture
def mock_gcs_and_pickleable_models():
    """
    A fixture that mocks the GCS client and provides pickleable mock objects for
    the preprocessor and model. This avoids the PicklingError with MagicMock.
    """
    # Instantiate our simple mock classes
    mock_preprocessor = MockPreprocessor()
    mock_model = MockModel()

    # Serialize the mock objects into in-memory buffers
    prep_buffer = BytesIO()
    joblib.dump(mock_preprocessor, prep_buffer)
    prep_buffer.seek(0)

    model_buffer = BytesIO()
    joblib.dump(mock_model, model_buffer)
    model_buffer.seek(0)

    # Patch the GCS client to avoid real network calls
    with patch('google.cloud.storage.Client') as mock_client:
        mock_bucket = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket

        # Configure mock blobs to serve the serialized mock objects
        mock_prep_blob = MagicMock()
        mock_prep_blob.download_to_file.side_effect = lambda f: f.write(prep_buffer.read())

        mock_model_blob = MagicMock()
        mock_model_blob.download_to_file.side_effect = lambda f: f.write(model_buffer.read())

        def get_blob(blob_name):
            if blob_name == "preprocessor.pkl":
                prep_buffer.seek(0) # Reset buffer before each read
                return mock_prep_blob
            elif blob_name == "confidence_model.pkl":
                model_buffer.seek(0) # Reset buffer before each read
                return mock_model_blob
            return MagicMock()

        mock_bucket.blob.side_effect = get_blob

        # Yield the mocked client and instances of our mocks for assertions
        yield mock_client, mock_preprocessor, mock_model


def test_calculator_initialization():
    """Test that the calculator initializes with default values."""
    calculator = ConfidenceScoreCalculator()
    assert calculator.bucket_name is not None
    assert calculator.model is None
    assert calculator.preprocessor is None
    assert calculator.storage_client is None


def test_load_models_success(mock_gcs_and_pickleable_models):
    """Test the successful loading of models from the mocked GCS."""
    calculator = ConfidenceScoreCalculator()
    assert calculator.load_models() is True
    assert calculator.model is not None
    assert calculator.preprocessor is not None
    assert hasattr(calculator.preprocessor, 'transform')
    assert hasattr(calculator.model, 'predict_proba')


def test_load_models_gcs_failure():
    """Test the model loading failure path when GCS is unavailable."""
    with patch('google.cloud.storage.Client') as mock_client:
        mock_client.return_value.bucket.side_effect = Exception("GCS connection failed")
        calculator = ConfidenceScoreCalculator()
        assert calculator.load_models() is False


def test_calculate_score_loads_models_automatically(mock_gcs_and_pickleable_models):
    """Test that `calculate_score` loads models if they haven't been loaded yet."""
    calculator = ConfidenceScoreCalculator()
    assert calculator.model is None  # Ensure models are not loaded initially

    live_features = {'open_interest': 100, 'funding_rate': 0.01, 'long_short_ratio': 1.2}
    score = calculator.calculate_score(live_features)

    assert score == 0.8  # Expected probability of class 1
    assert calculator.model is not None  # Models should now be loaded
    assert calculator.preprocessor is not None


def test_calculate_score_with_preloaded_models(mock_gcs_and_pickleable_models):
    """Test score calculation when models are already loaded."""
    calculator = ConfidenceScoreCalculator()
    calculator.load_models()  # Pre-load the models

    live_features = {'open_interest': 100, 'funding_rate': 0.01, 'long_short_ratio': 1.2}
    score = calculator.calculate_score(live_features)

    assert score == 0.8


def test_calculate_score_returns_zero_on_error(mock_gcs_and_pickleable_models):
    """Test that `calculate_score` returns 0.0 if an error occurs during prediction."""
    with patch('joblib.load') as mock_joblib_load:
        # Simulate an error during model loading/deserialization
        mock_joblib_load.side_effect = Exception("Deserialization failed")

        calculator = ConfidenceScoreCalculator()

        live_features = {'open_interest': 100, 'funding_rate': 0.01, 'long_short_ratio': 1.2}
        score = calculator.calculate_score(live_features)

        assert score == 0.0
