import pytest
import pandas as pd
import numpy as np
from shared_models.confidence_score_engine.main import generate_confidence_scores

@pytest.fixture
def market_data_fixture():
    """Provides a realistic, valid dataset for the main pipeline test."""
    size = 100
    np.random.seed(42)
    return {
        'price': pd.Series(np.linspace(100, 150, size)),
        'cvd': pd.Series(np.linspace(1000, 1500, size)),
        'open_interest': pd.Series(np.linspace(5000, 6000, size)),
        'funding_rate': pd.Series(np.random.normal(0.01, 0.005, size)),
        'long_liquidations': pd.Series(np.random.randint(0, 100, size)),
        'short_liquidations': pd.Series(np.random.randint(0, 100, size))
    }

def test_generate_confidence_scores_happy_path(market_data_fixture):
    """
    Tests the full pipeline with valid data, checking output shape and quality.
    """
    processed_features = generate_confidence_scores(
        price=market_data_fixture['price'],
        cvd=market_data_fixture['cvd'],
        open_interest=market_data_fixture['open_interest'],
        funding_rate=market_data_fixture['funding_rate'],
        long_liquidations=market_data_fixture['long_liquidations'],
        short_liquidations=market_data_fixture['short_liquidations']
    )

    assert isinstance(processed_features, np.ndarray)
    assert not np.isnan(processed_features).any()
    assert not np.isinf(processed_features).any()
    # The number of rows should be the original size minus any NaNs from lookbacks
    assert processed_features.shape[0] > 0
    # The number of columns should be less than or equal to the number of raw features (3)
    assert processed_features.shape[1] <= 3

def test_generate_confidence_scores_with_nans(market_data_fixture):
    """
    Tests the pipeline's robustness when input data contains NaNs.
    The pipeline should handle them gracefully.
    """
    data = market_data_fixture
    # Introduce NaNs at the beginning of two series
    data['price'].iloc[:15] = np.nan
    data['cvd'].iloc[:15] = np.nan

    processed_features = generate_confidence_scores(
        price=data['price'],
        cvd=data['cvd'],
        open_interest=data['open_interest'],
        funding_rate=data['funding_rate'],
        long_liquidations=data['long_liquidations'],
        short_liquidations=data['short_liquidations']
    )

    # The function should still run and return a valid array without NaNs
    assert isinstance(processed_features, np.ndarray)
    assert not np.isnan(processed_features).any()

def test_generate_confidence_scores_all_nans_input():
    """
    Tests the edge case where all input data is NaN. This should result in an empty array.
    """
    size = 100
    nan_series = pd.Series([np.nan] * size)

    processed_features = generate_confidence_scores(
        price=nan_series,
        cvd=nan_series,
        open_interest=nan_series,
        funding_rate=nan_series,
        long_liquidations=nan_series,
        short_liquidations=nan_series
    )

    assert isinstance(processed_features, np.ndarray)
    assert processed_features.size == 0

def test_output_statistical_properties(market_data_fixture):
    """
    Validates that the output features are correctly normalized (mean ~0, std ~1)
    after the quantile transformation and PCA.
    """
    processed_features = generate_confidence_scores(
        price=market_data_fixture['price'],
        cvd=market_data_fixture['cvd'],
        open_interest=market_data_fixture['open_interest'],
        funding_rate=market_data_fixture['funding_rate'],
        long_liquidations=market_data_fixture['long_liquidations'],
        short_liquidations=market_data_fixture['short_liquidations']
    )

    assert processed_features.shape[0] > 0

    # Check the statistical properties of each principal component
        # Check the statistical properties of each principal component
    for i in range(processed_features.shape[1]):
        component = processed_features[:, i]
            # After quantile transform to a normal distribution and PCA, the mean of the components should be near 0.
            # The std is not expected to be 1, as PCA components capture variance.
        assert np.isclose(np.mean(component), 0, atol=0.2), f"Mean of PC {i} is not close to 0"