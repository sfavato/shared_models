import pytest
import pandas as pd
import numpy as np
from shared_models.confidence_score_engine.main import generate_confidence_scores

@pytest.fixture
def full_market_data():
    """Provides a full, valid dataset for the main pipeline test."""
    size = 100
    return {
        'price': pd.Series(np.linspace(100, 150, size)),
        'cvd': pd.Series(np.linspace(1000, 1500, size)),
        'open_interest': pd.Series(np.linspace(5000, 6000, size)),
        'funding_rate': pd.Series(np.random.normal(0.01, 0.005, size)),
        'long_liquidations': pd.Series(np.random.randint(0, 100, size)),
        'short_liquidations': pd.Series(np.random.randint(0, 100, size))
    }

def test_generate_confidence_scores_happy_path(full_market_data):
    """
    Tests the full pipeline with valid data, checking output shape and quality.
    """
    processed_features = generate_confidence_scores(
        price=full_market_data['price'],
        cvd=full_market_data['cvd'],
        open_interest=full_market_data['open_interest'],
        funding_rate=full_market_data['funding_rate'],
        long_liquidations=full_market_data['long_liquidations'],
        short_liquidations=full_market_data['short_liquidations']
    )

    # 1. Check if the output is a numpy array
    assert isinstance(processed_features, np.ndarray)

    # 2. Check that there are no NaN or infinite values
    assert not np.isnan(processed_features).any()
    assert not np.isinf(processed_features).any()

    # 3. Check that PCA has reduced or maintained the number of features (<= 3)
    # The number of rows should be the original size minus the initial lookback period NaNs that get backfilled
    assert processed_features.shape[0] > 0
    assert processed_features.shape[1] <= 3

def test_generate_confidence_scores_with_nans(full_market_data):
    """
    Tests the pipeline's robustness when input data contains NaNs.
    """
    data = full_market_data
    # Introduce some NaNs at the beginning
    data['price'].iloc[:10] = np.nan
    data['cvd'].iloc[:10] = np.nan

    processed_features = generate_confidence_scores(
        price=data['price'],
        cvd=data['cvd'],
        open_interest=data['open_interest'],
        funding_rate=data['funding_rate'],
        long_liquidations=data['long_liquidations'],
        short_liquidations=data['short_liquidations']
    )

    # The function should still run without errors and return a valid array
    assert isinstance(processed_features, np.ndarray)
    assert not np.isnan(processed_features).any()

def test_generate_confidence_scores_all_nans_input():
    """
    Tests the edge case where input data is entirely NaN, which should result in an empty array.
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

    # If all data is NaN, the fillna logic can't fix it, should return an empty array
    assert isinstance(processed_features, np.ndarray)
    assert processed_features.size == 0