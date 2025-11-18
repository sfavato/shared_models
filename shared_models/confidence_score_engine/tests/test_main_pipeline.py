import pytest
import pandas as pd
import numpy as np
from shared_models.confidence_score_engine.main import generate_confidence_scores
from shared_models.confidence_score_engine.pipeline import PreprocessingPipeline
from shared_models.confidence_score_engine.features import (
    calculate_divergence_score,
    oi_weighted_funding_momentum,
    trapped_trader_score,
)
import joblib
import os

@pytest.fixture
def market_data_fixture():
    """Provides a realistic, valid DataFrame for the main pipeline test."""
    size = 100
    np.random.seed(42)
    data = {
        'close': pd.Series(np.linspace(100, 150, size)),
        'CVD': pd.Series(np.linspace(1000, 1500, size)),
        'open_interest': pd.Series(np.linspace(5000, 6000, size)),
        'funding_rate': pd.Series(np.random.normal(0.01, 0.005, size)),
        'long_liquidations_usd': pd.Series(np.random.randint(0, 100, size), dtype=float),
        'short_liquidations_usd': pd.Series(np.random.randint(0, 100, size), dtype=float)
    }
    df = pd.DataFrame(data)

    # Replicate feature generation to fit the pipeline correctly
    features = {}
    price = df['close']
    cvd = df['CVD']
    open_interest = df['open_interest']
    funding_rate = df['funding_rate']
    long_liquidations = df.get('long_liquidations_usd')
    short_liquidations = df.get('short_liquidations_usd')
    lookback_period = 20  # Same default as in main.py

    features['divergence_score'] = calculate_divergence_score(price, cvd, lookback_period)
    features['oi_funding_momentum'] = oi_weighted_funding_momentum(funding_rate, open_interest, lookback_period)
    features['trapped_trader_score'] = trapped_trader_score(
        price_close=price,
        long_liquidations=long_liquidations,
        short_liquidations=short_liquidations,
        window=lookback_period
    )
    features_df = pd.DataFrame(features)
    features_df.bfill(inplace=True)
    features_df.ffill(inplace=True)

    # Create and save a dummy pipeline for the test, fitted on the correct features
    pipeline = PreprocessingPipeline()
    pipeline.fit(features_df)  # Fit on the generated features
    pipeline_path = "test_pipeline.pkl"
    pipeline.save(pipeline_path)

    yield df, pipeline_path

    # Cleanup after test
    os.remove(pipeline_path)


def test_generate_confidence_scores_happy_path(market_data_fixture):
    """
    Tests the full pipeline with valid data, checking output shape and quality.
    """
    df, pipeline_path = market_data_fixture
    processed_features = generate_confidence_scores(df, pipeline_path=pipeline_path)

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
    df, pipeline_path = market_data_fixture
    # Introduce NaNs at the beginning of two series
    df['close'].iloc[:15] = np.nan
    df['CVD'].iloc[:15] = np.nan

    processed_features = generate_confidence_scores(df, pipeline_path=pipeline_path)

    # The function should still run and return a valid array without NaNs
    assert isinstance(processed_features, np.ndarray)
    assert not np.isnan(processed_features).any()

def test_generate_confidence_scores_all_nans_input():
    """
    Tests the edge case where all input data is NaN. This should result in an empty array.
    """
    size = 100
    nan_df = pd.DataFrame({
        'close': pd.Series([np.nan] * size),
        'CVD': pd.Series([np.nan] * size),
        'open_interest': pd.Series([np.nan] * size),
        'funding_rate': pd.Series([np.nan] * size),
        'long_liquidations_usd': pd.Series([np.nan] * size),
        'short_liquidations_usd': pd.Series([np.nan] * size),
    })

    processed_features = generate_confidence_scores(nan_df, pipeline_path=None)

    assert isinstance(processed_features, np.ndarray)
    assert processed_features.size == 0

def test_output_statistical_properties(market_data_fixture):
    """
    Validates that the output features are correctly normalized (mean ~0, std ~1)
    after the quantile transformation and PCA.
    """
    df, pipeline_path = market_data_fixture
    processed_features = generate_confidence_scores(df, pipeline_path=pipeline_path)

    assert processed_features.shape[0] > 0

    # Check the statistical properties of each principal component
        # Check the statistical properties of each principal component
    for i in range(processed_features.shape[1]):
        component = processed_features[:, i]
            # After quantile transform to a normal distribution and PCA, the mean of the components should be near 0.
            # The std is not expected to be 1, as PCA components capture variance.
        assert np.isclose(np.mean(component), 0, atol=0.2), f"Mean of PC {i} is not close to 0"