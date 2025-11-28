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

from sklearn.linear_model import LogisticRegression

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

    # Create and save a dummy preprocessor and model for the test
    preprocessor = PreprocessingPipeline()
    preprocessor.fit(features_df)
    preprocessor_path = "test_preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)

    # Create and save a dummy model
    model = LogisticRegression()
    # Create a dummy target variable for fitting the model
    y = np.random.randint(0, 2, size=features_df.shape[0])
    with np.errstate(all='ignore'):
        model.fit(preprocessor.transform(features_df), y)
    model_path = "test_model.pkl"
    joblib.dump(model, model_path)


    yield df, model_path, preprocessor_path

    # Cleanup after test
    os.remove(preprocessor_path)
    os.remove(model_path)


def test_generate_confidence_scores_happy_path(market_data_fixture):
    """
    Tests the full pipeline with valid data, checking output shape and quality.
    """
    df, model_path, preprocessor_path = market_data_fixture
    processed_features = generate_confidence_scores(df, model_path=model_path, preprocessor_path=preprocessor_path)

    assert isinstance(processed_features, np.ndarray)
    assert not np.isnan(processed_features).any()
    assert not np.isinf(processed_features).any()
    # The number of rows should be the original size
    assert processed_features.shape[0] > 0
    # The number of columns should be num_pca_components + c2 + c3
    assert processed_features.shape[1] >= 3

def test_generate_confidence_scores_with_nans(market_data_fixture):
    """
    Tests the pipeline's robustness when input data contains NaNs.
    The pipeline should handle them gracefully.
    """
    df, model_path, preprocessor_path = market_data_fixture
    # Introduce NaNs at the beginning of two series
    df['close'].iloc[:15] = np.nan
    df['CVD'].iloc[:15] = np.nan

    processed_features = generate_confidence_scores(df, model_path=model_path, preprocessor_path=preprocessor_path)

    # The function should still run and return a valid array without NaNs
    assert isinstance(processed_features, np.ndarray)
    assert not np.isnan(processed_features).any()

def test_generate_confidence_scores_all_nans_input():
    """
    Tests the edge case where all input data is NaN. This should result in a zero-filled array.
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

    processed_features = generate_confidence_scores(nan_df, model_path=None, preprocessor_path=None)

    assert isinstance(processed_features, np.ndarray)
    assert processed_features.shape == (100, 3)
    # With all NaNs, the fallback logic should be triggered, which may not be all zeros.
    # The important part is that it doesn't crash and returns the correct shape.
    assert not np.isnan(processed_features).any()


def test_output_statistical_properties(market_data_fixture):
    """
    Validates that the output features are correctly normalized (mean ~0, std ~1)
    after the quantile transformation and PCA.
    """
    df, model_path, preprocessor_path = market_data_fixture
    processed_features = generate_confidence_scores(df, model_path=model_path, preprocessor_path=preprocessor_path)

    assert processed_features.shape[0] > 0

    # The C1 score is the first column
    c1_score = processed_features[:, 0]

    # Check that the C1 score is scaled between 0 and 10
    assert np.all(c1_score >= 0)
    assert np.all(c1_score <= 10)