import pandas as pd
import numpy as np
import pytest

from shared_models.confidence_score_engine.features import (
    calculate_mvrv_score,
    calculate_exchange_netflow_score,
    calculate_whale_accumulation_score,
)

@pytest.fixture
def sample_data():
    """Fixture to create sample data for testing."""
    data = {
        'mvrv_usd': [0.8, 1.5, 3.5, np.nan, 0.0],
        'exchange_netflow_usd': [-1000, 500, 2000, np.nan, 0.0],
        'whale_accumulation_delta': [100, -50, 0, np.nan, 200],
    }
    return pd.DataFrame(data)

def test_calculate_mvrv_score(sample_data):
    """Test the MVRV score calculation."""
    mvrv_series = sample_data['mvrv_usd']
    scores = calculate_mvrv_score(mvrv_series)

    assert scores.iloc[0] == 0.9  # MVRV < 1.0 -> Bullish
    assert scores.iloc[1] == 0.5  # 1.0 < MVRV < 3.0 -> Neutral
    assert scores.iloc[2] == 0.1  # MVRV > 3.0 -> Bearish
    assert scores.iloc[3] == 0.5  # NaN -> Neutral
    assert scores.iloc[4] == 0.5  # 0.0 -> Neutral
    assert scores.notna().all()

def test_calculate_exchange_netflow_score(sample_data):
    """Test the exchange netflow score calculation."""
    netflow_series = sample_data['exchange_netflow_usd']
    scores = calculate_exchange_netflow_score(netflow_series)

    assert scores.iloc[0] > 0.5  # Negative netflow (outflow) -> Bullish
    assert scores.iloc[2] < 0.5  # Positive netflow (inflow) -> Bearish
    assert scores.iloc[3] == 0.5  # NaN -> Neutral
    assert scores.iloc[4] == 0.5  # 0.0 -> Neutral
    assert scores.notna().all()

def test_calculate_whale_accumulation_score(sample_data):
    """Test the whale accumulation score calculation."""
    whale_series = sample_data['whale_accumulation_delta']
    scores = calculate_whale_accumulation_score(whale_series)

    assert scores.iloc[0] > 0.5  # Positive delta -> Bullish
    assert scores.iloc[1] < 0.5  # Negative delta -> Bearish
    assert scores.iloc[2] == 0.5  # 0.0 -> Neutral
    assert scores.iloc[3] == 0.5  # NaN -> Neutral
    assert scores.iloc[4] > 0.5  # Positive delta -> Bullish
    assert scores.notna().all()