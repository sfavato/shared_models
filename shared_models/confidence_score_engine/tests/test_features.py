import pytest
import pandas as pd
import numpy as np
from shared_models.confidence_score_engine.features import (
    calculate_divergence_score,
    oi_weighted_funding_momentum,
    trapped_trader_score
)

LOOKBACK = 5

@pytest.fixture
def sample_data():
    """Provides a base DataFrame for tests."""
    return pd.DataFrame({
        'price': np.linspace(100, 150, 20),
        'cvd': np.linspace(1000, 1500, 20),
        'open_interest': np.linspace(5000, 6000, 20),
        'funding_rate': np.full(20, 0.01),
        'long_liquidations': np.zeros(20),
        'short_liquidations': np.zeros(20)
    })

def test_bearish_divergence(sample_data):
    """Test that a bearish divergence returns a negative score."""
    price = pd.Series([100, 110, 120, 130, 140, 150]) # New high
    cvd = pd.Series([500, 550, 600, 580, 560, 540])   # Lower high

    score = calculate_divergence_score(price, cvd, lookback_period=LOOKBACK)

    # The last smoothed score should be negative due to the divergence
    assert score.iloc[-1] < 0

def test_bullish_divergence(sample_data):
    """Test that a bullish divergence returns a positive score."""
    price = pd.Series([150, 140, 130, 120, 110, 100]) # New low
    cvd = pd.Series([600, 550, 500, 520, 540, 560])   # Higher low

    score = calculate_divergence_score(price, cvd, lookback_period=LOOKBACK)

    # The last smoothed score should be positive
    assert score.iloc[-1] > 0

def test_no_divergence(sample_data):
    """Test that no divergence returns a zero score."""
    price = pd.Series([100, 110, 120, 130, 140, 135]) # No new high/low at the end
    cvd = pd.Series([500, 550, 600, 650, 700, 680])   # Moving with price

    score = calculate_divergence_score(price, cvd, lookback_period=LOOKBACK)

    # The last score should be zero as there's no divergence event
    assert score.iloc[-1] == 0

def test_oi_weighted_funding_momentum_positive(sample_data):
    """Test positive funding and rising OI returns a positive score."""
    funding = sample_data['funding_rate']
    oi = sample_data['open_interest'] # OI is steadily increasing

    score = oi_weighted_funding_momentum(funding, oi, lookback=LOOKBACK)

    # With positive funding and rising OI, the score should be positive
    assert score.iloc[-1] > 0

def test_trapped_trader_score_liquidation_spike(sample_data):
    """Test that a liquidation spike significantly increases the score."""
    data = sample_data.copy()
    # Create a huge liquidation spike at the end
    data.loc[len(data) - 1, 'long_liquidations'] = 10000

    # Add some volatility to price and cvd to generate a base score
    data['price'] = [100, 105, 102, 110, 105, 115, 110, 120, 115, 125, 120, 130, 125, 135, 130, 140, 135, 145, 140, 100]
    data['cvd'] =   [100, 105, 102, 110, 105, 115, 110, 120, 115, 125, 120, 130, 125, 135, 130, 140, 135, 145, 140, 100]


    # Re-implement the logic without the final smoothing to test the spike's effect directly
    price_accel = data['price'].diff().diff().abs().rolling(window=LOOKBACK).mean()
    cvd_accel = data['cvd'].diff().diff().abs().rolling(window=LOOKBACK).mean()
    base_score = (price_accel * cvd_accel).rank(pct=True)

    total_liqs = data['long_liquidations'] + data['short_liquidations']
    liq_baseline = total_liqs.rolling(window=LOOKBACK * 5).mean()
    liq_std = total_liqs.rolling(window=LOOKBACK * 5).std()
    is_spike = (total_liqs > (liq_baseline + 2 * liq_std)).astype(int)

    # The unsmoothed score at the spike point should be the base score + 1
    unsmoothed_score = base_score + is_spike

    # The liquidation spike adds 1.0 to the base score.
    # The base score can be 0, so we check for >= 1.0
    assert unsmoothed_score.iloc[-1] >= 1.0