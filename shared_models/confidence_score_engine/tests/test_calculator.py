import pandas as pd
import numpy as np
import pytest
from shared_models.confidence_score_engine.calculator import calculate_derivatives_score

# Helper function to create test data
def create_test_data(oi_trend, fr_trend):
    data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=30, freq='h')),
        'open_interest': np.linspace(1000, 1000 + oi_trend, 30),
        'funding_rate': np.linspace(0.001, 0.001 + fr_trend, 30),
        'cvd': np.random.rand(30)
    }
    return pd.DataFrame(data)

def test_empty_dataframe():
    """Test that an empty DataFrame returns a score of 0."""
    df = pd.DataFrame()
    assert calculate_derivatives_score(df) == 0

def test_not_enough_data():
    """Test that a DataFrame with insufficient data for a Z-score returns 0."""
    df = create_test_data(100, 0.001)[:10]  # Less than the 24-period window
    assert calculate_derivatives_score(df) == 0

def test_strong_bullish_signal():
    """Test for a strong bullish signal: high OI z-score and very negative funding rate z-score."""
    # To get a high OI z-score, we need a sharp increase at the end
    oi_data = np.linspace(1000, 1100, 30)
    oi_data[-1] = 1500 # Sharp increase to get z-score > 1

    # To get a very negative FR z-score, we need a sharp decrease at the end
    fr_data = np.linspace(0.001, 0.00, 30)
    fr_data[-1] = -0.05 # Sharp decrease to get z-score < -1.5

    data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=30, freq='h')),
        'open_interest': oi_data,
        'funding_rate': fr_data,
    }
    df = pd.DataFrame(data)
    # Expected score: +1 (OI) + 2 (FR) = 3
    assert calculate_derivatives_score(df) == 3

def test_strong_bearish_signal():
    """Test for a strong bearish signal: very positive funding rate z-score."""
    # To get a very positive FR z-score, we need a sharp increase at the end
    fr_data = np.linspace(0.001, 0.002, 30)
    fr_data[-1] = 0.05 # Sharp increase to get z-score > 1.5

    data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=30, freq='h')),
        'open_interest': np.linspace(1000, 1000, 30), # Truly stable OI
        'funding_rate': fr_data,
    }
    df = pd.DataFrame(data)
    # Expected score: -2 (FR)
    assert calculate_derivatives_score(df) == -2

def test_bullish_oi_only():
    """Test for a bullish signal from Open Interest only."""
    oi_data = np.linspace(1000, 1100, 30)
    oi_data[-1] = 1500 # Sharp increase

    data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=30, freq='h')),
        'open_interest': oi_data,
        'funding_rate': np.linspace(0.001, 0.001, 30), # Stable funding
    }
    df = pd.DataFrame(data)
    # Expected score: +1 (OI)
    assert calculate_derivatives_score(df) == 1

def test_neutral_signal():
    """Test for a neutral signal where no thresholds are met."""
    data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=30, freq='h')),
        'open_interest': np.linspace(1000, 1000, 30), # Truly stable OI
        'funding_rate': np.linspace(0.001, 0.001, 30), # Stable funding
    }
    df = pd.DataFrame(data)
    assert calculate_derivatives_score(df) == 0