import pytest
import pandas as pd
from unittest.mock import patch
from shared_models.confidence_score_engine.calculator import calculate_zscore_momentum, ConfidenceScoreCalculator

def test_calculate_zscore_momentum():
    series = pd.Series([100, 102, 105, 103, 106, 108, 110, 112, 115, 113, 116, 118, 120, 122, 125, 123, 126, 128, 130, 132])
    z_score = calculate_zscore_momentum(series)
    assert isinstance(z_score, float)

@patch('shared_models.confidence_score_engine.calculator.calculate_zscore_momentum')
@patch('shared_models.confidence_score_engine.calculator.calculate_market_regime_features')
@patch('shared_models.confidence_score_engine.calculator.calculate_onchain_features')
def test_confidence_score_calculator(mock_onchain, mock_regime, mock_momentum):
    mock_momentum.return_value = 0.5
    mock_regime.return_value = {'regime_score': 0.8}
    mock_onchain.return_value = {'cvd_divergence_score': -0.4}

    calculator = ConfidenceScoreCalculator()
    score = calculator.calculate(symbol="BTCUSDT", timeframe="1h")

    expected_weighted_score = (0.5 * 0.4) + (0.8 * 0.3) + (-0.4 * 0.3)
    expected_normalized_score = calculator._normalize_score(expected_weighted_score)

    assert score == expected_normalized_score

def test_normalize_score():
    calculator = ConfidenceScoreCalculator()

    assert calculator._normalize_score(0) == 5.0
    assert calculator._normalize_score(1) == 10.0
    assert calculator._normalize_score(-1) == 0.0
    assert calculator._normalize_score(0.5) == 7.5
    assert calculator._normalize_score(-0.5) == 2.5