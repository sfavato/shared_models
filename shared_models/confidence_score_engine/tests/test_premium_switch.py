import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
from shared_models.confidence_score_engine.main import generate_confidence_scores
from shared_models.confidence_score_engine.calculator import DynamicScoreCalculator

@patch.dict(os.environ, {"PREMIUM_DATA_ACTIVE": "False"})
def test_generate_confidence_scores_without_premium_data():
    """
    Test that generate_confidence_scores works when PREMIUM_DATA_ACTIVE is False
    and premium columns (CVD, open_interest, funding_rate) are missing.
    """
    # Create a dummy DataFrame with only basic data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "close": np.random.rand(100) * 100,
        "volume": np.random.rand(100) * 1000,
        "high": np.random.rand(100) * 100,
        "low": np.random.rand(100) * 100,
        "open": np.random.rand(100) * 100,
    }, index=dates)

    # Currently, without the fix, this should return an empty array because of missing required cols
    scores = generate_confidence_scores(df)

    # We expect this to FAIL (return empty) before the fix, but PASS (return scores) after the fix.
    # For reproduction, we can assert that it IS empty to confirm current behavior,
    # or assert that it is NOT empty to drive the TDD process.
    # I'll assert it is NOT empty, so the test fails now.
    assert scores.size > 0, "Should return scores even without premium data when PREMIUM_DATA_ACTIVE is False"

@patch.dict(os.environ, {"PREMIUM_DATA_ACTIVE": "False"})
def test_calculator_weights_premium_inactive():
    """
    Test that DynamicScoreCalculator adjusts weights when PREMIUM_DATA_ACTIVE is False.
    """
    # We want to check if the calculation logic changes.
    # Standard weights: ML=0.4, Purity=0.2, Regime=0.25, History=0.15
    # Expected non-premium weights (from plan): ML=0.0, Purity=0.8, Regime=0.0, History=0.2

    ml_proba = 1.0
    purity_score = 1.0
    market_regime_score = 1.0
    win_rate_score = 1.0

    # Calculate score
    score = DynamicScoreCalculator.calculate_dynamic_score(
        ml_proba=ml_proba,
        purity_score=purity_score,
        market_regime_score=market_regime_score,
        win_rate_score=win_rate_score
    )

    # With non-premium weights, if we pass 1.0 for everything, it should still be 10.0.
    # So we need to vary inputs to detect weight changes.

    # Let's say we have perfect purity (1.0) and history (1.0), but 0 ML and 0 Regime.
    # Standard: (0*0.4) + (1*0.2) + (0*0.25) + (1*0.15) = 0.35 -> 3.5
    # Non-Premium: (0*0.0) + (1*0.8) + (0*0.0) + (1*0.2) = 1.0 -> 10.0

    score_mixed = DynamicScoreCalculator.calculate_dynamic_score(
        ml_proba=0.0,
        purity_score=1.0,
        market_regime_score=0.0,
        win_rate_score=1.0
    )

    assert score_mixed == 10.0, f"Score should be 10.0 with pure technicals/history in non-premium mode, got {score_mixed}"

@patch.dict(os.environ, {"PREMIUM_DATA_ACTIVE": "True"})
def test_calculator_weights_premium_active():
    """
    Test that DynamicScoreCalculator uses standard weights when PREMIUM_DATA_ACTIVE is True.
    """
    # Standard: (0*0.4) + (1*0.2) + (0*0.25) + (1*0.15) = 0.35 -> 3.5
    score_mixed = DynamicScoreCalculator.calculate_dynamic_score(
        ml_proba=0.0,
        purity_score=1.0,
        market_regime_score=0.0,
        win_rate_score=1.0
    )

    assert score_mixed == 3.5, f"Score should be 3.5 with standard weights, got {score_mixed}"
