import pytest

# Import the class to be tested
from shared_models.confidence_score_engine.calculator import DynamicScoreCalculator

def test_calculate_dynamic_score_all_high():
    """Test the score calculation with high input values."""
    score = DynamicScoreCalculator.calculate_dynamic_score(
        ml_proba=0.9,
        purity_score=0.8,
        market_regime_score=0.85,
        win_rate_score=0.95
    )
    # Expected: (0.9*0.4) + (0.8*0.2) + (0.85*0.25) + (0.95*0.15) = 0.36 + 0.16 + 0.2125 + 0.1425 = 0.875
    # Scaled: 0.875 * 10 = 8.75, Rounded: 8.8
    assert score == pytest.approx(8.8)

def test_calculate_dynamic_score_all_low():
    """Test the score calculation with low input values."""
    score = DynamicScoreCalculator.calculate_dynamic_score(
        ml_proba=0.2,
        purity_score=0.3,
        market_regime_score=0.25,
        win_rate_score=0.1
    )
    # Expected: (0.2*0.4) + (0.3*0.2) + (0.25*0.25) + (0.1*0.15) = 0.08 + 0.06 + 0.0625 + 0.015 = 0.2175
    # Scaled: 0.2175 * 10 = 2.175, Rounded: 2.2
    assert score == pytest.approx(2.2)

def test_calculate_dynamic_score_mixed_values():
    """Test the score calculation with a mix of input values."""
    score = DynamicScoreCalculator.calculate_dynamic_score(
        ml_proba=0.75,
        purity_score=0.5,
        market_regime_score=0.9,
        win_rate_score=0.4
    )
    # Expected: (0.75*0.4) + (0.5*0.2) + (0.9*0.25) + (0.4*0.15) = 0.3 + 0.1 + 0.225 + 0.06 = 0.685
    # Scaled: 0.685 * 10 = 6.85, Rounded: 6.8 (round half to even)
    assert score == pytest.approx(6.8)

def test_calculate_dynamic_score_zero_values():
    """Test the score calculation when all inputs are zero."""
    score = DynamicScoreCalculator.calculate_dynamic_score(
        ml_proba=0.0,
        purity_score=0.0,
        market_regime_score=0.0,
        win_rate_score=0.0
    )
    assert score == 0.0

def test_calculate_dynamic_score_perfect_score():
    """Test the score calculation when all inputs are 1.0."""
    score = DynamicScoreCalculator.calculate_dynamic_score(
        ml_proba=1.0,
        purity_score=1.0,
        market_regime_score=1.0,
        win_rate_score=1.0
    )
    # Expected: (1.0*0.4) + (1.0*0.2) + (1.0*0.25) + (1.0*0.15) = 1.0
    # Scaled: 1.0 * 10 = 10.0
    assert score == 10.0

def test_legacy_alias_import():
    """Vérifie que l'alias ConfidenceScoreCalculator est importable et fonctionnel."""
    try:
        from shared_models.confidence_score_engine.calculator import ConfidenceScoreCalculator
        instance = ConfidenceScoreCalculator()
        assert isinstance(instance, DynamicScoreCalculator)
    except ImportError:
        pytest.fail("ImportError: Impossible d'importer ConfidenceScoreCalculator.")

def test_calculate_score_wrapper_full_dict():
    """Teste la méthode wrapper calculate_score avec un dictionnaire complet."""
    calculator = DynamicScoreCalculator()
    features = {
        'ml_proba': 0.75,
        'purity': 0.5,
        'regime': 0.9,
        'win_rate': 0.4
    }
    score = calculator.calculate_score(features)
    # Le calcul doit être identique à test_calculate_dynamic_score_mixed_values
    assert score == pytest.approx(6.8)

def test_calculate_score_wrapper_partial_dict():
    """Teste la méthode wrapper avec un dictionnaire partiel (ml_proba manquant)."""
    calculator = DynamicScoreCalculator()
    features = {
        'purity': 0.8,
        'regime': 0.85,
        'win_rate': 0.95
    }
    # ml_proba doit utiliser la valeur par défaut de 0.5
    # Expected: (0.5*0.4) + (0.8*0.2) + (0.85*0.25) + (0.95*0.15) = 0.2 + 0.16 + 0.2125 + 0.1425 = 0.715
    # Scaled: 0.715 * 10 = 7.15, Rounded: 7.2
    score = calculator.calculate_score(features)
    assert score == pytest.approx(7.2)

def test_calculate_score_wrapper_only_ml_proba():
    """Teste le wrapper avec seulement ml_proba présent."""
    calculator = DynamicScoreCalculator()
    features = {'ml_proba': 0.9}
    # Les autres valeurs doivent utiliser la valeur par défaut de 0.5
    # Expected: (0.9*0.4) + (0.5*0.2) + (0.5*0.25) + (0.5*0.15) = 0.36 + 0.1 + 0.125 + 0.075 = 0.66
    # Scaled: 0.66 * 10 = 6.6
    score = calculator.calculate_score(features)
    assert score == pytest.approx(6.6)
