import pytest
import pandas as pd
import numpy as np
import time
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


def test_output_statistical_properties_and_performance():
    """
    Valide les propriétés statistiques des features en sortie et mesure la performance.
    Ce test simule une charge de travail réaliste.
    """
    # ÉTAPE 1: Créer un jeu de données réaliste et de taille significative.
    num_rows = 10000  # Simule un historique de données conséquent
    lookback = 20
    data = {
        'price': np.random.randn(num_rows).cumsum() + 100,
        'cvd': np.random.randn(num_rows).cumsum(),
        'open_interest': np.random.randn(num_rows).cumsum() + 1000,
        'funding_rate': np.random.randn(num_rows) * 0.0001,
        'long_liquidations': np.random.randint(0, 100, size=num_rows),
        'short_liquidations': np.random.randint(0, 100, size=num_rows)
    }
    df = pd.DataFrame(data)

    # ÉTAPE 2: Mesurer le temps d'exécution.
    start_time = time.time()

    processed_features = generate_confidence_scores(
        price=df['price'],
        cvd=df['cvd'],
        open_interest=df['open_interest'],
        funding_rate=df['funding_rate'],
        long_liquidations=df['long_liquidations'],
        short_liquidations=df['short_liquidations'],
        lookback_period=lookback
    )

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time for {num_rows} rows: {execution_time:.4f} seconds")

    # ÉTAPE 3: Valider la performance.
    # Le calcul pour 10k lignes doit être très rapide.
    assert execution_time < 1.0  # Fails if execution takes more than 1 second.

    # ÉTAPE 4: Valider les propriétés statistiques des données en sortie.
    # Après le QuantileTransformer(output_distribution='normal'),
    # la moyenne de chaque composante doit être proche de 0 et l'écart-type proche de 1.

    # Le DataFrame initial peut contenir des NaNs au début à cause des rollings.
    # La fonction `generate_confidence_scores` les gère, mais la sortie peut être plus courte.
    assert processed_features.shape[0] > 0

    # On vérifie la moyenne et l'écart-type pour chaque composante principale (colonne).
    for i in range(processed_features.shape[1]):
        column = processed_features[:, i]
        assert np.isclose(np.mean(column), 0, atol=0.2), f"Mean of component {i} is not close to 0"
        assert np.isclose(np.std(column), 1, atol=0.2), f"Std dev of component {i} is not close to 1"