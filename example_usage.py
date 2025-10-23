import pandas as pd
import numpy as np
from shared_models.confidence_score_engine.calculator import ConfidenceScoreCalculator

def generate_mock_price_data(rows: int) -> pd.Series:
    """Generates a mock price series with a discernible trend."""
    # Create a base trend with some noise
    trend = np.linspace(start=100, stop=150, num=rows)
    noise = np.random.normal(loc=0, scale=2, size=rows)
    price_data = trend + noise

    # Create a pandas Series
    price_series = pd.Series(price_data)

    # Introduce a few outliers to make the data more realistic
    for _ in range(int(rows * 0.05)): # 5% outliers
        idx = np.random.randint(0, rows)
        price_series.iloc[idx] *= np.random.uniform(0.9, 1.1)

    return price_series

def main():
    """
    Main function to demonstrate the ConfidenceScoreCalculator usage.
    """
    print("--- Demonstration of ConfidenceScoreCalculator ---")

    # 1. Instantiate the calculator
    # For Phase 1, no db_engine is needed as feature functions are placeholders.
    calculator = ConfidenceScoreCalculator()
    print("✅ Calculator instantiated successfully.")

    # 2. Prepare the necessary data
    symbol = "BTC/USDT"
    timeframe = "1h"

    # The calculator's `calculate` method now requires a pandas Series of prices
    # to compute the momentum score. Let's generate some mock data.
    # A minimum of 20 periods is needed for the z-score calculation.
    print("\nGenerating mock price data for the momentum score...")
    mock_klines = generate_mock_price_data(100)
    print(f"Generated a mock price series with {len(mock_klines)} data points.")

    # 3. Call the calculate method
    print(f"\nCalculating confidence score for {symbol} on timeframe {timeframe}...")
    final_score = calculator.calculate(
        symbol=symbol,
        timeframe=timeframe,
        price_series=mock_klines
    )

    # 4. Print the result
    print("\n--- CALCULATION COMPLETE ---")
    print(f"➡️ The final calculated confidence score is: {final_score:.2f} / 10")

    # Explain the components of the score
    print("\n--- Score Breakdown (for demonstration) ---")

    # Recalculate individual components to show the logic
    from shared_models.confidence_score_engine.calculator import calculate_zscore_momentum
    from shared_models.confidence_score_engine.features import calculate_market_regime_features, calculate_onchain_features

    momentum_score = calculate_zscore_momentum(mock_klines)
    regime_score = calculate_market_regime_features(None, symbol, timeframe)['regime_score']
    onchain_score = calculate_onchain_features(None, symbol, timeframe)['cvd_divergence_score']

    weights = calculator.weights
    weighted_momentum = momentum_score * weights['momentum_zscore']
    weighted_regime = regime_score * weights['market_regime']
    weighted_onchain = onchain_score * weights['on_chain_score']
    total_weighted_score = weighted_momentum + weighted_regime + weighted_onchain

    print(f"Momentum Z-Score: {momentum_score:.4f} (Weight: {weights['momentum_zscore']}) -> Weighted: {weighted_momentum:.4f}")
    print(f"Market Regime Score: {regime_score} (Weight: {weights['market_regime']}) -> Weighted: {weighted_regime:.4f}")
    print(f"On-Chain Score: {onchain_score} (Weight: {weights['on_chain_score']}) -> Weighted: {weighted_onchain:.4f}")
    print(f"Total Weighted Score (before normalization): {total_weighted_score:.4f}")

    normalized_score_check = calculator._normalize_score(total_weighted_score)
    print(f"Final Score (after normalization from -1 to 1 range): {normalized_score_check:.2f} / 10")

if __name__ == "__main__":
    main()
