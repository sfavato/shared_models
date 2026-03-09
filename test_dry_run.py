import os
import pandas as pd
import numpy as np
os.environ["PREMIUM_DATA_ACTIVE"] = "False"
from shared_models.confidence_score_engine.main import generate_confidence_scores

df = pd.DataFrame({
    'symbol': ['BTCUSDC'] * 100,
    'close': np.random.rand(100),
    'volume': np.random.rand(100),
})

scores = generate_confidence_scores(df)
print(scores)
