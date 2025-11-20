# Confidence Score Engine - Documentation Technique

Ce module est le cœur décisionnel pour l'évaluation de la qualité des trades. Il combine l'analyse technique classique (Géométrie de marché) avec des modèles statistiques avancés (Machine Learning) et des données fondamentales (On-Chain, Sentiment).

## Architecture

Le calcul du score de confiance repose sur 4 piliers pondérés (`DynamicScoreCalculator`) :

1.  **Modèle ML (40%)** : Probabilité issue d'un classifieur entraîné sur l'historique.
2.  **Régime de Marché (25%)** : Analyse du contexte (Tendance vs Range, Volatilité).
3.  **Pureté Géométrique (20%)** : Précision des ratios de Fibonacci du pattern.
4.  **Historique (15%)** : Taux de réussite passé du pattern spécifique.

## Features Sélectionnées (Phase I)

Les features suivantes ont été retenues pour leur capacité prédictive et leur robustesse :

*   **`divergence_score`** : Détecte les divergences Prix vs CVD (Cumulative Volume Delta). Indique un épuisement de tendance ou une absorption.
*   **`oi_funding_momentum`** : Combine le taux de financement et le momentum de l'Open Interest. Identifie les situations de "crowding" (marché surpeuplé) propices aux liquidations.
*   **`trapped_trader_score`** : Corrélation entre pics de liquidations et mouvements de prix contraires. Détecte les traders piégés.
*   **`mvrv_score`** : (On-Chain) Valuation du marché (Surchauffe vs Capitulation).
*   **`netflow_score`** : (On-Chain) Flux nets des exchanges (Accumulation vs Distribution).
*   **`whale_accumulation_score`** : (On-Chain) Activité des gros portefeuilles.

## Pipeline de Prétraitement (.pkl)

Le fichier `.pkl` (`confidence_model.pkl` ou `preprocessor.pkl`) contient l'état "gelé" du pipeline de transformation des données. Il est crucial pour garantir que les données de production subissent *exactement* le même traitement que les données d'entraînement.

### Étapes du Pipeline

1.  **DtypeCoercion** : Force toutes les entrées en `float64`.
2.  **QuantileTransformer** (Gaussian Rank) :
    *   Force les données à suivre une distribution normale.
    *   Réduit l'impact des outliers extrêmes (ex: un crash flash, ou des données on-chain aberrantes).
3.  **PCA (Principal Component Analysis)** :
    *   Réduit la dimensionnalité en ne gardant que les composantes expliquant 95% de la variance.
    *   Décorrèle les features (orthogonalisation).

### Procédure de Régénération

Si de nouvelles features sont ajoutées ou si la dynamique de marché change radicalement, le pipeline doit être réentraîné.

1.  Collecter un nouveau dataset historique large via `backtest_engine`.
2.  Instancier un nouveau `PreprocessingPipeline`.
3.  Appeler `pipeline.fit(new_data)`.
4.  Sauvegarder via `pipeline.save('path/to/new_pipeline.pkl')`.

**Attention :** Ne jamais utiliser `fit()` en production (inférence), uniquement `transform()`.
