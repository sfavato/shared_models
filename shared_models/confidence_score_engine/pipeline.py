from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Définir les étapes du pipeline de prétraitement
preprocessing_pipeline = Pipeline([
    (
        'quantile_transformer',
        QuantileTransformer(output_distribution='normal', random_state=42)
    ),
    (
        'pca',
        PCA(n_components=0.95) # Conserve 95% de la variance expliquée par les composantes.
    )
])

"""
NOTES SUR LE PIPELINE :
1. QuantileTransformer : Cette étape normalise les données en se basant sur leur rang (quantile),
   ce qui les rend robustes aux valeurs extrêmes ('outliers') fréquentes dans les données financières.
   'output_distribution="normal"' transforme la distribution pour qu'elle suive une loi normale.

2. PCA (Analyse en Composantes Principales) : Cette étape réduit la dimensionnalité et dé-corrèle
   les caractéristiques. En conservant 95% de la variance, on s'assure de garder l'essentiel de
   l'information tout en éliminant le bruit et la redondance entre les facteurs.
"""