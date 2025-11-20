import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any, Union

def calculate_divergence_score(price: pd.Series, cvd: pd.Series, lookback_period: int) -> pd.Series:
    """
    Calcule un score basé sur les divergences entre le prix et le CVD (Cumulative Volume Delta).

    L'objectif est de détecter les épuisements de tendance ou les absorptions cachées.
    Une divergence indique que l'agressivité des acheteurs/vendeurs (CVD) ne valide pas
    le mouvement du prix, suggérant un retournement imminent.

    Une divergence haussière (+1) est détectée si le prix atteint un nouveau plus bas
    sur la période de référence alors que le CVD refuse de faire un plus bas (absorption des ventes).
    Une divergence baissière (-1) est détectée si le prix atteint un nouveau plus haut
    alors que le CVD échoue à suivre (épuisement des achats).

    Args:
        price (pd.Series): Série temporelle des prix, utilisée pour identifier les extrêmes du marché.
        cvd (pd.Series): Série temporelle du Cumulative Volume Delta, représentant le flux d'ordres agressif.
        lookback_period (int): La fenêtre glissante (en nombre de périodes) pour identifier les plus hauts/bas locaux.

    Returns:
        pd.Series: Une série temporelle contenant le score de divergence lissé, permettant de filtrer le bruit.
    """
    # ÉTAPE 1: Trouver les extrêmes du prix et du CVD sur la fenêtre glissante pour établir le contexte local.
    price_high = price.rolling(window=lookback_period).max()
    cvd_high = cvd.rolling(window=lookback_period).max()
    price_low = price.rolling(window=lookback_period).min()
    cvd_low = cvd.rolling(window=lookback_period).min()

    # ÉTAPE 2: Identifier les moments où le prix actuel "break" la structure locale (nouveau sommet/creux).
    is_new_price_high = (price == price_high)
    is_new_price_low = (price == price_low)

    # ÉTAPE 3: Définir les conditions de divergence.
    # On compare le CVD actuel aux extrêmes précédents (.shift(1)) pour confirmer l'absence de validation par le volume.
    bearish_divergence = (is_new_price_high) & (cvd < cvd_high.shift(1))
    bullish_divergence = (is_new_price_low) & (cvd > cvd_low.shift(1))

    # ÉTAPE 4: Créer le score brut directionnel.
    # Le score est +1 pour une divergence haussière (signal d'achat), -1 pour une baissière (signal de vente).
    divergence_score = bullish_divergence.astype(int) - bearish_divergence.astype(int)

    # ÉTAPE 5: Lisser le signal pour capturer un état persistant plutôt qu'un pic instantané.
    # Cela stabilise le feature pour le modèle ML.
    smoothed_score = divergence_score.rolling(window=5, min_periods=1).sum()

    return smoothed_score


def oi_weighted_funding_momentum(funding_rate: pd.Series, open_interest: pd.Series, lookback: int) -> pd.Series:
    """
    Crée un facteur de sentiment en pondérant la tendance du taux de financement par le momentum de l'Open Interest (OI).

    Cette métrique vise à identifier les marchés en surchauffe ou sur-endettés.
    Le taux de financement indique le coût du levier (sentiment), tandis que l'OI indique
    la quantité d'argent engagée. La combinaison des deux révèle la conviction du marché.

    Un score positif élevé suggère un fort momentum haussier engagé (crowding long),
    tandis qu'un score négatif élevé suggère un fort momentum baissier (crowding short),
    augmentant le risque de liquidation en cascade (squeeze).

    Args:
        funding_rate (pd.Series): Série temporelle des taux de financement, proxy du sentiment des traders à levier.
        open_interest (pd.Series): Série temporelle de l'Open Interest, proxy de la liquidité et de l'engagement.
        lookback (int): La fenêtre glissante pour calculer la tendance et le momentum.

    Returns:
        pd.Series: Une série temporelle représentant le facteur de sentiment pondéré.
    """
    # ÉTAPE 1: Calculer le momentum de l'Open Interest (taux de changement).
    # On cherche à savoir si de l'argent frais entre ou sort du marché.
    oi_roc = open_interest.pct_change(periods=lookback, fill_method=None)

    # ÉTAPE 2: Calculer la tendance de fond du taux de financement.
    # On lisse le funding rate pour éviter les bruits horaires et capturer le biais structurel.
    funding_ma = funding_rate.rolling(window=lookback).mean()

    # ÉTAPE 3: Calculer le facteur de sentiment.
    # On amplifie le signal du funding rate par la croissance de l'OI.
    # Si le funding est positif et l'OI monte -> Confiance haussière forte.
    sentiment_factor = funding_ma * (1 + oi_roc.fillna(0))

    return sentiment_factor


def trapped_trader_score(price_close: pd.Series, long_liquidations: pd.Series, short_liquidations: pd.Series, window: int = 24) -> pd.Series:
    """
    Calcule un score de "traders piégés" en corrélant les pics de liquidations avec les mouvements de prix contraires.

    L'objectif est de détecter les points d'inflexion où le marché a "puni" un côté (longs ou shorts)
    de manière excessive, créant souvent un déséquilibre offre/demande favorable à un retournement (reversion).

    Args:
        price_close (pd.Series): Série temporelle des prix de clôture.
        long_liquidations (pd.Series): Série temporelle des liquidations 'long'.
        short_liquidations (pd.Series): Série temporelle des liquidations 'short'.
        window (int): La fenêtre glissante pour définir ce qu'est un "pic" statistique (ex: 24h).

    Returns:
        pd.Series: Un score normalisé indiquant l'intensité de la situation de "piège".
    """
    # ÉTAPE 1: Calculer le volume total des liquidations pour évaluer l'activité globale de purge.
    total_liquidations = long_liquidations + short_liquidations

    # ÉTAPE 2: Identifier les pics statistiques de liquidations.
    # On utilise Z-Score ou écart-type (ici moyenne + 2 sigma) pour isoler les événements anormaux.
    liq_mean = total_liquidations.rolling(window=window).mean()
    liq_std = total_liquidations.rolling(window=window).std()
    liquidation_spikes = (total_liquidations > (liq_mean + 2 * liq_std)).astype(int)

    # ÉTAPE 3: Évaluer la direction du mouvement des prix pendant ces pics.
    # Si on liquide des LONGS et que le prix BAISSE -> Longs piégés / Cascade de vente.
    # Si on liquide des SHORTS et que le prix MONTE -> Shorts piégés / Short squeeze.
    price_change = price_close.diff()

    # Score pour les longs piégés : on pondère par le volume liquidé pour donner de l'importance aux gros événements.
    trapped_longs_score = (liquidation_spikes * (price_change < 0) * long_liquidations).rank(pct=True)

    # Score pour les shorts piégés.
    trapped_shorts_score = (liquidation_spikes * (price_change > 0) * short_liquidations).rank(pct=True)

    # ÉTAPE 4: Combiner les scores.
    # On somme les deux pour obtenir un score d'intensité globale d'activité de piège, quelle que soit la direction.
    final_score = (trapped_longs_score + trapped_shorts_score).fillna(0)

    # ÉTAPE 5: Lisser le score final pour éviter que le modèle ne réagisse à des micro-bruits.
    return final_score.rolling(window=3).mean()


def _calculate_ratios_from_points(points: Dict[str, float]) -> Dict[str, float]:
    """
    Calcule les ratios de Fibonacci internes à partir des coordonnées de prix d'un pattern harmonique.

    Cette fonction utilitaire transforme des niveaux de prix absolus en ratios relatifs,
    qui sont la "signature" mathématique d'un pattern harmonique. Cela permet de comparer
    n'importe quel pattern quelle que soit l'échelle de prix.

    Args:
        points (Dict[str, float]): Un dictionnaire de prix pour les points X, A, B, C, D.
                                   Ex: {'X': 100.0, 'A': 110.0, ...}

    Returns:
        Dict[str, float]: Un dictionnaire contenant les ratios clés (B/XA, C/AB, D/BC, D/XA).
    """
    # Extraction des prix
    p_x = points.get('X')
    p_a = points.get('A')
    p_b = points.get('B')
    p_c = points.get('C')
    p_d = points.get('D')

    # Vérification d'intégrité : tous les points sont nécessaires pour la géométrie complète.
    if not all([p_x, p_a, p_b, p_c, p_d]):
        return {}

    # Calcul des longueurs absolues des segments (vagues).
    xa = abs(p_a - p_x)
    ab = abs(p_b - p_a)
    bc = abs(p_c - p_b)
    cd = abs(p_d - p_c)
    ad = abs(p_d - p_a)

    if xa == 0 or ab == 0 or bc == 0:
        return {} # Protection contre la division par zéro (marché plat/erreur de données)

    # Calcul des ratios de retracement et de projection standards.
    ratio_b = ab / xa       # Le point B définit souvent le type de pattern
    ratio_c = bc / ab       # Le point C valide la structure interne
    ratio_d_bc = cd / bc    # Projection de la jambe BC
    ratio_d_xa = ad / xa    # Retracement total ou extension XA

    return {
        'B': ratio_b,
        'C': ratio_c,
        'D': ratio_d_bc,
        'XA': ratio_d_xa
    }


def calculate_geometric_purity_score(pattern_details: Dict[str, Any], pattern_config: Optional[Dict[str, float]] = None) -> float:
    """
    Calcule un score de "pureté" (1-10) quantifiant à quel point un pattern réel ressemble à son modèle théorique idéal.

    Ce score est crucial pour filtrer les "faux positifs" : un pattern peut avoir la bonne forme générale
    mais des ratios horribles. Un score élevé indique une géométrie respectant précisément les ratios de Fibonacci,
    ce qui augmente statistiquement la probabilité de succès du trade.

    Args:
        pattern_details (Dict[str, Any]): Données du pattern détecté (ratios actuels, nom).
        pattern_config (Optional[Dict[str, float]]): Configuration définissant les bornes idéales (Min/Max) pour chaque ratio.

    Returns:
        float: Un score sur 10.0 (10 = perfection géométrique, < 5 = pattern douteux).
    """
    actual_ratios = pattern_details.get('ratios', {})

    # 1. Définir les ratios idéaux basés sur la configuration fournie (Source de Vérité).
    # On calcule le milieu de la plage [Min, Max] pour obtenir la "cible parfaite".
    ideal_ratios = {}

    if pattern_config:
        # Extraction dynamique pour s'adapter à tout type de pattern (Gartley, Shark, etc.) sans codage en dur.
        if 'XBmin' in pattern_config and 'XBmax' in pattern_config:
            ideal_ratios['B'] = (pattern_config['XBmin'] + pattern_config['XBmax']) / 2

        if 'ACmin' in pattern_config and 'ACmax' in pattern_config:
            ideal_ratios['C'] = (pattern_config['ACmin'] + pattern_config['ACmax']) / 2

        if 'XDmin' in pattern_config and 'XDmax' in pattern_config:
            ideal_ratios['D'] = (pattern_config['XDmin'] + pattern_config['XDmax']) / 2

    elif 'name' in pattern_details:
        # Fallback pour rétrocompatibilité : valeurs par défaut si la config n'est pas injectée.
        LEGACY_DEFAULTS = {
            'Gartley': {'B': 0.618, 'C': 0.786, 'D': 0.786},
            'Bat': {'B': 0.5, 'C': 0.886, 'D': 0.886},
            'Butterfly': {'B': 0.786, 'C': 0.886, 'D': 1.272},
            'Crab': {'B': 0.618, 'C': 0.886, 'D': 1.618}
        }
        ideal_ratios = LEGACY_DEFAULTS.get(pattern_details['name'], {})

    if not ideal_ratios:
        return 5.0  # Retourne un score neutre si on ne sait pas quoi évaluer (évite de bloquer le système).

    # 2. Calcul de l'erreur quadratique moyenne (MSE).
    errors = []
    for point, ideal_val in ideal_ratios.items():
        # Normalisation des clés de ratios entre les différents formats de données du système.
        actual = None
        if point == 'B': actual = actual_ratios.get('XB') or actual_ratios.get('B')
        if point == 'C': actual = actual_ratios.get('AC') or actual_ratios.get('C')
        if point == 'D': actual = actual_ratios.get('XD') or actual_ratios.get('D')

        if actual is not None:
            # On utilise le carré de l'erreur pour pénaliser plus fortement les gros écarts (outliers).
            errors.append((float(actual) - float(ideal_val)) ** 2)

    if not errors:
        return 5.0

    mse = sum(errors) / len(errors)

    # 3. Transformation en score d'utilité (1-10).
    # La formule 1 / (1 + 100*mse) crée une courbe de décroissance rapide :
    # - Une erreur quasi-nulle donne un score proche de 1 (x10).
    # - Une petite erreur fait chuter le score rapidement, reflétant l'exigence de précision des harmoniques.
    normalized_score = 1 / (1 + 100 * mse)

    return round(normalized_score * 10, 2)


def get_historical_performance_score(pattern_details: Dict[str, Any], db_connection: Any) -> float:
    """
    Interroge la base de données pour obtenir un score basé sur la performance passée de ce type de setup.

    L'idée est d'appliquer un principe de "Preuve par l'Historique" : si un pattern spécifique (ex: Gartley sur BTC 1h)
    a un taux de victoire de 80% sur les 100 derniers trades, on booste le score de confiance.

    Args:
        pattern_details (Dict[str, Any]): Métadonnées du pattern (nom, symbole, timeframe).
        db_connection (Any): Objet de connexion DB conforme à la DBAPI.

    Returns:
        float: Un ajustement de score (bonus positif ou malus négatif).
    """
    pattern_name = pattern_details.get('name')
    symbol = pattern_details.get('symbol')
    timeframe = pattern_details.get('timeframe')

    if not all([pattern_name, symbol, timeframe, db_connection]):
        return 0.0

    try:
        cursor = db_connection.cursor()
        # On limite à 100 pour avoir une tendance récente mais statistiquement significative.
        query = """
            SELECT win_rate
            FROM trades_log
            WHERE pattern_name = %s AND symbol = %s AND timeframe = %s
            ORDER BY trade_date DESC
            LIMIT 100;
        """
        cursor.execute(query, (pattern_name, symbol, timeframe))
        results = cursor.fetchall()

        if not results:
            return 0.0  # Pas de données = pas d'opinion (neutralité).

        # Calcul du taux de victoire moyen historique.
        win_rates = [row[0] for row in results]
        average_win_rate = sum(win_rates) / len(win_rates)

        # Application d'un barème discret pour influencer le score final.
        if average_win_rate > 0.7:
            return 1.5  # Bonus fort pour excellence historique.
        elif average_win_rate > 0.5:
            return 0.5  # Petit bonus pour avantage statistique.
        elif average_win_rate < 0.4:
            return -1.0 # Malus pour performance médiocre.
        else:
            return 0.0  # Neutre.

    except Exception as e:
        print(f"Database error in get_historical_performance_score: {e}")
        return 0.0


import numpy as np

def calculate_mvrv_score(mvrv_series: pd.Series) -> pd.Series:
    """
    Calcule un score de valuation basé sur le ratio MVRV (Market Value to Realized Value).

    Le MVRV permet d'identifier si l'actif est surévalué (sommet de cycle) ou sous-évalué (fond de cycle)
    par rapport au prix moyen d'achat des participants.

    Args:
        mvrv_series (pd.Series): Série temporelle des valeurs MVRV.

    Returns:
        pd.Series: Un score normalisé (0.0 - 1.0) indiquant la favorabilité de l'entrée (bas = risqué, haut = opportunité).
    """
    # Remplacer les valeurs aberrantes ou manquantes par la neutralité (1.0 pour le ratio MVRV).
    mvrv_filled = mvrv_series.replace(0.0, 1.0).fillna(1.0)

    # Logique de seuils psychologiques majeurs :
    # - MVRV > 3.0 : Zone d'euphorie extrême -> Score bas (0.1) car risque de crash élevé.
    # - MVRV < 1.0 : Zone de capitulation/sous-évaluation -> Score haut (0.9) car opportunité générationnelle.
    # - Entre les deux : Zone neutre.
    score = np.where(mvrv_filled > 3.0, 0.1,
            np.where(mvrv_filled < 1.0, 0.9, 0.5))

    # Nettoyage final pour garantir l'intégrité des données.
    score[mvrv_series.isna() | (mvrv_series == 0.0)] = 0.5

    return pd.Series(score, index=mvrv_series.index)

def calculate_exchange_netflow_score(netflow_series: pd.Series) -> pd.Series:
    """
    Calcule un score basé sur les flux nets de capitaux vers/hors des exchanges.

    L'hypothèse est que des sorties nettes (outflows) indiquent une accumulation (stockage à froid),
    tandis que des entrées nettes (inflows) indiquent une intention de vente potentielle.

    Args:
        netflow_series (pd.Series): Série des flux nets en USD.

    Returns:
        pd.Series: Un score (0-1) où > 0.5 est haussier (sorties) et < 0.5 est baissier (entrées).
    """
    # Gestion des données manquantes pour éviter les crashs.
    netflow_filled = netflow_series.replace(0.0, np.nan).fillna(0)

    # Normalisation adaptative : on utilise une fenêtre glissante pour s'adapter aux volumes changeants du marché.
    # Cela évite qu'un flux énorme en 2021 n'écrase les signaux de 2024.
    rolling_max_abs = netflow_filled.abs().rolling(window=30, min_periods=1).max()
    normalized_netflow = netflow_filled / (rolling_max_abs + 1e-9) # Epsilon pour éviter div/0

    # Transformation Sigmoïde : Convertit le flux (-infini, +infini) en probabilité (0, 1).
    # Inversion du signe : Flux négatif (sortie) = Bon signe = Score élevé.
    score = 1 / (1 + np.exp(normalized_netflow * 5))

    # Fallback neutre.
    score[netflow_series.isna() | (netflow_series == 0.0)] = 0.5

    return pd.Series(score, index=netflow_series.index)

def calculate_whale_accumulation_score(whale_accumulation_series: pd.Series) -> pd.Series:
    """
    Interprète le comportement des "Whales" (gros porteurs) pour générer un score de confiance.

    Les mouvements des baleines précèdent souvent ceux du marché (Smart Money).
    Une accumulation croissante par les gros portefeuilles est un signal haussier fort.

    Args:
        whale_accumulation_series (pd.Series): Delta ou métrique d'accumulation des whales.

    Returns:
        pd.Series: Score normalisé reflétant le sentiment des gros investisseurs.
    """
    accumulation_filled = whale_accumulation_series.replace(0.0, np.nan).fillna(0)

    # Normalisation pour ramener les données à une échelle comparable avant la sigmoïde.
    normalized_accumulation = accumulation_filled / (accumulation_filled.abs().max() + 1e-9)

    # Sigmoïde standard : Accumulation positive -> Score > 0.5.
    score = 1 / (1 + np.exp(-normalized_accumulation * 5))

    # Fallback neutre.
    score[whale_accumulation_series.isna() | (whale_accumulation_series == 0.0)] = 0.5

    return pd.Series(score, index=whale_accumulation_series.index)

def calculate_market_regime_features(db_engine: Any, symbol: str, timeframe: str) -> Dict[str, float]:
    """
    Détermine le régime de marché actuel pour ajuster les stratégies (ex: Tendance vs Range).

    Pour la Phase 1, retourne un score statique, mais est conçu pour intégrer plus tard
    des analyses de volatilité (ATR) et de force de tendance (ADX).

    Args:
        db_engine (Any): Connexion DB.
        symbol (str): Actif concerné.
        timeframe (str): Unité de temps.

    Returns:
        Dict[str, float]: Un dictionnaire de features de régime.
    """
    return {'regime_score': 0.5}

def calculate_onchain_features(db_engine: Any, symbol: str, timeframe: str) -> Dict[str, float]:
    """
    Agrège les métriques on-chain spécifiques pour enrichir l'analyse technique.

    Sert de placeholder pour l'intégration future de données complexes (NUPL, SOPR).

    Args:
        db_engine (Any): Connexion DB.
        symbol (str): Actif concerné.
        timeframe (str): Unité de temps.

    Returns:
        Dict[str, float]: Un dictionnaire de features on-chain.
    """
    return {'cvd_divergence_score': -0.2}
