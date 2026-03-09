def calculate_moving_average(data, period):
    if len(data) < period:
        return None  # Not enough data for calculation

    closing_prices = [float(entry[4]) for entry in data]
    sum_prices = sum(closing_prices[:period])

    for i in range(period, len(closing_prices)):
        sum_prices += closing_prices[i] - closing_prices[i - period]

    return sum_prices / period

def obtenir_noms_usdt(indices):
    from . import BASE_CURRENCY
    # Récupérer les noms uniques à partir des objets "Trade"
    # Ensemble pour éviter les doublons
    noms_uniques = {indice.nom for indice in indices}
    # Trier les noms par ordre alphabétique
    noms_triés = sorted(noms_uniques)
    # Ajouter la devise de base comme suffixe
    noms_symboles = [nom + BASE_CURRENCY for nom in noms_triés]
    return noms_symboles