# Import the required libraries
import base64
import hmac
import json
import logging
import time
import logging
from typing import Optional, List, Dict, Any, Tuple, Union

import requests

def get_timestamp() -> int:
    """
    Retourne le timestamp actuel en millisecondes.

    Utilisé pour synchroniser les requêtes API et signer les payloads.
    """
    return int(time.time() * 1000)

def to_query_with_no_encode(params: Dict[str, Any]) -> str:
    """
    Convertit un dictionnaire de paramètres en chaîne de requête brute sans encodage URL.

    Nécessaire pour la signature API de certains exchanges qui exigent un format spécifique.
    """
    return '&'.join(f"{key}={value}" for key, value in params.items())

def pre_hash(timestamp: int, method: str, request_path: str, body: str) -> str:
    """
    Prépare la chaîne de caractères à signer pour l'authentification API.

    Concatène les éléments critiques de la requête pour garantir son intégrité.
    """
    return f"{timestamp}{method.upper()}{request_path}{body}"


def sign(message: str, secret_key: str) -> str:
    """
    Signe un message avec une clé secrète en utilisant HMAC-SHA256.

    Garantit l'authenticité de la requête auprès de l'exchange.
    """
    mac = hmac.new(bytes(secret_key, encoding='utf8'), bytes(
        message, encoding='utf-8'), digestmod='sha256')
    digest = mac.digest()
    return base64.b64encode(digest).decode()

def parse_params_to_str(params: Dict[str, Any]) -> str:
    """
    Trie et formate les paramètres d'URL.

    Certains exchanges exigent un ordre lexicographique des paramètres pour la signature.
    """
    params_list = [(key, val) for key, val in params.items()]
    params_list.sort(key=lambda x: x[0])  # Sort by key
    url = '?' + to_query_with_no_encode(dict(params_list))
    return '' if url == '?' else url


class Bitget:
    """
    Wrapper pour l'API Bitget (Futures).

    Gère l'authentification, le passage d'ordres et la récupération des configurations de contrats.
    """
    def __init__(self, api_key: str, secret_key: str, passphrase: str, url: str, request_path: str):
        """
        Initialise le client Bitget.

        Args:
            api_key (str): Clé API publique.
            secret_key (str): Clé secrète.
            passphrase (str): Passphrase de sécurité du compte.
            url (str): URL de base de l'API.
            request_path (str): Chemin par défaut (ex: '/api/v2/...').
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.url = url
        self.request_path = request_path


    def placeorder(self, api_key: str, secret_key: str, passphrase: str, request_path: str, base_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Envoie une requête signée pour placer un ordre sur Bitget.

        Args:
            params (Dict[str, Any]): Les détails de l'ordre (symbol, side, size, etc.).

        Returns:
            Dict[str, Any]: La réponse JSON de l'API.
        """
        timestamp = get_timestamp()
        body = json.dumps(params)  # Convert the request body to JSON format
        pre_hash_string = pre_hash(timestamp, "POST", request_path, body)
        signature = sign(pre_hash_string, secret_key)

        # HTTP headers for Bitget API
        headers = {
            "ACCESS-KEY": api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": str(timestamp),
            "Content-Type": "application/json",
            "ACCESS-PASSPHRASE": passphrase,

        }

        url = f"{base_url}{request_path}"  # Full URL

        response = requests.post(url, headers=headers, data=body)
        return response.json()


    # Function to fetch decimals and minimum position size


    def get_contract_config(self, api_key: str, secret_key: str, passphrase: str, base_url: str, symbol: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Récupère les spécifications du contrat (précision prix/quantité).

        Essentiel pour normaliser les ordres avant envoi (arrondis corrects) et éviter les rejets API.

        Args:
            symbol (Optional[str]): Le symbole spécifique à chercher (ex: 'BTCUSDT'). Si None, retourne tout.

        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: Configuration du contrat ou liste de configs.
        """
        request_path = "/api/v2/mix/market/contracts"
        url = f"{base_url}{request_path}"
        timestamp = get_timestamp()
        body = ""  # Empty for GET requests

        # Generate the signature
        pre_hash_string = pre_hash(timestamp, "GET", request_path, body)
        signature = sign(pre_hash_string, secret_key)

        # Set the headers
        headers = {
            "ACCESS-KEY": api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": str(timestamp),
            "Content-Type": "application/json",
            "ACCESS-PASSPHRASE": passphrase
        }
        # Make the GET request
        logging.log(logging.INFO,
            f"Bitget Query for contract config: {{'productType':'usdt-futures','symbol':{symbol}}}")
        response = requests.get(url, headers=headers, params={
                                "productType": "usdt-futures", "symbol": symbol})
        logging.log(logging.INFO, f"Bitget Response for contract config: {json.dumps(response.json(), indent=4)} ")
        # Handle response
        if response.status_code == 200:
            data = response.json().get("data", [])
            if symbol:
                # Filter the data for the specified symbol
                for contract in data:
                    if contract.get("symbol") == symbol:
                        logging.log(logging.INFO, "Bitget contract config : " +
                                        json.dumps(contract, indent=4))
                        return {
                            "symbol": contract.get("symbol"),
                            "priceDecimals": contract.get("pricePlace"),
                            "positionSizeDecimals": contract.get("volPlace"),
                            "minTradeUSDT": contract.get("minTradeUSDT")
                        }
                return []
            else:
                # Return all contract data if no specific symbol is provided
                return data
        else:
            return []




    # Parameters for the POST request

    def place_trade(self, symbol: str, price: float, leverage: int, quantityUSDT: float, side: str, tp: float, sl: float, priceDecimals: int, minQuantity: float, marketPrice: bool) -> Dict[str, Any]:
        """
        Place un trade complet avec TP et SL.

        Cette méthode wrapper simplifie la création d'ordres complexes en gérant
        le calcul de taille (size) à partir du montant USDT et du levier.

        Args:
            symbol (str): Pair de trading.
            price (float): Prix d'entrée (limit) ou courant (market).
            leverage (int): Levier à utiliser.
            quantityUSDT (float): Montant de la marge en USDT.
            side (str): 'open_long' ou 'open_short'.
            tp (float): Prix du Take Profit.
            sl (float): Prix du Stop Loss.
            priceDecimals (int): Précision décimale du prix.
            minQuantity (float): Quantité minimum du contrat (non utilisé directement ici mais utile pour validation).
            marketPrice (bool): Si True, ordre Market. Si False, ordre Limit.

        Returns:
            Dict[str, Any]: Résultat de l'ordre.
        """
        if marketPrice:

            order_params = {
                "symbol": symbol,        # Trading pair
                "marginCoin": "USDT",       # Margin coin
                "productType": "USDT-FUTURES",
                "marginMode": "isolated",
                # "price": round(price,int(priceDecimals)),                  # Limit price
                # Order size (quantity)
                "size": (quantityUSDT/price)*leverage,
                "side": side,            # 'open_long', 'open_short', 'close_long', 'close_short'
                "tradeSide": "open",

                "orderType": "market",           # 'limit' or 'market'
                "leverage": leverage,                 # Leverage value (e.g., 20x)
                # Optional: Take-profit price
                "presetStopSurplusPrice": round(tp, int(priceDecimals)),
                # Optional: Stop-loss price
                "presetStopLossPrice": round(sl, int(priceDecimals)),
                "force": "gtc"               # Order execution type
            }
        else:
            order_params = {
                "symbol": symbol,        # Trading pair
                "marginCoin": "USDT",       # Margin coin
                "productType": "USDT-FUTURES",
                "marginMode": "isolated",
                # Limit price
                "price": round(price, int(priceDecimals)),
                # Order size (quantity)
                "size": (quantityUSDT/price)*leverage,
                "side": side,            # 'open_long', 'open_short', 'close_long', 'close_short'
                "tradeSide": "open",

                "orderType": "limit",           # 'limit' or 'market'
                "leverage": leverage,                 # Leverage value (e.g., 20x)
                # Optional: Take-profit price
                "presetStopSurplusPrice": round(tp, int(priceDecimals)),
                # Optional: Stop-loss price
                "presetStopLossPrice": round(sl, int(priceDecimals)),
                "force": "gtc"               # Order execution type
            }
        logging.log(logging.INFO, f"Bitget in-trade placing order: {symbol} {side} x{leverage} current: {price}, tp: {tp} sl: {sl} qty:{round(int(quantityUSDT)/price)*leverage}")
        result = self.placeorder(self.api_key, self.secret_key, self.passphrase, self.request_path, self.url, order_params)
        logging.log(logging.INFO, f"Bitget in-trade result: {json.dumps(result, indent=4)}")
        return result


class Binance:
    """
    Wrapper pour l'API publique de Binance (Data).

    Utilisé principalement pour récupérer des données de marché historiques et temps réel (Klines).
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def obtenir_tous_les_prix(symbols: List[str], chunk_size: int = 100) -> Optional[List[Dict[str, Any]]]:
        """
        Récupère les prix (tickers) pour une liste massive de symboles en optimisant les requêtes par lots.

        Args:
            symbols (List[str]): Liste des symboles à récupérer.
            chunk_size (int): Taille maximale des symboles par requête pour éviter les limites d'URL.

        Returns:
            Optional[List[Dict[str, Any]]]: Liste des résultats combinés ou None en cas d'erreur critique.
        """
        url = "https://data-api.binance.vision/api/v3/ticker"
        results = []

        # Diviser les symboles en morceaux
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]  # Prendre un morceau de la liste
            symbols_json = f'["{"\",\"".join(chunk)}"]'  # Créer la chaîne JSON brute

            try:
                # Envoyer la requête GET avec les symboles du morceau
                response = requests.get(f"{url}?symbols={symbols_json}")
                response.raise_for_status()  # Vérifier si la requête a réussi
                results.extend(response.json())  # Ajouter les résultats à la liste
            except requests.exceptions.RequestException as e:
                logging.log(logging.ERROR, f"Erreur lors de la requête pour les symboles {chunk}: {e}")
                return None

        return results  # Retourner tous les résultats combinés
    
    
    def get_symbol_precision(self, symbol: str) -> Dict[str, int]:
        """
        Récupère la précision requise pour le prix et la quantité d'un actif sur Binance.

        Args:
            symbol (str): Symbole (ex: 'MINAUSDT').

        Returns:
            Dict[str, int]: Dictionnaire {'price_precision': int, 'quantity_precision': int}.
        """
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url, params={"symbol": symbol})
        data = response.json()

        if "symbols" not in data or len(data["symbols"]) == 0:
            raise ValueError(
                f"Symbol {symbol} not found in Binance exchange info.")

        symbol_data = data["symbols"][0]
        price_precision = len(
            symbol_data["filters"][0]["tickSize"].split(".")[1].rstrip("0"))
        quantity_precision = len(
            symbol_data["filters"][3]["stepSize"].split(".")[1].rstrip("0"))

        return {
            "price_precision": price_precision,
            "quantity_precision": quantity_precision
        }
        
    

    # Get the candlestick data for a given symbol and timeframe
    # The start_time and end_time parameters should be in milliseconds
    # The symbol parameter should be the trading pair symbol (e.g. "BTC")
    def get_candlestick_data(self, symbol: str, start_time: int, end_time: int, timeframe: str = '4h') -> Optional[List[List[Any]]]:
        """
        Récupère les bougies (Klines) historiques.

        Args:
            symbol (str): Symbole.
            start_time (int): Timestamp début (ms).
            end_time (int): Timestamp fin (ms).
            timeframe (str): Intervalle (ex: '4h').

        Returns:
            Optional[List[List[Any]]]: Données brutes OHLCV de Binance ou None.
        """
        base_url = "https://data-api.binance.vision"
        endpoint = f"/api/v3/klines?symbol={symbol}&interval={timeframe}&startTime={start_time}&endTime={end_time}"

        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.get(base_url + endpoint, headers=headers)
        if response.status_code != 200:
            self.logger.error(f"Failed to retrieve data: {symbol} : {response.status_code}, {response.text}")
            self.logger.error(f"Endpoint: {base_url + endpoint}")
            return None
        return response.json()
    
    def get_highest_3min(self, symbol: str) -> Union[Tuple[float, float, float, float], bool]:
        """
        Récupère les données OHLC de la dernière bougie 3 minutes terminée.

        Utilisé pour des vérifications micro-structurelles (High Frequency checks).

        Returns:
            Tuple[float, float, float, float]: (High, Low, Close, Open) ou False si échec.
        """
        current_timestamp = int(time.time() * 1000)
        previous_3m_candlestick_end = (
            current_timestamp // (3 * 60 * 1000)) * (3 * 60 * 1000)
        previous_3m_candlestick_start = previous_3m_candlestick_end - \
            (3 * 60 * 1000)

        jsonData = self.get_candlestick_data(
            symbol, previous_3m_candlestick_start, previous_3m_candlestick_end, '3m')
        if not jsonData or len(jsonData) == 0:
            self.logger.error("No data returned from the API")
            return False
        return float(jsonData[0][2]),  float(jsonData[0][3]), float(jsonData[0][4]), float(jsonData[0][1])
    
    # Get the last 15m candlestick data and returns high, low, close, open
    def get_highest_15min(self, symbol: str) -> Union[Tuple[float, float, float, float], bool]:
        """
        Récupère les données OHLC de la dernière bougie 15 minutes terminée.

        Returns:
            Tuple[float, float, float, float]: (High, Low, Close, Open) ou False si échec.
        """
        current_timestamp = int(time.time() * 1000)
        previous_15m_candlestick_end = (
            current_timestamp // (15 * 60 * 1000)) * (15 * 60 * 1000)
        previous_15m_candlestick_start = previous_15m_candlestick_end - \
            (15 * 60 * 1000)

        jsonData = self.get_candlestick_data(
            symbol, previous_15m_candlestick_start, previous_15m_candlestick_end, '15m')

        if not jsonData or len(jsonData) == 0:
            self.logger.error("No data returned from the API")
            return False
        return float(jsonData[0][2]),  float(jsonData[0][3]), float(jsonData[0][4]), float(jsonData[0][1])

                
    @staticmethod
    def calculate_current_timestamps() -> Tuple[int, int]:
        """
        Calcule les bornes temporelles de la dernière bougie 4H complétée.

        Returns:
            Tuple[int, int]: (Start Time, End Time) en ms.
        """
        current_timestamp = int(time.time() * 1000)
        previous_4h_candlestick_end = (
            current_timestamp // (4 * 60 * 60 * 1000)) * (4 * 60 * 60 * 1000)
        previous_4h_candlestick_start = previous_4h_candlestick_end - \
            (4 * 60 * 60 * 1000)
        return previous_4h_candlestick_start, previous_4h_candlestick_end

    @staticmethod
    def calculate_previous_4h_timestamps() -> Tuple[int, int]:
        """
        Calcule les bornes temporelles de l'avant-dernière bougie 4H.

        Returns:
            Tuple[int, int]: (Start Time, End Time) en ms.
        """
        current_timestamp = int(time.time() * 1000)
        previous_4h_candlestick_end = (
            current_timestamp // (4 * 60 * 60 * 1000)) * (4 * 60 * 60 * 1000)
        previous_4h_candlestick_start = previous_4h_candlestick_end - \
            (4 * 60 * 60 * 1000)
        start_of_before_4h_candlestick = previous_4h_candlestick_start - \
            (4 * 60 * 60 * 1000)
        end_of_before_4h_candlestick = previous_4h_candlestick_start
        return start_of_before_4h_candlestick, end_of_before_4h_candlestick

    @staticmethod
    def calculate_3d_timestamps() -> Tuple[int, int]:
        """
        Calcule les bornes temporelles de l'avant-dernière bougie 3 jours.

        Returns:
            Tuple[int, int]: (Start Time, End Time) en ms.
        """
        current_timestamp = int(time.time() * 1000)
        previous_3d_candlestick_end = (
            current_timestamp // (3 * 24 * 60 * 60 * 1000)) * (3 * 24 * 60 * 60 * 1000)
        previous_3d_candlestick_start = previous_3d_candlestick_end - \
            (3 * 24 * 60 * 60 * 1000)
        start_of_before_3d_candlestick = previous_3d_candlestick_start - \
            (3 * 24 * 60 * 60 * 1000)
        end_of_before_3d_candlestick = previous_3d_candlestick_start
        return start_of_before_3d_candlestick, end_of_before_3d_candlestick
    
    @staticmethod
    def calculate_previous_two_4h_candlesticks() -> Tuple[int, int]:
        """
        Calcule le timestamp de début de la bougie 4H d'il y a 8 heures.

        Returns:
            Tuple[int, int]: (Start Time, End Time) en ms.
        """
        current_timestamp = int(time.time() * 1000)

        # Calculate the timestamp of the previous 4-hour candlestick
        previous_4h_candlestick_end = (
            current_timestamp // (4 * 60 * 60 * 1000)) * (4 * 60 * 60 * 1000)
        previous_4h_candlestick_start = previous_4h_candlestick_end - \
            (4 * 60 * 60 * 1000)
        previous_4h_candlestick_end = previous_4h_candlestick_start
        previous_4h_candlestick_start = previous_4h_candlestick_end - \
            (4 * 60 * 60 * 1000)

        start_of_before_4h_candlestick = previous_4h_candlestick_start - \
            (4 * 60 * 60 * 1000)
        end_of_before_4h_candlestick = previous_4h_candlestick_start

        return start_of_before_4h_candlestick, end_of_before_4h_candlestick


if __name__ == "__main__":
    myBinance = Binance()
    print(myBinance.get_highest_3min("BTCUSDT"))
    print(myBinance.get_highest_15min("BTCUSDT"))
    print(myBinance.calculate_current_timestamps())
    print(myBinance.calculate_previous_4h_timestamps())
    print(myBinance.calculate_3d_timestamps())
    print(myBinance.calculate_previous_two_4h_candlesticks())
    print(myBinance.get_symbol_precision("BTCUSDT"))
    print(myBinance.obtenir_tous_les_prix(["BTCUSDT", "ETHUSDT", "BNBUSDT"]))
    print(myBinance.get_candlestick_data("BTC", 1609459200000, 1609545600000, '4h'))
