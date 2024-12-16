# Import the required libraries
import base64
import hmac
import json
import logging
import time
import logging

import requests

def get_timestamp():
    return int(time.time() * 1000)

def to_query_with_no_encode(params):
    return '&'.join(f"{key}={value}" for key, value in params)

def pre_hash(timestamp, method, request_path, body):
    return f"{timestamp}{method.upper()}{request_path}{body}"


def sign(message, secret_key):
    mac = hmac.new(bytes(secret_key, encoding='utf8'), bytes(
        message, encoding='utf-8'), digestmod='sha256')
    digest = mac.digest()
    return base64.b64encode(digest).decode()

def parse_params_to_str(params):
    params = [(key, val) for key, val in params.items()]
    params.sort(key=lambda x: x[0])  # Sort by key
    url = '?' + to_query_with_no_encode(params)
    return '' if url == '?' else url


class Bitget:
    def __init__(self, api_key, secret_key, passphrase, url, request_path):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.url = url
        self.request_path = request_path





    def placeorder(self, api_key, secret_key, passphrase, request_path, base_url, params):
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


    def get_contract_config(self, api_key, secret_key, passphrase, base_url, symbol=None):
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
        logging.log(
            "Bitget Query for contract config: '{'productType':'usdt-futures','symbol':{symbol}'}'")
        response = requests.get(url, headers=headers, params={
                                "productType": "usdt-futures", "symbol": symbol})
        logging.log(f"Bitget Response for contract config: {
                        json.dumps(response.json(), indent=4)} ")
        # Handle response
        if response.status_code == 200:
            data = response.json().get("data", [])
            if symbol:
                # Filter the data for the specified symbol
                for contract in data:
                    if contract.get("symbol") == symbol:
                        logging.log("Bitget contract config : " +
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

    def place_trade(self,symbol, price, leverage, quantityUSDT, side, tp, sl, priceDecimals, minQuantity, marketPrice):
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
        logging.log(f"Bitget in-trade placing order: {symbol} {side} x{leverage} current: {
                        price}, tp: {tp} sl: {sl} qty:{round(int(quantityUSDT)/price)*leverage}")
        result = self.placeorder(self.api_key, self.secret_key, self.passphrase, self.request_path, self.url, order_params)
        logging.log(f"Bitget in-trade result: {json.dumps(result, indent=4)}")
class Binance:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def obtenir_tous_les_prix(symbols, chunk_size=100):
        """
        Récupère les prix pour une liste de symboles en effectuant des requêtes en morceaux.
        :param symbols: Liste des symboles à récupérer.
        :param chunk_size: Taille maximale des symboles par requête.
        :return: Liste des résultats combinés ou None en cas d'erreur.
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
                logging.log(f"Erreur lors de la requête pour les symboles {chunk}: {e}")
                return None

        return results  # Retourner tous les résultats combinés

    def get_symbol_precision(self, symbol):
        """
        Fetch the price and quantity precision for a given symbol.

        Args:
            symbol (str): The trading pair symbol (e.g., 'MINAUSDT').

        Returns:
            dict: A dictionary with 'price_precision' and 'quantity_precision'.
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
    def get_candlestick_data(self,symbol, start_time, end_time, timeframe='4h'):
        base_url = "https://data-api.binance.vision"
        endpoint = f"/api/v3/klines?symbol={
            symbol}USDT&interval={timeframe}&startTime={start_time}&endTime={end_time}"

        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.get(base_url + endpoint, headers=headers)
        if response.status_code != 200:
            self.logger.error(f"Failed to retrieve data: {symbol} : {response.status_code}, {response.text}")
            return None
        return response.json()
    
    def get_highest_3min(self,symbol):
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
    def get_highest_15min(self,symbol):
        current_timestamp = int(time.time() * 1000)
        previous_15m_candlestick_end = (
            current_timestamp // (15 * 60 * 1000)) * (15 * 60 * 1000)
        previous_15m_candlestick_start = previous_15m_candlestick_end - \
            (15 * 60 * 1000)

        jsonData = Binance.get_candlestick_data(
            symbol, previous_15m_candlestick_start, previous_15m_candlestick_end, '15m')

        if not jsonData or len(jsonData) == 0:
            self.logger.error("No data returned from the API")
            return False
        return float(jsonData[0][2]),  float(jsonData[0][3]), float(jsonData[0][4]), float(jsonData[0][1])

                
    @staticmethod
    def calculate_current_timestamps():
        current_timestamp = int(time.time() * 1000)
        previous_4h_candlestick_end = (
            current_timestamp // (4 * 60 * 60 * 1000)) * (4 * 60 * 60 * 1000)
        previous_4h_candlestick_start = previous_4h_candlestick_end - \
            (4 * 60 * 60 * 1000)
        return previous_4h_candlestick_start, previous_4h_candlestick_end

    @staticmethod
    def calculate_previous_4h_timestamps():
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
    def calculate_3d_timestamps():
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
    def calculate_previous_two_4h_candlesticks():
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
