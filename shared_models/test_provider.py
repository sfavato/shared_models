import unittest
from unittest.mock import patch, Mock
from provider import Binance

class TestBinance(unittest.TestCase):

	@patch('provider.requests.get')
	def test_get_candlestick_data_success(self, mock_get):
		# Arrange
		binance = Binance()
		symbol = "BTC"
		start_time = 1609459200000  # Example start time in milliseconds
		end_time = 1609545600000    # Example end time in milliseconds
		timeframe = '4h'
		expected_url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}USDT&interval={timeframe}&startTime={start_time}&endTime={end_time}"
		
		mock_response = Mock()
		mock_response.status_code = 200
		mock_response.json.return_value = [{"open": "1", "high": "2", "low": "0.5", "close": "1.5"}]
		mock_get.return_value = mock_response

		# Act
		result = binance.get_candlestick_data(symbol, start_time, end_time, timeframe)

		# Assert
		mock_get.assert_called_once_with(expected_url, headers={'Content-Type': 'application/json'})
		self.assertEqual(result, [{"open": "1", "high": "2", "low": "0.5", "close": "1.5"}])

	@patch('provider.requests.get')
	def test_get_candlestick_data_failure(self, mock_get):
		# Arrange
		binance = Binance()
		symbol = "BTC"
		start_time = 1609459200000  # Example start time in milliseconds
		end_time = 1609545600000    # Example end time in milliseconds
		timeframe = '4h'
		expected_url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}USDT&interval={timeframe}&startTime={start_time}&endTime={end_time}"
		
		mock_response = Mock()
		mock_response.status_code = 404
		mock_response.text = "Not Found"
		mock_get.return_value = mock_response

		# Act
		result = binance.get_candlestick_data(symbol, start_time, end_time, timeframe)

		# Assert
		mock_get.assert_called_once_with(expected_url, headers={'Content-Type': 'application/json'})
		self.assertIsNone(result)

if __name__ == '__main__':
	unittest.main()