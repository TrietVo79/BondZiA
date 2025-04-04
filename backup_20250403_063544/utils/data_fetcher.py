import os
import json
import requests
import websocket
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor
from utils.logger_config import logger
import traceback

class PolygonDataFetcher:
    """Lớp xử lý việc lấy dữ liệu từ Polygon.io API"""
    
    def __init__(self, api_key=None, config_path="../config/system_config.json"):
        """
        Khởi tạo lớp lấy dữ liệu Polygon.io
        
        Args:
            api_key (str, optional): API key của Polygon.io
            config_path (str): Đường dẫn tới file cấu hình
        """
        # Đọc cấu hình
        self.config_path = config_path
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Đọc api key nếu không được cung cấp
        if api_key is None:
            api_keys_path = os.path.join(os.path.dirname(self.config_path), "api_keys.json")
            with open(api_keys_path, 'r') as f:
                api_keys = json.load(f)
            api_key = api_keys['polygon']['api_key']
        
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
        # Đọc danh sách cổ phiếu từ cấu hình
        stocks_config_path = os.path.join(os.path.dirname(self.config_path), "stocks.json")
        with open(stocks_config_path, 'r') as f:
            stocks_config = json.load(f)
        
        self.stocks = [stock['symbol'] for stock in stocks_config['stocks'] if stock['enabled']]
        
        # Thông số về thời gian lấy dữ liệu
        self.data_config = self.config['data']
        self.intraday_history_days = self.data_config['intraday_history_days']
        self.five_day_history_days = self.data_config['five_day_history_days']
        self.monthly_history_days = self.data_config['monthly_history_days']
        
        # Cache dữ liệu
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_expiry_minutes = self.data_config['cache_expiry_minutes']
        
        # Timezone
        self.market_timezone = pytz.timezone(self.config['market']['timezone'])
        
        logger.info(f"Khởi tạo PolygonDataFetcher với {len(self.stocks)} cổ phiếu")
    
    def _make_request(self, endpoint, params=None):
        """
        Thực hiện request tới Polygon API
        
        Args:
            endpoint (str): Endpoint API
            params (dict, optional): Tham số query
            
        Returns:
            dict: Dữ liệu JSON phản hồi từ API
        """
        if params is None:
            params = {}
        
        # Thêm API key vào params
        params['apiKey'] = self.api_key
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise exception cho lỗi HTTP
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP Error: {http_err} - URL: {url}")
            if response.status_code == 429:
                # Rate limit - Chờ và thử lại
                logger.warning("Rate limit exceeded, waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(endpoint, params)
            else:
                return {"status": "error", "error": str(http_err)}
        except Exception as err:
            logger.error(f"Error: {err} - URL: {url}")
            return {"status": "error", "error": str(err)}
    
    def _check_cache(self, cache_key, max_age_minutes=None):
        """
        Kiểm tra cache cho dữ liệu đã lưu
        
        Args:
            cache_key (str): Khóa cache
            max_age_minutes (int, optional): Tuổi tối đa của cache tính bằng phút
            
        Returns:
            bool: True nếu cache hợp lệ, False nếu không
        """
        if max_age_minutes is None:
            max_age_minutes = self.cache_expiry_minutes
            
        if cache_key not in self.cache or cache_key not in self.cache_timestamps:
            return False
        
        # Kiểm tra thời gian cache
        cache_age = datetime.now() - self.cache_timestamps[cache_key]
        return cache_age.total_seconds() / 60 < max_age_minutes
    
    def _store_cache(self, cache_key, data):
        """
        Lưu dữ liệu vào cache
        
        Args:
            cache_key (str): Khóa cache
            data: Dữ liệu cần lưu
        """
        self.cache[cache_key] = data
        self.cache_timestamps[cache_key] = datetime.now()
    
    def get_ticker_details(self, symbol):
        """
        Lấy thông tin chi tiết về một mã cổ phiếu
        
        Args:
            symbol (str): Mã cổ phiếu
            
        Returns:
            dict: Thông tin chi tiết của cổ phiếu
        """
        cache_key = f"ticker_details_{symbol}"
        
        if self._check_cache(cache_key, max_age_minutes=1440):  # Cache 24 giờ
            return self.cache[cache_key]
        
        endpoint = f"/v3/reference/tickers/{symbol}"
        response = self._make_request(endpoint)
        
        if response.get("status") == "OK" or response.get("status") == "ok":
            self._store_cache(cache_key, response.get("results", {}))
            return response.get("results", {})
        else:
            logger.error(f"Lỗi khi lấy thông tin cổ phiếu {symbol}: {response.get('error', 'Unknown error')}")
            return {}
    
    def get_aggregates(self, symbol, multiplier, timespan, from_date, to_date, adjusted=True):
        """
        Lấy dữ liệu tổng hợp cho một cổ phiếu
        
        Args:
            symbol (str): Mã cổ phiếu
            multiplier (int): Bội số của timespan
            timespan (str): Đơn vị thời gian (minute, hour, day, week, month, quarter, year)
            from_date (str): Ngày bắt đầu định dạng YYYY-MM-DD
            to_date (str): Ngày kết thúc định dạng YYYY-MM-DD
            adjusted (bool): True để lấy dữ liệu đã điều chỉnh (adjusted)
            
        Returns:
            DataFrame: Dữ liệu tổng hợp
        """
        cache_key = f"agg_{symbol}_{multiplier}_{timespan}_{from_date}_{to_date}_{adjusted}"
        
        if self._check_cache(cache_key):
            return self.cache[cache_key]
        
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "adjusted": "true" if adjusted else "false",
        }
        
        response = self._make_request(endpoint, params)
        
        if response.get("status") == "OK" or response.get("resultsCount", 0) > 0:
            results = response.get("results", [])
            
            if not results:
                logger.warning(f"Không có dữ liệu cho {symbol} từ {from_date} đến {to_date}")
                return pd.DataFrame()
            
            # Chuyển đổi sang DataFrame
            df = pd.DataFrame(results)
            
            # Đổi tên cột theo chuẩn OHLCV
            column_map = {
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vw': 'vwap',
                't': 'timestamp',
                'n': 'transactions'
            }
            
            df = df.rename(columns=column_map)
            
            # Chuyển đổi timestamp từ ms sang datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['date'] = df['timestamp'].dt.date
            
            # Set index
            df.set_index('timestamp', inplace=True)
            
            # Lưu vào cache
            self._store_cache(cache_key, df)
            
            return df
        else:
            logger.error(f"Lỗi khi lấy dữ liệu tổng hợp cho {symbol}: {response.get('error', 'Unknown error')}")
            return pd.DataFrame()
    
    def get_intraday_data(self, symbol, days_back=1, interval='5min'):
        """
        Lấy dữ liệu intraday cho một cổ phiếu
        
        Args:
            symbol (str): Mã cổ phiếu
            days_back (int): Số ngày lấy dữ liệu trong quá khứ
            interval (str): Khoảng thời gian ('1min', '5min', '15min', etc.)
            
        Returns:
            DataFrame: Dữ liệu intraday
        """
        # Phân tách khoảng thời gian thành multiplier và timespan
        if interval.endswith('min'):
            multiplier = int(interval[:-3])
            timespan = 'minute'
        elif interval.endswith('hour'):
            multiplier = int(interval[:-4])
            timespan = 'hour'
        else:
            raise ValueError(f"Không hỗ trợ khoảng thời gian: {interval}")
        
        # Tính toán ngày bắt đầu và kết thúc
        to_date = datetime.now(self.market_timezone).strftime('%Y-%m-%d')
        from_date = (datetime.now(self.market_timezone) - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        return self.get_aggregates(symbol, multiplier, timespan, from_date, to_date)
    
    def get_daily_data(self, symbol, days_back=365):
        """
        Lấy dữ liệu ngày cho một cổ phiếu
        
        Args:
            symbol (str): Mã cổ phiếu
            days_back (int): Số ngày lấy dữ liệu trong quá khứ
            
        Returns:
            DataFrame: Dữ liệu ngày
        """
        # Tính toán ngày bắt đầu và kết thúc
        to_date = datetime.now(self.market_timezone).strftime('%Y-%m-%d')
        from_date = (datetime.now(self.market_timezone) - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        return self.get_aggregates(symbol, 1, 'day', from_date, to_date)
    
    def get_technical_indicators(self, dataframe):
        """
        Tính toán các chỉ báo kỹ thuật từ dữ liệu giá
        
        Args:
            dataframe (DataFrame): DataFrame với dữ liệu giá OHLCV
            
        Returns:
            DataFrame: DataFrame với các chỉ báo kỹ thuật thêm vào
        """
        df = dataframe.copy()
        
        # Nếu DataFrame rỗng, trả về ngay
        if df.empty:
            return df
        
        # Đảm bảo chỉ mục thời gian được sắp xếp
        df = df.sort_index()
        
        # Tính toán SMA (Simple Moving Average)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Tính toán EMA (Exponential Moving Average)
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        
        # Tính toán RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Tính RSI
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Tính toán Bollinger Bands
        df['bollinger_mid'] = df['close'].rolling(window=20).mean()
        df['bollinger_std'] = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = df['bollinger_mid'] + 2 * df['bollinger_std']
        df['bollinger_lower'] = df['bollinger_mid'] - 2 * df['bollinger_std']
        
        # Tính toán MACD (Moving Average Convergence Divergence)
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Tính ATR (Average True Range)
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr_14'] = df['tr'].rolling(window=14).mean()
        
        # Tính Volatility (HV)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['historical_volatility'] = df['log_return'].rolling(window=20).std() * np.sqrt(252) * 100
        
        # Loại bỏ các cột tạm thời
        df = df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1, errors='ignore')
        
        return df
    
    def get_intraday_with_indicators(self, symbol, days_back=None, interval='5min'):
        """
        Lấy dữ liệu intraday kèm chỉ báo kỹ thuật
        
        Args:
            symbol (str): Mã cổ phiếu
            days_back (int, optional): Số ngày lấy dữ liệu. Mặc định theo cấu hình.
            interval (str): Khoảng thời gian
            
        Returns:
            DataFrame: Dữ liệu intraday với chỉ báo kỹ thuật
        """
        if days_back is None:
            days_back = self.intraday_history_days
            
        df = self.get_intraday_data(symbol, days_back, interval)
        return self.get_technical_indicators(df)
    
    def get_daily_with_indicators(self, symbol, days_back=None):
        """
        Lấy dữ liệu ngày kèm chỉ báo kỹ thuật
        
        Args:
            symbol (str): Mã cổ phiếu
            days_back (int, optional): Số ngày lấy dữ liệu. Mặc định theo cấu hình.
            
        Returns:
            DataFrame: Dữ liệu ngày với chỉ báo kỹ thuật
        """
        if days_back is None:
            days_back = self.five_day_history_days
            
        df = self.get_daily_data(symbol, days_back)
        return self.get_technical_indicators(df)
    
    def get_latest_quote(self, symbol):
        """
        Lấy báo giá mới nhất cho một cổ phiếu
        
        Args:
            symbol (str): Mã cổ phiếu
            
        Returns:
            dict: Thông tin báo giá mới nhất
        """
        cache_key = f"latest_quote_{symbol}"
        
        # Đối với báo giá, chỉ cache trong 1 phút
        if self._check_cache(cache_key, max_age_minutes=1):
            return self.cache[cache_key]
        
        endpoint = f"/v2/last/nbbo/{symbol}"
        response = self._make_request(endpoint)
        
        if response.get("status") == "OK":
            result = response.get("results", {})
            self._store_cache(cache_key, result)
            return result
        else:
            logger.error(f"Lỗi khi lấy báo giá cho {symbol}: {response.get('error', 'Unknown error')}")
            return {}
    
    def get_multiple_quotes(self, symbols):
        """
        Lấy báo giá cho nhiều cổ phiếu
        
        Args:
            symbols (list): Danh sách mã cổ phiếu
            
        Returns:
            dict: Dict với key là symbol và value là thông tin báo giá
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(self.get_latest_quote, symbol): symbol for symbol in symbols}
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    results[symbol] = data
                except Exception as exc:
                    logger.error(f"Lỗi khi lấy báo giá cho {symbol}: {str(exc)}")
        
        return results
    
    def get_news(self, symbol, limit=10):
        """
        Lấy tin tức cho một cổ phiếu
        
        Args:
            symbol (str): Mã cổ phiếu
            limit (int): Số lượng tin tức tối đa
            
        Returns:
            list: Danh sách tin tức
        """
        cache_key = f"news_{symbol}_{limit}"
        
        if self._check_cache(cache_key, max_age_minutes=60):  # Cache 1 giờ
            return self.cache[cache_key]
        
        endpoint = f"/v2/reference/news"
        params = {
            "ticker": symbol,
            "limit": limit,
            "order": "desc",
            "sort": "published_utc"
        }
        
        response = self._make_request(endpoint, params)
        
        if response.get("status") == "OK":
            results = response.get("results", [])
            self._store_cache(cache_key, results)
            return results
        else:
            logger.error(f"Lỗi khi lấy tin tức cho {symbol}: {response.get('error', 'Unknown error')}")
            return []
    
    def get_all_data_for_prediction(self, symbol):
        """
        Lấy tất cả dữ liệu cần thiết cho việc dự đoán
        
        Args:
            symbol (str): Mã cổ phiếu
            
        Returns:
            dict: Dictionary chứa tất cả dữ liệu
        """
        try:
            # Lấy dữ liệu intraday
            intraday_data = self.get_intraday_with_indicators(symbol, 
                                                            days_back=self.intraday_history_days, 
                                                            interval='5min')
            
            # Lấy dữ liệu ngày
            daily_data = self.get_daily_with_indicators(symbol, 
                                                     days_back=self.five_day_history_days)
            
            # Lấy báo giá mới nhất
            latest_quote = self.get_latest_quote(symbol)
            
            # Lấy tin tức
            news = self.get_news(symbol, limit=5)
            
            # Đóng gói tất cả dữ liệu
            return {
                'symbol': symbol,
                'intraday_data': intraday_data,
                'daily_data': daily_data,
                'latest_quote': latest_quote,
                'news': news,
                'timestamp': datetime.now(self.market_timezone).isoformat()
            }
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu cho {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now(self.market_timezone).isoformat()
            }
    
    def get_batch_data_for_all_stocks(self):
        """
        Lấy dữ liệu cho tất cả các cổ phiếu được cấu hình
        
        Returns:
            dict: Dictionary với key là symbol và value là dữ liệu tương ứng
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {executor.submit(self.get_all_data_for_prediction, symbol): symbol 
                              for symbol in self.stocks}
            
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    results[symbol] = data
                except Exception as exc:
                    logger.error(f"Lỗi khi lấy dữ liệu cho {symbol}: {str(exc)}")
                    results[symbol] = {'symbol': symbol, 'error': str(exc)}
        
        return results
    
    def init_websocket(self, symbols=None, callback=None):
        """
        Khởi tạo kết nối WebSocket cho dữ liệu thời gian thực
        
        Args:
            symbols (list, optional): Danh sách mã cổ phiếu. Mặc định là danh sách đã cấu hình.
            callback (function, optional): Hàm callback xử lý dữ liệu nhận được
        
        Returns:
            websocket.WebSocketApp: Đối tượng WebSocket
        """
        if symbols is None:
            symbols = self.stocks
            
        # Tạo chuỗi đăng ký WebSocket
        channels = []
        for symbol in symbols:
            channels.extend([f"T.{symbol}", f"Q.{symbol}", f"AM.{symbol}"])
        
        def on_message(ws, message):
            # Parse message
            try:
                data = json.loads(message)
                if callback:
                    callback(data)
                else:
                    # Default processing
                    logger.debug(f"WebSocket nhận: {data}")
            except json.JSONDecodeError:
                logger.error(f"Không thể parse dữ liệu WebSocket: {message}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket đóng kết nối: {close_status_code} - {close_msg}")
        
        def on_open(ws):
            logger.info(f"WebSocket mở kết nối")
            # Đăng ký kênh
            ws.send(json.dumps({
                "action": "auth",
                "params": self.api_key
            }))
            
            ws.send(json.dumps({
                "action": "subscribe",
                "params": channels
            }))
        
        # Khởi tạo WebSocket
        ws = websocket.WebSocketApp(
            f"wss://socket.polygon.io/stocks",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        return ws
    
    def save_data_to_disk(self, data, folder="../data/raw", filename=None):
        """
        Lưu dữ liệu vào đĩa
        
        Args:
            data: Dữ liệu cần lưu
            folder (str): Thư mục lưu trữ
            filename (str, optional): Tên file. Mặc định tạo tên theo ngày giờ hiện tại.
            
        Returns:
            str: Đường dẫn tới file đã lưu
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"polygon_data_{timestamp}.json"
        
        # Đảm bảo thư mục tồn tại
        folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            folder.lstrip("../"))
        
        os.makedirs(folder, exist_ok=True)
        
        file_path = os.path.join(folder, filename)
        
        try:
            with open(file_path, 'w') as f:
                # Chuyển đổi DataFrame thành dict nếu cần
                serializable_data = {}
                for symbol, symbol_data in data.items():
                    serializable_data[symbol] = {}
                    for key, value in symbol_data.items():
                        if isinstance(value, pd.DataFrame):
                            # Chuyển DataFrame thành dict
                            serializable_data[symbol][key] = value.reset_index().to_dict(orient='records')
                        else:
                            serializable_data[symbol][key] = value
                
                json.dump(serializable_data, f, indent=2, default=str)
            
            logger.info(f"Đã lưu dữ liệu vào {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Lỗi khi lưu dữ liệu: {str(e)}")
            return None

if __name__ == "__main__":
    # Test module
    logger.info("Kiểm tra module lấy dữ liệu từ Polygon.io")
    fetcher = PolygonDataFetcher()
    
    # Test lấy thông tin cổ phiếu
    logger.info("Lấy thông tin cổ phiếu AAPL")
    ticker_details = fetcher.get_ticker_details("AAPL")
    logger.info(f"Ticker details: {ticker_details}")
    
    # Test lấy dữ liệu intraday
    logger.info("Lấy dữ liệu intraday AAPL")
    intraday_data = fetcher.get_intraday_with_indicators("AAPL", days_back=1)
    logger.info(f"Intraday data shape: {intraday_data.shape}")
    
    # Test lấy báo giá mới nhất
    logger.info("Lấy báo giá mới nhất AAPL")
    latest_quote = fetcher.get_latest_quote("AAPL")
    logger.info(f"Latest quote: {latest_quote}")
