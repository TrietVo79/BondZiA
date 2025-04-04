import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from bs4 import BeautifulSoup
import re
from pytrends.request import TrendReq
from utils.logger_config import logger

class EnhancedDataFetcher:
    """
    Lớp nâng cao cho việc lấy dữ liệu từ nhiều nguồn khác nhau
    """
    
    def __init__(self, config_path="../config/system_config.json"):
        """
        Khởi tạo EnhancedDataFetcher
        
        Args:
            config_path (str): Đường dẫn đến file cấu hình
        """
        # Đọc cấu hình
        self.config_path = config_path
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Đọc API keys
        api_keys_path = os.path.join(os.path.dirname(self.config_path), "api_keys.json")
        with open(api_keys_path, 'r') as f:
            self.api_keys = json.load(f)
            
        # Lấy Polygon API key
        self.polygon_api_key = self.api_keys['polygon']['api_key']
        
        # Khởi tạo Google Trends
        try:
            self.pytrends = TrendReq(hl='en-US', tz=360)
            self.trend_enabled = True
        except Exception as e:
            logger.warning(f"Không thể khởi tạo Google Trends: {e}")
            self.trend_enabled = False
            
        # Đường dẫn lưu trữ dữ liệu
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        self.raw_data_dir = os.path.join(self.data_dir, "raw")
        self.processed_data_dir = os.path.join(self.data_dir, "processed")
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Cache các dữ liệu đã tải xuống
        self.news_cache = {}
        self.trend_cache = {}
        self.economic_data_cache = {}
        
        logger.info("Đã khởi tạo EnhancedDataFetcher")
        
    def get_stock_data_from_yfinance(self, symbol, period="1y", interval="1d"):
        """
        Lấy dữ liệu cổ phiếu từ Yahoo Finance
        
        Args:
            symbol (str): Mã cổ phiếu
            period (str): Khoảng thời gian (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Khoảng thời gian (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame: Dữ liệu cổ phiếu
        """
        try:
            logger.info(f"Lấy dữ liệu từ Yahoo Finance cho {symbol}, period={period}, interval={interval}")
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)
            
            # Đổi tên cột về lowercase cho thống nhất với Polygon
            df.columns = [col.lower() for col in df.columns]
            
            # Xử lý giá đóng cửa đã điều chỉnh
            if 'adj close' in df.columns:
                df.rename(columns={'adj close': 'adjclose'}, inplace=True)
                
            # Reset index để date trở thành cột
            df.reset_index(inplace=True)
            
            # Đổi tên cột Date nếu có
            if 'Date' in df.columns:
                df.rename(columns={'Date': 'date'}, inplace=True)
                
            # Đổi tên cột Datetime nếu có
            if 'Datetime' in df.columns:
                df.rename(columns={'Datetime': 'date'}, inplace=True)
            
            logger.info(f"Đã lấy {len(df)} dòng dữ liệu từ Yahoo Finance cho {symbol}")
            return df
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu từ Yahoo Finance cho {symbol}: {e}")
            return pd.DataFrame()
            
    def get_company_financial_data(self, symbol):
        """
        Lấy dữ liệu tài chính cơ bản của công ty từ Yahoo Finance
        
        Args:
            symbol (str): Mã cổ phiếu
            
        Returns:
            dict: Dữ liệu tài chính cơ bản
        """
        try:
            logger.info(f"Lấy dữ liệu tài chính cơ bản cho {symbol}")
            stock = yf.Ticker(symbol)
            
            # Lấy thông tin cơ bản
            info = stock.info
            
            # Lọc các thông tin quan trọng
            financial_data = {
                'marketCap': info.get('marketCap'),
                'trailingPE': info.get('trailingPE'),
                'forwardPE': info.get('forwardPE'),
                'dividendYield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow'),
                'averageVolume': info.get('averageVolume'),
                'averageVolume10days': info.get('averageVolume10days'),
                'shortRatio': info.get('shortRatio'),
                'shortPercentOfFloat': info.get('shortPercentOfFloat'),
                'sector': info.get('sector'),
                'industry': info.get('industry')
            }
            
            return financial_data
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu tài chính cơ bản cho {symbol}: {e}")
            return {}
    
    def get_stock_news(self, symbol, limit=10):
        """
        Lấy tin tức liên quan đến cổ phiếu từ các nguồn miễn phí
        
        Args:
            symbol (str): Mã cổ phiếu
            limit (int): Số lượng tin tức tối đa
            
        Returns:
            list: Danh sách các tin tức
        """
        # Kiểm tra cache
        cache_key = f"{symbol}_{limit}"
        if cache_key in self.news_cache:
            # Chỉ sử dụng cache nếu tin tức đã được tải trong vòng 6 giờ qua
            if (datetime.now() - self.news_cache[cache_key]['timestamp']).total_seconds() < 21600:  # 6 giờ
                return self.news_cache[cache_key]['data']
        
        news = []
        try:
            # Lấy tin tức từ Yahoo Finance
            logger.info(f"Lấy tin tức từ Yahoo Finance cho {symbol}")
            stock = yf.Ticker(symbol)
            yahoo_news = stock.news
            
            for item in yahoo_news[:limit]:
                news.append({
                    'title': item.get('title'),
                    'published': datetime.fromtimestamp(item.get('providerPublishTime')).strftime('%Y-%m-%d %H:%M:%S'),
                    'source': item.get('publisher'),
                    'summary': item.get('summary'),
                    'url': item.get('link')
                })
            
            # Lưu vào cache
            self.news_cache[cache_key] = {
                'data': news,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Đã lấy {len(news)} tin tức cho {symbol}")
            return news
        except Exception as e:
            logger.error(f"Lỗi khi lấy tin tức cho {symbol}: {e}")
            return []
    
    def get_google_trends(self, symbol, timeframe='today 3-m'):
        """
        Lấy dữ liệu xu hướng tìm kiếm từ Google Trends
        
        Args:
            symbol (str): Mã cổ phiếu
            timeframe (str): Khoảng thời gian
            
        Returns:
            DataFrame: Dữ liệu xu hướng
        """
        if not self.trend_enabled:
            logger.warning("Google Trends không được kích hoạt")
            return pd.DataFrame()
        
        # Kiểm tra cache
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.trend_cache:
            # Chỉ sử dụng cache nếu dữ liệu đã được tải trong vòng 1 ngày qua
            if (datetime.now() - self.trend_cache[cache_key]['timestamp']).total_seconds() < 86400:  # 1 ngày
                return self.trend_cache[cache_key]['data']
        
        try:
            logger.info(f"Lấy dữ liệu xu hướng từ Google Trends cho {symbol}, timeframe={timeframe}")
            self.pytrends.build_payload([symbol], cat=0, timeframe=timeframe, geo='', gprop='')
            trend_data = self.pytrends.interest_over_time()
            
            if not trend_data.empty:
                # Lọc dữ liệu
                trend_data = trend_data.drop(columns=['isPartial'])
                
                # Lưu vào cache
                self.trend_cache[cache_key] = {
                    'data': trend_data,
                    'timestamp': datetime.now()
                }
                
                logger.info(f"Đã lấy {len(trend_data)} dòng dữ liệu xu hướng cho {symbol}")
                return trend_data
            else:
                logger.warning(f"Không có dữ liệu xu hướng cho {symbol}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu xu hướng cho {symbol}: {e}")
            return pd.DataFrame()
    
    def get_economic_indicators(self):
        """
        Lấy các chỉ số kinh tế từ FRED API (nếu có) hoặc Yahoo Finance
        
        Returns:
            dict: Dữ liệu các chỉ số kinh tế
        """
        # Kiểm tra cache
        if 'economic_data' in self.economic_data_cache:
            # Chỉ sử dụng cache nếu dữ liệu đã được tải trong vòng 1 ngày qua
            if (datetime.now() - self.economic_data_cache['timestamp']).total_seconds() < 86400:  # 1 ngày
                return self.economic_data_cache['data']
        
        try:
            logger.info("Lấy dữ liệu chỉ số kinh tế")
            
            # Sử dụng Yahoo Finance để lấy dữ liệu chỉ số
            indicators = {
                '^DJI': 'Dow Jones Industrial Average',
                '^GSPC': 'S&P 500',
                '^IXIC': 'NASDAQ Composite',
                '^VIX': 'CBOE Volatility Index',
                '^TNX': '10-Year Treasury Yield',
                'DX-Y.NYB': 'US Dollar Index'
            }
            
            economic_data = {}
            
            for symbol, name in indicators.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="5d")
                    if not hist.empty:
                        latest_value = hist['Close'].iloc[-1]
                        prev_value = hist['Close'].iloc[-2] if len(hist) > 1 else None
                        change = latest_value - prev_value if prev_value else 0
                        change_percent = (change / prev_value * 100) if prev_value else 0
                        
                        economic_data[name] = {
                            'value': latest_value,
                            'change': change,
                            'change_percent': change_percent,
                            'last_updated': hist.index[-1].strftime('%Y-%m-%d')
                        }
                except Exception as e:
                    logger.warning(f"Không thể lấy dữ liệu cho {name} ({symbol}): {e}")
            
            # Lưu vào cache
            self.economic_data_cache = {
                'data': economic_data,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Đã lấy {len(economic_data)} chỉ số kinh tế")
            return economic_data
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu chỉ số kinh tế: {e}")
            return {}
    
    def get_enhanced_stock_data(self, symbol):
        """
        Lấy dữ liệu cổ phiếu đã được cải thiện từ nhiều nguồn
        
        Args:
            symbol (str): Mã cổ phiếu
            
        Returns:
            dict: Dữ liệu cổ phiếu đã được cải thiện
        """
        try:
            logger.info(f"Lấy dữ liệu cổ phiếu đã được cải thiện cho {symbol}")
            
            # Khởi tạo kết quả
            result = {}
            
            # 1. Lấy dữ liệu intraday từ Yahoo Finance
            intraday_data = self.get_stock_data_from_yfinance(symbol, period="5d", interval="5m")
            if not intraday_data.empty:
                result['intraday_data'] = intraday_data
            
            # 2. Lấy dữ liệu daily
            daily_data = self.get_stock_data_from_yfinance(symbol, period="2y", interval="1d")
            if not daily_data.empty:
                result['daily_data'] = daily_data
            
            # 3. Lấy dữ liệu tài chính cơ bản
            financial_data = self.get_company_financial_data(symbol)
            if financial_data:
                result['financial_data'] = financial_data
            
            # 4. Lấy tin tức
            news = self.get_stock_news(symbol, limit=5)
            if news:
                result['news'] = news
            
            # 5. Lấy dữ liệu xu hướng
            if self.trend_enabled:
                trends = self.get_google_trends(symbol)
                if not trends.empty:
                    # Chuyển DataFrame thành dict để dễ dàng lưu trữ
                    result['trends'] = trends.reset_index().to_dict(orient='records')
            
            # 6. Lấy chỉ số kinh tế chung
            economic_data = self.get_economic_indicators()
            if economic_data:
                result['economic_indicators'] = economic_data
            
            # 7. Xử lý outliers và missing values
            if 'intraday_data' in result:
                result['intraday_data'] = self._clean_data(result['intraday_data'])
            
            if 'daily_data' in result:
                result['daily_data'] = self._clean_data(result['daily_data'])
            
            # 8. Thêm các chỉ báo kỹ thuật
            if 'daily_data' in result:
                result['daily_data'] = self._add_technical_indicators(result['daily_data'])
            
            if 'intraday_data' in result:
                result['intraday_data'] = self._add_technical_indicators(result['intraday_data'])
            
            # 9. Mã hóa tin tức thành sentiment score
            if 'news' in result:
                result['news_sentiment'] = self._calculate_news_sentiment(result['news'])
            
            logger.info(f"Đã hoàn tất dữ liệu cải thiện cho {symbol}")
            return result
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu cải thiện cho {symbol}: {e}")
            return {}
    
    def _clean_data(self, df):
        """
        Làm sạch dữ liệu: xử lý outliers và missing values
        
        Args:
            df (DataFrame): DataFrame cần làm sạch
            
        Returns:
            DataFrame: DataFrame đã được làm sạch
        """
        try:
            # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
            df_clean = df.copy()
            
            # 1. Xử lý missing values
            # Xử lý cột volume (thay bằng giá trị median)
            if 'volume' in df_clean.columns and df_clean['volume'].isnull().any():
                median_volume = df_clean['volume'].median()
                df_clean['volume'].fillna(median_volume, inplace=True)
            
            # Xử lý cột OHLC (forward fill)
            for col in ['open', 'high', 'low', 'close']:
                if col in df_clean.columns and df_clean[col].isnull().any():
                    df_clean[col].fillna(method='ffill', inplace=True)
            
            # 2. Xử lý outliers bằng phương pháp Capping
            # Chỉ xử lý với dữ liệu cần thiết
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df_clean.columns:
                    # Tính toán Q1, Q3, và IQR
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Xác định giới hạn
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    
                    # Cắt giới hạn
                    df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
                    df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
            
            return df_clean
        except Exception as e:
            logger.error(f"Lỗi khi làm sạch dữ liệu: {e}")
            return df
    
    def _add_technical_indicators(self, df):
        """
        Thêm các chỉ báo kỹ thuật vào DataFrame
        
        Args:
            df (DataFrame): DataFrame cần thêm chỉ báo
            
        Returns:
            DataFrame: DataFrame đã được thêm chỉ báo
        """
        try:
            # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
            df_tech = df.copy()
            
            # Đảm bảo cột date là index (nếu không phải)
            if 'date' in df_tech.columns:
                df_tech.set_index('date', inplace=True)
            
            # 1. RSI (Relative Strength Index)
            df_tech['rsi_14'] = self._calculate_rsi(df_tech['close'], window=14)
            
            # 2. MACD (Moving Average Convergence Divergence)
            df_tech['macd'], df_tech['macd_signal'], df_tech['macd_hist'] = self._calculate_macd(df_tech['close'])
            
            # 3. SMA (Simple Moving Average)
            df_tech['sma_20'] = df_tech['close'].rolling(window=20).mean()
            df_tech['sma_50'] = df_tech['close'].rolling(window=50).mean()
            df_tech['sma_200'] = df_tech['close'].rolling(window=200).mean()
            
            # 4. EMA (Exponential Moving Average)
            df_tech['ema_9'] = df_tech['close'].ewm(span=9, adjust=False).mean()
            df_tech['ema_12'] = df_tech['close'].ewm(span=12, adjust=False).mean()
            df_tech['ema_26'] = df_tech['close'].ewm(span=26, adjust=False).mean()
            
            # 5. Bollinger Bands
            df_tech['bb_middle'] = df_tech['close'].rolling(window=20).mean()
            df_tech['bb_std'] = df_tech['close'].rolling(window=20).std()
            df_tech['bb_upper'] = df_tech['bb_middle'] + (df_tech['bb_std'] * 2)
            df_tech['bb_lower'] = df_tech['bb_middle'] - (df_tech['bb_std'] * 2)
            
            # 6. ATR (Average True Range)
            df_tech['atr_14'] = self._calculate_atr(df_tech, window=14)
            
            # 7. Stochastic Oscillator
            df_tech['stoch_k'], df_tech['stoch_d'] = self._calculate_stochastic(df_tech, k_period=14, d_period=3)
            
            # 8. On-Balance Volume (OBV)
            df_tech['obv'] = self._calculate_obv(df_tech)
            
            # 9. Price Rate of Change (ROC)
            df_tech['roc_10'] = self._calculate_roc(df_tech['close'], window=10)
            
            # 10. Average Directional Index (ADX)
            df_tech['adx_14'] = self._calculate_adx(df_tech, window=14)
            
            # Reset index nếu date là index
            if 'date' not in df_tech.columns:
                df_tech.reset_index(inplace=True)
            
            return df_tech
        except Exception as e:
            logger.error(f"Lỗi khi thêm chỉ báo kỹ thuật: {e}")
            return df
    
    def _calculate_rsi(self, prices, window=14):
        """
        Tính RSI (Relative Strength Index)
        
        Args:
            prices (Series): Giá đóng cửa
            window (int): Chu kỳ RSI
            
        Returns:
            Series: RSI
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """
        Tính MACD (Moving Average Convergence Divergence)
        
        Args:
            prices (Series): Giá đóng cửa
            fast (int): Chu kỳ EMA nhanh
            slow (int): Chu kỳ EMA chậm
            signal (int): Chu kỳ đường signal
            
        Returns:
            tuple: (MACD, Signal, Histogram)
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    def _calculate_atr(self, df, window=14):
        """
        Tính ATR (Average True Range)
        
        Args:
            df (DataFrame): DataFrame với cột 'high', 'low', 'close'
            window (int): Chu kỳ ATR
            
        Returns:
            Series: ATR
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def _calculate_stochastic(self, df, k_period=14, d_period=3):
        """
        Tính Stochastic Oscillator
        
        Args:
            df (DataFrame): DataFrame với cột 'high', 'low', 'close'
            k_period (int): Chu kỳ %K
            d_period (int): Chu kỳ %D
            
        Returns:
            tuple: (%K, %D)
        """
        high_roll = df['high'].rolling(k_period).max()
        low_roll = df['low'].rolling(k_period).min()
        
        # Stochastic %K
        k = 100 * ((df['close'] - low_roll) / (high_roll - low_roll))
        
        # Stochastic %D
        d = k.rolling(d_period).mean()
        
        return k, d
    
    def _calculate_obv(self, df):
        """
        Tính On-Balance Volume (OBV)
        
        Args:
            df (DataFrame): DataFrame với cột 'close', 'volume'
            
        Returns:
            Series: OBV
        """
        close_diff = df['close'].diff()
        
        obv = pd.Series(index=df.index)
        obv.iloc[0] = 0
        
        for i in range(1, len(df)):
            if close_diff.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif close_diff.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_roc(self, prices, window=10):
        """
        Tính Price Rate of Change (ROC)
        
        Args:
            prices (Series): Giá đóng cửa
            window (int): Chu kỳ ROC
            
        Returns:
            Series: ROC
        """
        roc = ((prices - prices.shift(window)) / prices.shift(window)) * 100
        return roc
    
    def _calculate_adx(self, df, window=14):
        """
        Tính Average Directional Index (ADX)
        
        Args:
            df (DataFrame): DataFrame với cột 'high', 'low', 'close'
            window (int): Chu kỳ ADX
            
        Returns:
            Series: ADX
        """
        # Tính true range
        df = df.copy()
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift())
        df['tr2'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        df['atr'] = df['tr'].rolling(window).mean()
        
        # Plus Directional Movement (+DM)
        df['h_diff'] = df['high'] - df['high'].shift()
        df['l_diff'] = df['low'].shift() - df['low']
        df['+dm'] = np.where((df['h_diff'] > df['l_diff']) & (df['h_diff'] > 0), df['h_diff'], 0)
        df['-dm'] = np.where((df['l_diff'] > df['h_diff']) & (df['l_diff'] > 0), df['l_diff'], 0)
        
        # Directional Indicators
        df['+di'] = 100 * (df['+dm'].rolling(window).mean() / df['atr'])
        df['-di'] = 100 * (df['-dm'].rolling(window).mean() / df['atr'])
        
        # Directional Movement Index
        df['di_diff'] = abs(df['+di'] - df['-di'])
        df['di_sum'] = df['+di'] + df['-di']
        df['dx'] = 100 * (df['di_diff'] / df['di_sum'])
        
        # Average Directional Index
        df['adx'] = df['dx'].rolling(window).mean()
        
        return df['adx']
    
    def _calculate_news_sentiment(self, news_list):
        """
        Tính điểm sentiment từ tin tức bằng phương pháp đơn giản
        
        Args:
            news_list (list): Danh sách các tin tức
            
        Returns:
            float: Điểm sentiment từ -1 (rất tiêu cực) đến 1 (rất tích cực)
        """
        try:
            if not news_list:
                return 0.0
            
            # Danh sách từ tích cực và tiêu cực (đơn giản)
            positive_words = ['up', 'rise', 'rising', 'grew', 'growth', 'gain', 'gains', 'positive', 'profit', 
                              'increase', 'increased', 'increases', 'higher', 'improved', 'strong', 'stronger',
                              'success', 'successful', 'bullish', 'outperform', 'buy', 'opportunity']
            
            negative_words = ['down', 'fall', 'falling', 'fell', 'drop', 'dropped', 'decline', 'declined', 
                              'decrease', 'decreased', 'lower', 'weak', 'weaker', 'loss', 'losses', 'negative',
                              'bearish', 'underperform', 'sell', 'warning', 'risk', 'recession']
            
            # Tính toán sentiment
            total_sentiment = 0.0
            
            for news in news_list:
                title = news.get('title', '').lower()
                summary = news.get('summary', '').lower()
                
                # Kết hợp tiêu đề và tóm tắt
                text = f"{title} {summary}"
                
                # Đếm số từ tích cực và tiêu cực
                positive_count = sum(1 for word in positive_words if word in text)
                negative_count = sum(1 for word in negative_words if word in text)
                
                # Tính điểm cho tin tức này
                if positive_count == 0 and negative_count == 0:
                    news_sentiment = 0.0
                else:
                    news_sentiment = (positive_count - negative_count) / (positive_count + negative_count)
                
                total_sentiment += news_sentiment
            
            # Lấy trung bình
            avg_sentiment = total_sentiment / len(news_list)
            
            return avg_sentiment
        except Exception as e:
            logger.error(f"Lỗi khi tính sentiment tin tức: {e}")
            return 0.0
    
    def get_batch_data_for_all_stocks(self, symbols=None):
        """
        Lấy dữ liệu cho tất cả cổ phiếu
        
        Args:
            symbols (list, optional): Danh sách các mã cổ phiếu. Nếu None, lấy từ file cấu hình.
            
        Returns:
            dict: Dữ liệu cho tất cả cổ phiếu
        """
        try:
            # Nếu không có danh sách symbols, lấy từ file cấu hình
            if symbols is None:
                stocks_config_path = os.path.join(os.path.dirname(self.config_path), "stocks.json")
                with open(stocks_config_path, 'r') as f:
                    stocks_config = json.load(f)
                
                symbols = [stock['symbol'] for stock in stocks_config['stocks'] if stock['enabled']]
            
            logger.info(f"Lấy dữ liệu cho {len(symbols)} cổ phiếu")
            
            results = {}
            
            for symbol in symbols:
                try:
                    results[symbol] = self.get_enhanced_stock_data(symbol)
                except Exception as e:
                    logger.error(f"Lỗi khi lấy dữ liệu cho {symbol}: {e}")
                    results[symbol] = {'error': str(e)}
            
            logger.info(f"Đã hoàn tất lấy dữ liệu cho {len(results)} cổ phiếu")
            return results
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu hàng loạt: {e}")
            return {}
    
    def save_data_to_disk(self, data, filename=None):
        """
        Lưu dữ liệu vào đĩa
        
        Args:
            data (dict): Dữ liệu cần lưu
            filename (str, optional): Tên file. Nếu None, sử dụng timestamp.
            
        Returns:
            str: Đường dẫn đến file đã lưu
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"market_data_{timestamp}.json"
            
            # Xử lý DataFrame (chuyển thành dict)
            processed_data = {}
            
            for symbol, stock_data in data.items():
                processed_data[symbol] = {}
                
                for key, value in stock_data.items():
                    if isinstance(value, pd.DataFrame):
                        processed_data[symbol][key] = value.to_dict(orient='records')
                    else:
                        processed_data[symbol][key] = value
            
            # Lưu file
            file_path = os.path.join(self.raw_data_dir, filename)
            
            with open(file_path, 'w') as f:
                json.dump(processed_data, f)
            
            logger.info(f"Đã lưu dữ liệu vào {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Lỗi khi lưu dữ liệu vào đĩa: {e}")
            return None
        