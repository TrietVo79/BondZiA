import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pytz
from utils.logger_config import logger
import traceback

class HistoricalDataLoader:
    """
    Lớp tải và xử lý dữ liệu lịch sử cho việc huấn luyện mô hình
    """
    
    def __init__(self, config_path="../config/system_config.json"):
        """
        Khởi tạo DataLoader
        
        Args:
            config_path (str): Đường dẫn đến file cấu hình
        """
        # Đọc cấu hình
        self.config_path = config_path
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Đọc API key Polygon
        api_keys_path = os.path.join(os.path.dirname(self.config_path), "api_keys.json")
        with open(api_keys_path, 'r') as f:
            api_keys = json.load(f)
        
        self.polygon_api_key = api_keys['polygon']['api_key']
        self.polygon_plan = api_keys['polygon']['plan']
        
        # Đọc danh sách cổ phiếu
        stocks_path = os.path.join(os.path.dirname(self.config_path), "stocks.json")
        with open(stocks_path, 'r') as f:
            stocks_config = json.load(f)
        
        self.stocks = stocks_config['stocks']
        
        # Đọc cấu hình dữ liệu
        self.data_config = self.config['data']
        
        # Đảm bảo thư mục dữ liệu tồn tại
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    "data")
        self.raw_data_dir = os.path.join(self.data_dir, "raw")
        self.processed_data_dir = os.path.join(self.data_dir, "processed")
        
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Cấu hình timezone
        self.market_timezone = pytz.timezone(self.config['market']['timezone'])
    
    def _get_polygon_bars(self, symbol, multiplier, timespan, from_date, to_date, limit=5000):
        """
        Lấy dữ liệu bars từ Polygon API
        
        Args:
            symbol (str): Mã cổ phiếu
            multiplier (int): Bội số thời gian
            timespan (str): Đơn vị thời gian ('minute', 'hour', 'day', 'week', 'month', 'quarter', 'year')
            from_date (str): Ngày bắt đầu (YYYY-MM-DD)
            to_date (str): Ngày kết thúc (YYYY-MM-DD)
            limit (int): Số lượng kết quả tối đa mỗi trang
            
        Returns:
            list: Danh sách các bars
        """
        all_results = []
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        
        params = {
            "apiKey": self.polygon_api_key,
            "limit": limit
        }
        
        try:
            # Lấy trang đầu tiên
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'OK' and data.get('results'):
                all_results.extend(data['results'])
                
                # Xử lý phân trang nếu có
                while data.get('next_url'):
                    next_url = data['next_url']
                    next_url += f"&apiKey={self.polygon_api_key}"
                    
                    # Đợi một chút để không vượt quá rate limit
                    time.sleep(0.2)
                    
                    response = requests.get(next_url)
                    response.raise_for_status()
                    data = response.json()
                    
                    if data.get('status') == 'OK' and data.get('results'):
                        all_results.extend(data['results'])
                    else:
                        break
            
            return all_results
        except Exception as e:
            print(f"Lỗi khi lấy dữ liệu từ Polygon: {str(e)}")
            return []
            
    def _get_polygon_monthly(self, symbol, from_date, to_date):
        """
        Lấy dữ liệu monthly bars trực tiếp từ Polygon API
        
        Args:
            symbol (str): Mã cổ phiếu
            from_date (str): Ngày bắt đầu (YYYY-MM-DD)
            to_date (str): Ngày kết thúc (YYYY-MM-DD)
            
        Returns:
            list: Danh sách các monthly bars
        """
        return self._get_polygon_bars(symbol, 1, 'month', from_date, to_date)
    
    def _convert_bars_to_dataframe(self, bars):
        """
        Chuyển đổi danh sách bars thành DataFrame
        
        Args:
            bars (list): Danh sách các bars từ Polygon API
            
        Returns:
            DataFrame: DataFrame chứa dữ liệu bars
        """
        if not bars:
            return pd.DataFrame()
        
        # Tạo DataFrame từ danh sách bars
        df = pd.DataFrame(bars)
        
        # Đổi tên cột
        column_mapping = {
            'v': 'volume',
            'o': 'open',
            'c': 'close',
            'h': 'high',
            'l': 'low',
            't': 'timestamp',
            'n': 'transactions'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Chuyển đổi timestamp (milliseconds từ epoch) sang datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Đặt timestamp làm index
        df = df.set_index('timestamp')
        
        # Sắp xếp theo thời gian
        df = df.sort_index()
        
        return df
    
    def _convert_daily_to_monthly(self, df_daily):
        """
        Chuyển đổi dữ liệu ngày thành dữ liệu tháng
        
        Args:
            df_daily (DataFrame): DataFrame dữ liệu theo ngày
            
        Returns:
            DataFrame: DataFrame dữ liệu theo tháng
        """
        try:
            # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
            df = df_daily.copy()
            
            # Đặt lại index để làm việc với cột datetime
            df = df.reset_index()
            
            # Tạo cột tháng
            df['year_month'] = df['timestamp'].dt.to_period('M')
            
            # Nhóm theo tháng và tính giá trị OHLCV
            monthly_df = df.groupby('year_month').agg({
                'open': 'first',     # Giá mở cửa đầu tháng
                'high': 'max',       # Giá cao nhất trong tháng
                'low': 'min',        # Giá thấp nhất trong tháng
                'close': 'last',     # Giá đóng cửa cuối tháng
                'volume': 'sum',     # Tổng khối lượng trong tháng
                'transactions': 'sum' # Tổng số giao dịch
            })
            
            # Chuyển đổi index từ PeriodIndex về DatetimeIndex
            monthly_df.index = monthly_df.index.to_timestamp()
            
            return monthly_df
        except Exception as e:
            print(f"Lỗi khi chuyển đổi dữ liệu ngày thành tháng: {str(e)}")
            return pd.DataFrame()
        
    def _calculate_technical_indicators(self, df):
        """
        Tính toán các chỉ báo kỹ thuật
        
        Args:
            df (DataFrame): DataFrame giá cổ phiếu
            
        Returns:
            DataFrame: DataFrame với các chỉ báo kỹ thuật bổ sung
        """
        try:
            # Sao chép DataFrame để không thay đổi dữ liệu gốc
            df_tech = df.copy()
            
            # Tính RSI-14
            delta = df_tech['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            df_tech['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Tính SMA-20
            df_tech['sma_20'] = df_tech['close'].rolling(window=20).mean()
            
            # Tính EMA-9
            df_tech['ema_9'] = df_tech['close'].ewm(span=9, adjust=False).mean()
            
            # Tính MACD
            ema_12 = df_tech['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df_tech['close'].ewm(span=26, adjust=False).mean()
            df_tech['macd'] = ema_12 - ema_26
            df_tech['macd_signal'] = df_tech['macd'].ewm(span=9, adjust=False).mean()
            df_tech['macd_hist'] = df_tech['macd'] - df_tech['macd_signal']
            
            # Tính Bollinger Bands
            sma_20 = df_tech['close'].rolling(window=20).mean()
            std_20 = df_tech['close'].rolling(window=20).std()
            df_tech['bb_upper'] = sma_20 + (std_20 * 2)
            df_tech['bb_middle'] = sma_20
            df_tech['bb_lower'] = sma_20 - (std_20 * 2)
            
            # Đặt giá trị NaN cho các dòng đầu tiên không có đủ dữ liệu
            df_tech = df_tech.replace([np.inf, -np.inf], np.nan)
            
            # Cải tiến xử lý NaN: Thay vì loại bỏ hoàn toàn, chỉ thay thế các giá trị NaN
            df_tech = df_tech.fillna(method='ffill').fillna(method='bfill')
            
            return df_tech
        except Exception as e:
            print(f"Lỗi khi tính toán chỉ báo kỹ thuật: {str(e)}")
            return df
    
    def load_and_save_historical_data(self, symbol, years=10, save=True):
        """
        Tải dữ liệu lịch sử cho một cổ phiếu và lưu vào file
        
        Args:
            symbol (str): Mã cổ phiếu
            years (int): Số năm lấy dữ liệu về quá khứ (đã tăng từ 5 lên 10)
            save (bool): Lưu dữ liệu vào file
            
        Returns:
            DataFrame: DataFrame chứa dữ liệu lịch sử
        """
        try:
            # Tính toán khoảng thời gian
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
            
            # Lấy dữ liệu daily bars
            daily_bars = self._get_polygon_bars(symbol, 1, 'day', from_date, to_date)
            
            if not daily_bars:
                print(f"Không có dữ liệu daily bars cho {symbol}")
                return None
            
            # Chuyển đổi thành DataFrame
            df_daily = self._convert_bars_to_dataframe(daily_bars)
            
            # Tính toán chỉ báo kỹ thuật
            df_with_indicators = self._calculate_technical_indicators(df_daily)
            
            # Lưu dữ liệu
            if save:
                # Lưu dữ liệu raw
                raw_file_path = os.path.join(self.raw_data_dir, f"{symbol}_daily_{from_date}_{to_date}.csv")
                df_daily.to_csv(raw_file_path)
                
                # Lưu dữ liệu đã xử lý (bỏ dropna để giữ nhiều dữ liệu hơn)
                processed_file_path = os.path.join(self.processed_data_dir, f"{symbol}_daily_processed_{from_date}_{to_date}.csv")
                df_with_indicators.to_csv(processed_file_path)
            
            return df_with_indicators
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu lịch sử cho {symbol}: {str(e)}")
            return None
    
    def load_and_save_monthly_data(self, symbol, years=10, save=True):
        """
        Tải và lưu dữ liệu monthly cho một cổ phiếu
        
        Args:
            symbol (str): Mã cổ phiếu
            years (int): Số năm lấy dữ liệu
            save (bool): Lưu dữ liệu vào file
            
        Returns:
            DataFrame: DataFrame dữ liệu monthly đã xử lý
        """
        try:
            # Tính toán khoảng thời gian
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
            
            # Phương pháp 1: Lấy trực tiếp monthly bars từ Polygon
            monthly_bars = self._get_polygon_monthly(symbol, from_date, to_date)
            df_monthly_direct = self._convert_bars_to_dataframe(monthly_bars)
            
            # Phương pháp 2: Chuyển đổi từ dữ liệu daily
            df_daily = self.load_and_save_historical_data(symbol, years, False)
            df_monthly_from_daily = None
            
            if df_daily is not None and not df_daily.empty:
                df_monthly_from_daily = self._convert_daily_to_monthly(df_daily)
            
            # Kiểm tra và chọn nguồn dữ liệu tốt nhất
            if df_monthly_direct is not None and not df_monthly_direct.empty and len(df_monthly_direct) >= 12:
                df_monthly = df_monthly_direct
                print(f"Sử dụng dữ liệu monthly trực tiếp từ Polygon cho {symbol}: {len(df_monthly_direct)} tháng")
            elif df_monthly_from_daily is not None and not df_monthly_from_daily.empty and len(df_monthly_from_daily) >= 12:
                df_monthly = df_monthly_from_daily
                print(f"Sử dụng dữ liệu monthly từ daily data cho {symbol}: {len(df_monthly_from_daily)} tháng")
            else:
                # Chọn nguồn có nhiều dữ liệu nhất
                sources = [
                    (df_monthly_direct, "Polygon direct"),
                    (df_monthly_from_daily, "Daily conversion")
                ]
                
                valid_sources = [(df, name) for df, name in sources if df is not None and not df.empty]
                
                if not valid_sources:
                    print(f"Không có dữ liệu monthly cho {symbol} từ bất kỳ nguồn nào")
                    return None
                
                # Sắp xếp theo số lượng dữ liệu giảm dần
                valid_sources.sort(key=lambda x: len(x[0]), reverse=True)
                df_monthly, source_name = valid_sources[0]
                
                print(f"Sử dụng nguồn dữ liệu {source_name} với {len(df_monthly)} tháng cho {symbol}")
            
            # Tính toán chỉ báo kỹ thuật cho dữ liệu monthly
            df_monthly_indicators = self._calculate_technical_indicators(df_monthly)
            
            # Lưu dữ liệu
            if save and df_monthly_indicators is not None and not df_monthly_indicators.empty:
                # Lưu dữ liệu monthly đã xử lý
                processed_file_path = os.path.join(self.processed_data_dir, f"{symbol}_monthly_processed_{from_date}_{to_date}.csv")
                df_monthly_indicators.to_csv(processed_file_path)
            
            return df_monthly_indicators
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu monthly cho {symbol}: {str(e)}")
            return None
    
    def load_data_for_all_stocks(self, years=10):
        """
        Tải dữ liệu lịch sử cho tất cả cổ phiếu trong danh sách
        
        Args:
            years (int): Số năm lấy dữ liệu về quá khứ (đã tăng từ 5 lên 10)
            
        Returns:
            dict: Dictionary chứa DataFrame dữ liệu lịch sử cho mỗi cổ phiếu
        """
        all_data = {}
        
        for stock in self.stocks:
            symbol = stock['symbol']
            print(f"Tải dữ liệu daily cho {symbol}")
            
            df = self.load_and_save_historical_data(symbol, years=years)
            
            if df is not None:
                all_data[symbol] = df
                
                # Đợi một chút giữa các request để tránh rate limit
                time.sleep(0.5)
        
        print(f"Đã tải xong dữ liệu daily cho {len(all_data)}/{len(self.stocks)} cổ phiếu")
        return all_data
    
    def load_monthly_data_for_all_stocks(self, years=10):
        """
        Tải dữ liệu monthly cho tất cả cổ phiếu trong danh sách
        
        Args:
            years (int): Số năm lấy dữ liệu
            
        Returns:
            dict: Dictionary chứa DataFrame dữ liệu monthly cho mỗi cổ phiếu
        """
        all_monthly_data = {}
        
        for stock in self.stocks:
            symbol = stock['symbol']
            print(f"Tải dữ liệu monthly cho {symbol}")
            
            df = self.load_and_save_monthly_data(symbol, years=years)
            
            if df is not None and not df.empty:
                all_monthly_data[symbol] = df
                
                # Đợi một chút giữa các request để tránh rate limit
                time.sleep(0.5)
        
        print(f"Đã tải xong dữ liệu monthly cho {len(all_monthly_data)}/{len(self.stocks)} cổ phiếu")
        
        # Kiểm tra số lượng dữ liệu
        for symbol, df in all_monthly_data.items():
            print(f"{symbol} monthly data: {len(df)} dòng từ {df.index[0]} đến {df.index[-1]}")
            if len(df) < 12:  # Cảnh báo nếu có ít hơn 12 tháng dữ liệu
                print(f"CẢNH BÁO: {symbol} chỉ có {len(df)} tháng dữ liệu, có thể không đủ để huấn luyện!")
        
        return all_monthly_data


if __name__ == "__main__":
    data_loader = HistoricalDataLoader()
    
    # Tải dữ liệu daily cho tất cả cổ phiếu với 10 năm lịch sử
    print("==== BẮT ĐẦU TẢI DỮ LIỆU DAILY ====")
    all_daily_data = data_loader.load_data_for_all_stocks(years=10)
    
    # Tải dữ liệu monthly
    print("==== BẮT ĐẦU TẢI DỮ LIỆU MONTHLY ====")
    all_monthly_data = data_loader.load_monthly_data_for_all_stocks(years=10)
    
    # Kiểm tra số lượng dữ liệu cho mô hình monthly
    monthly_data_sufficient = True
    min_required = 12  # Số lượng mẫu tối thiểu cần thiết cho mô hình monthly
    
    print("==== KIỂM TRA ĐỦ DỮ LIỆU CHO MÔ HÌNH MONTHLY ====")
    for symbol, df in all_monthly_data.items():
        if len(df) < min_required:
            print(f"THIẾU DỮ LIỆU: {symbol} chỉ có {len(df)} mẫu monthly, cần ít nhất {min_required}")
            monthly_data_sufficient = False
        else:
            print(f"ĐỦ DỮ LIỆU: {symbol} có {len(df)} mẫu monthly")
    
    if monthly_data_sufficient:
        print("ĐỦ DỮ LIỆU cho tất cả các mô hình monthly")
    else:
        print("MỘT SỐ CỔ PHIẾU KHÔNG ĐỦ DỮ LIỆU cho mô hình monthly")
    
    # In thông tin tóm tắt
    print("==== THÔNG TIN TÓM TẮT ====")
    print("Dữ liệu Daily:")
    for symbol, df in all_daily_data.items():
        print(f"{symbol}: {len(df)} dòng dữ liệu từ {df.index[0].strftime('%Y-%m-%d')} đến {df.index[-1].strftime('%Y-%m-%d')}")
    
    print("Dữ liệu Monthly:")
    for symbol, df in all_monthly_data.items():
        print(f"{symbol}: {len(df)} dòng dữ liệu từ {df.index[0].strftime('%Y-%m-%d')} đến {df.index[-1].strftime('%Y-%m-%d')}")    