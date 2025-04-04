import datetime
import json
import os
import pytz
import requests
from dateutil import parser
from datetime import datetime, time, timedelta
import pandas as pd
from utils.logger_config import logger

class MarketHours:
    def __init__(self, config_path="../config/system_config.json"):
        """
        Khởi tạo quản lý giờ thị trường với tự động điều chỉnh DST
        
        Args:
            config_path (str): Đường dẫn đến file cấu hình
        """
        # Đọc cấu hình
        self.config_path = config_path
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        self.market_config = self.config['market']
        self.exchange = self.market_config['exchange']
        
        # Lấy múi giờ thị trường
        self.market_timezone = pytz.timezone(self.market_config['timezone'])
        
        # Khởi tạo giờ giao dịch
        self._init_trading_hours()
        
        # Cache lịch ngày nghỉ
        self.market_holidays = self._get_market_holidays()
        self.last_holiday_update = datetime.now()
        
        logger.info(f"MarketHours đã được khởi tạo với múi giờ {self.market_timezone}")
    
    def _init_trading_hours(self):
        """Khởi tạo các mốc thời gian giao dịch từ cấu hình"""
        self.pre_market_start = parser.parse(self.market_config['pre_market_start']).time()
        self.market_open = parser.parse(self.market_config['market_open']).time()
        self.market_close = parser.parse(self.market_config['market_close']).time()
        self.after_market_end = parser.parse(self.market_config['after_market_end']).time()
    
    def _get_market_holidays(self):
        """
        Lấy danh sách các ngày nghỉ lễ của thị trường từ Polygon API
        
        Returns:
            list: Danh sách các ngày nghỉ lễ trong năm hiện tại
        """
        try:
            # Sử dụng tính năng Polygon Market Holidays
            current_year = datetime.now().year
            holidays = []
            
            # Thử lấy từ Polygon API
            try:
                with open('./config/api_keys.json', 'r') as f:
                    api_keys = json.load(f)
                
                polygon_key = api_keys['polygon']['api_key']
                url = f"https://api.polygon.io/v1/marketstatus/upcoming?apiKey={polygon_key}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    holidays_data = response.json()
                    for holiday in holidays_data:
                        date_str = holiday.get('date')
                        if date_str:
                            holiday_date = parser.parse(date_str).date()
                            holidays.append(holiday_date)
                    logger.info(f"Đã lấy {len(holidays)} ngày nghỉ lễ từ Polygon API")
                    return holidays
            except Exception as e:
                logger.warning(f"Không thể lấy ngày nghỉ từ Polygon API: {e}")
            
            # Danh sách ngày nghỉ cố định cho năm hiện tại (backup)
            standard_holidays = [
                datetime(current_year, 1, 1),  # Năm mới
                datetime(current_year, 1, 15),  # Martin Luther King Jr. Day (thứ 2 thứ 3 của tháng 1)
                datetime(current_year, 2, 19),  # President's Day (thứ 2 thứ 3 của tháng 2)
                datetime(current_year, 3, 29),  # Good Friday
                datetime(current_year, 5, 27),  # Memorial Day (thứ 2 cuối cùng của tháng 5)
                datetime(current_year, 6, 19),  # Juneteenth
                datetime(current_year, 7, 4),   # Independence Day
                datetime(current_year, 9, 2),   # Labor Day (thứ 2 đầu tiên của tháng 9)
                datetime(current_year, 11, 28), # Thanksgiving (thứ 5 thứ 4 của tháng 11)
                datetime(current_year, 12, 25)  # Giáng sinh
            ]
            
            holidays = [holiday.date() for holiday in standard_holidays]
            logger.info(f"Sử dụng danh sách ngày nghỉ cố định: {len(holidays)} ngày")
            return holidays
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách ngày nghỉ: {e}")
            return []
    
    def update_holidays_if_needed(self):
        """Cập nhật danh sách ngày nghỉ nếu đã quá 7 ngày kể từ lần cập nhật cuối"""
        if (datetime.now() - self.last_holiday_update).days >= 7:
            self.market_holidays = self._get_market_holidays()
            self.last_holiday_update = datetime.now()
            logger.info("Đã cập nhật danh sách ngày nghỉ thị trường")
    
    def get_current_market_time(self):
        """
        Lấy thời gian hiện tại theo múi giờ của thị trường
        
        Returns:
            datetime: Thời gian hiện tại theo múi giờ thị trường
        """
        return datetime.now(self.market_timezone)
    
    def is_market_open(self, check_time=None):
        """
        Kiểm tra xem thị trường có đang mở cửa hay không
        
        Args:
            check_time (datetime, optional): Thời gian cần kiểm tra. Mặc định là thời gian hiện tại.
        
        Returns:
            bool: True nếu thị trường đang mở cửa, False nếu ngược lại
        """
        if check_time is None:
            check_time = self.get_current_market_time()
        
        # Kiểm tra xem có phải ngày nghỉ không
        if self.is_holiday(check_time):
            return False
        
        # Kiểm tra xem có phải cuối tuần không
        if check_time.weekday() >= 5:  # 5 = Thứ 7, 6 = Chủ nhật
            return False
        
        # Kiểm tra giờ giao dịch
        current_time = check_time.time()
        return self.market_open <= current_time < self.market_close
    
    def is_pre_market(self, check_time=None):
        """
        Kiểm tra xem có đang trong giờ pre-market hay không
        
        Args:
            check_time (datetime, optional): Thời gian cần kiểm tra. Mặc định là thời gian hiện tại.
        
        Returns:
            bool: True nếu đang trong giờ pre-market, False nếu ngược lại
        """
        if check_time is None:
            check_time = self.get_current_market_time()
        
        # Kiểm tra xem có phải ngày nghỉ không
        if self.is_holiday(check_time):
            return False
        
        # Kiểm tra xem có phải cuối tuần không
        if check_time.weekday() >= 5:
            return False
        
        # Kiểm tra giờ pre-market
        current_time = check_time.time()
        return self.pre_market_start <= current_time < self.market_open
    
    def is_after_market(self, check_time=None):
        """
        Kiểm tra xem có đang trong giờ after-market hay không
        
        Args:
            check_time (datetime, optional): Thời gian cần kiểm tra. Mặc định là thời gian hiện tại.
        
        Returns:
            bool: True nếu đang trong giờ after-market, False nếu ngược lại
        """
        if check_time is None:
            check_time = self.get_current_market_time()
        
        # Kiểm tra xem có phải ngày nghỉ không
        if self.is_holiday(check_time):
            return False
        
        # Kiểm tra xem có phải cuối tuần không
        if check_time.weekday() >= 5:
            return False
        
        # Kiểm tra giờ after-market
        current_time = check_time.time()
        return self.market_close <= current_time < self.after_market_end
    
    def is_trading_hours(self, check_time=None):
        """
        Kiểm tra xem có đang trong giờ giao dịch mở rộng (7:00-16:00, thứ 2-6) hay không
        
        Args:
            check_time (datetime, optional): Thời gian cần kiểm tra. Mặc định là thời gian hiện tại.
        
        Returns:
            bool: True nếu đang trong giờ giao dịch mở rộng, False nếu ngược lại
        """
        if check_time is None:
            check_time = self.get_current_market_time()
        
        # Kiểm tra ngày (chỉ thứ 2-6)
        if check_time.weekday() >= 5:  # 5 = Thứ 7, 6 = Chủ nhật
            return False
        
        # Kiểm tra nếu là ngày nghỉ lễ
        if self.is_holiday(check_time):
            return False
        
        # Kiểm tra giờ (7:00-16:00)
        current_time = check_time.time()
        trading_start = time(7, 0)  # 7:00 AM
        trading_end = time(16, 0)   # 4:00 PM
        
        return trading_start <= current_time < trading_end
    
    def is_holiday(self, check_time=None):
        """
        Kiểm tra xem ngày hiện tại có phải là ngày nghỉ lễ hay không
        
        Args:
            check_time (datetime, optional): Thời gian cần kiểm tra. Mặc định là thời gian hiện tại.
        
        Returns:
            bool: True nếu là ngày nghỉ lễ, False nếu ngược lại
        """
        self.update_holidays_if_needed()
        
        if check_time is None:
            check_time = self.get_current_market_time()
        
        check_date = check_time.date()
        return check_date in self.market_holidays
    
    def seconds_to_next_market_open(self, from_time=None):
        """
        Tính số giây đến lần mở cửa thị trường tiếp theo
        
        Args:
            from_time (datetime, optional): Thời gian bắt đầu tính. Mặc định là thời gian hiện tại.
        
        Returns:
            int: Số giây đến lần mở cửa thị trường tiếp theo
        """
        if from_time is None:
            from_time = self.get_current_market_time()
        
        # Nếu đang trong giờ giao dịch, trả về 0
        if self.is_market_open(from_time):
            return 0
        
        # Tính toán thời gian mở cửa tiếp theo
        target_date = from_time.date()
        target_time = datetime.combine(target_date, self.market_open)
        target_datetime = self.market_timezone.localize(target_time)
        
        # Nếu đã qua giờ mở cửa của ngày hôm nay, tính cho ngày mai
        if from_time.time() >= self.market_open:
            target_datetime += timedelta(days=1)
        
        # Kiểm tra nếu target_datetime rơi vào cuối tuần hoặc ngày lễ
        while target_datetime.weekday() >= 5 or target_datetime.date() in self.market_holidays:
            target_datetime += timedelta(days=1)
            target_datetime = datetime.combine(target_datetime.date(), self.market_open)
            target_datetime = self.market_timezone.localize(target_datetime)
        
        # Tính số giây còn lại
        delta = target_datetime - from_time
        return int(delta.total_seconds())
    
    def seconds_to_next_pre_market(self, from_time=None):
        """
        Tính số giây đến lần mở cửa pre-market tiếp theo
        
        Args:
            from_time (datetime, optional): Thời gian bắt đầu tính. Mặc định là thời gian hiện tại.
        
        Returns:
            int: Số giây đến lần mở cửa pre-market tiếp theo
        """
        if from_time is None:
            from_time = self.get_current_market_time()
        
        # Nếu đang trong giờ pre-market, trả về 0
        if self.is_pre_market(from_time):
            return 0
        
        # Tính toán thời gian pre-market tiếp theo
        target_date = from_time.date()
        target_time = datetime.combine(target_date, self.pre_market_start)
        target_datetime = self.market_timezone.localize(target_time)
        
        # Nếu đã qua giờ pre-market của ngày hôm nay, tính cho ngày mai
        if from_time.time() >= self.pre_market_start:
            target_datetime += timedelta(days=1)
        
        # Kiểm tra nếu target_datetime rơi vào cuối tuần hoặc ngày lễ
        while target_datetime.weekday() >= 5 or target_datetime.date() in self.market_holidays:
            target_datetime += timedelta(days=1)
            target_datetime = datetime.combine(target_datetime.date(), self.pre_market_start)
            target_datetime = self.market_timezone.localize(target_datetime)
        
        # Tính số giây còn lại
        delta = target_datetime - from_time
        return int(delta.total_seconds())
    
    def get_next_trading_day(self, from_date=None):
        """
        Lấy ngày giao dịch tiếp theo
        
        Args:
            from_date (date, optional): Ngày bắt đầu tính. Mặc định là ngày hiện tại.
        
        Returns:
            date: Ngày giao dịch tiếp theo
        """
        if from_date is None:
            from_date = self.get_current_market_time().date()
        
        next_day = from_date + timedelta(days=1)
        
        # Kiểm tra nếu next_day rơi vào cuối tuần hoặc ngày lễ
        while next_day.weekday() >= 5 or next_day in self.market_holidays:
            next_day += timedelta(days=1)
        
        return next_day
    
    def get_market_status(self):
        """
        Lấy trạng thái hiện tại của thị trường
        
        Returns:
            dict: Trạng thái thị trường với các thông tin chi tiết
        """
        now = self.get_current_market_time()
        
        status = {
            "time": now.strftime('%Y-%m-%d %H:%M:%S %Z'),
            "is_holiday": self.is_holiday(now),
            "is_weekend": now.weekday() >= 5,
            "is_pre_market": self.is_pre_market(now),
            "is_market_open": self.is_market_open(now),
            "is_after_market": self.is_after_market(now),
            "next_market_open_seconds": self.seconds_to_next_market_open(now),
            "next_pre_market_seconds": self.seconds_to_next_pre_market(now),
            "next_trading_day": self.get_next_trading_day(now.date()).strftime('%Y-%m-%d')
        }
        
        return status

if __name__ == "__main__":
    # Test module
    market_hours = MarketHours()
    status = market_hours.get_market_status()
    
    print("\n=== Trạng thái thị trường ===")
    for key, value in status.items():
        print(f"{key}: {value}")
