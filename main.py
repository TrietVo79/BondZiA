import os
import sys
import json
import time
import schedule
import threading
import pandas as pd
from datetime import datetime, timedelta
import traceback
import argparse
from utils.logger_config import logger

# Thêm thư mục hiện tại vào PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import các module
from utils.market_hours import MarketHours
from utils.data_fetcher import PolygonDataFetcher
from utils.discord_notifier import DiscordNotifier
from utils.visualization import StockVisualizer
from utils.error_handler import ErrorHandler, setup_exception_handler
from models.specific_predictors import IntradayPredictor, FiveDayPredictor, MonthlyPredictor
from evolution.hyperparameter_tuner import ModelEvolutionManager
from evolution.version_manager import VersionManager

class BondZiA:
    """Lớp chính điều khiển hệ thống BondZiA AI"""
    
    def __init__(self, config_path="config/system_config.json"):
        """
        Khởi tạo BondZiA
        
        Args:
            config_path (str): Đường dẫn đến file cấu hình
        """
        # Đặt đường dẫn tuyệt đối đến file cấu hình
        self.config_path = os.path.join(current_dir, config_path)
        
        # Đọc cấu hình
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Thiết lập logging
        self._setup_logging()
        
        logger.info("Khởi động BondZiA AI")
        
        # Đọc danh sách cổ phiếu
        stocks_path = os.path.join(os.path.dirname(self.config_path), "stocks.json")
        with open(stocks_path, 'r') as f:
            stocks_config = json.load(f)
        
        self.stocks = [stock['symbol'] for stock in stocks_config['stocks'] if stock['enabled']]
        
        # Khởi tạo các module
        self.market_hours = MarketHours(config_path=self.config_path)
        self.discord = DiscordNotifier(config_path=self.config_path)
        
        # Thông báo khởi động
        self.discord.send_system_update(
            title="BondZiA AI đang khởi động",
            message=f"Hệ thống BondZiA AI đang khởi động và sẽ theo dõi {len(self.stocks)} cổ phiếu.",
            fields=[
                {
                    "name": "Phiên bản",
                    "value": self.config['system']['version'],
                    "inline": True
                },
                {
                    "name": "Cổ phiếu theo dõi",
                    "value": ", ".join(self.stocks),
                    "inline": False
                }
            ]
        )
        
        # Khởi tạo error handler
        self.error_handler = ErrorHandler(config_path=self.config_path, discord_notifier=self.discord)
        setup_exception_handler(self.error_handler)
        
        # Khởi tạo các module khác
        self.data_fetcher = PolygonDataFetcher(config_path=self.config_path)
        self.visualizer = StockVisualizer(config_path=self.config_path)
        self.version_manager = VersionManager(config_path=self.config_path)
        
        # Khởi động watchdog
        self.error_handler.setup_watchdog()
        
        # Tình trạng hiện tại
        self.current_data = {}
        self.last_prediction_time = {}
        for symbol in self.stocks:
            self.last_prediction_time[symbol] = {
                'intraday': None,
                'five_day': None,
                'monthly': None
            }
        
        self.previous_predictions = {}  # Lưu trữ dự đoán gần nhất cho mỗi cổ phiếu

        # Trạng thái hệ thống
        self.is_running = True
        self.is_predicting = False
        self.is_evolving = False
        self.is_initial_training = False  # Thêm trạng thái huấn luyện ban đầu
        
        # Bật các tiến trình
        self.scheduler_thread = None
        
        logger.info(f"BondZiA AI Phiên bản {self.config['system']['version']} đã sẵn sàng!")
    
    def _setup_logging(self):
        """Thiết lập logging"""
        # Đường dẫn thư mục logs
        logs_dir = os.path.join(current_dir, "logs")
        system_logs_dir = os.path.join(logs_dir, "system")
        predictions_logs_dir = os.path.join(logs_dir, "predictions")
        evolution_logs_dir = os.path.join(logs_dir, "evolution")  # Thêm thư mục cho log tiến hóa
        errors_logs_dir = os.path.join(logs_dir, "errors")
        
        # Tạo các thư mục nếu chưa tồn tại
        os.makedirs(system_logs_dir, exist_ok=True)
        os.makedirs(predictions_logs_dir, exist_ok=True)
        os.makedirs(evolution_logs_dir, exist_ok=True)  # Tạo thư mục cho log tiến hóa
        os.makedirs(errors_logs_dir, exist_ok=True)
        
        # Xóa các handler mặc định
        logger.remove()
        
        # Định dạng log
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        
        # Thêm handler ghi log ra console
        logger.add(sys.stderr, format=log_format, level="INFO")
        
        # Thêm handler ghi log hệ thống
        logger.add(
            os.path.join(system_logs_dir, "bondzia_{time:YYYY-MM-DD}.log"),
            format=log_format,
            level="INFO",
            rotation="00:00",  # Mỗi ngày một file mới
            retention="30 days",  # Giữ log trong 30 ngày
            compression="zip"     # Nén file log cũ
        )
        
        # Thêm handler ghi log dự đoán (giới hạn 10MB)
        logger.add(
            os.path.join(predictions_logs_dir, "predictions_{time:YYYY-MM-DD}.log"),
            format=log_format,
            level="INFO",
            rotation="10 MB",  # Giới hạn 10MB mỗi file
            retention="30 days",
            compression="zip",
            filter=lambda record: "prediction" in record["message"].lower() or "dự đoán" in record["message"].lower()
        )
        
        # Thêm handler ghi log tiến hóa (giới hạn 10MB)
        logger.add(
            os.path.join(evolution_logs_dir, "evolution_{time:YYYY-MM-DD}.log"),
            format=log_format,
            level="INFO",
            rotation="10 MB",  # Giới hạn 10MB mỗi file
            retention="30 days",
            compression="zip",
            filter=lambda record: "evolution" in record["message"].lower() or "evolve" in record["message"].lower() or "tiến hóa" in record["message"].lower() or "huấn luyện" in record["message"].lower() or "train" in record["message"].lower()
        )
        
        # Thêm handler ghi log lỗi
        logger.add(
            os.path.join(errors_logs_dir, "errors_{time:YYYY-MM-DD}.log"),
            format=log_format,
            level="ERROR",
            rotation="10 MB",  # Giới hạn 10MB mỗi file
            retention="30 days",
            compression="zip"
        )
    
    def start(self):
        """Khởi động BondZiA"""
        try:
            # Khởi động scheduler
            self._setup_schedule()
            
            # Khởi động thread scheduler
            self.scheduler_thread = threading.Thread(target=self._run_scheduler)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            
            # Chạy lần đầu ngay khi khởi động
            self._update_data()
            
            # Kiểm tra và tạo thư mục cho các mô hình
            self._ensure_model_directories()
            
            # Kiểm tra và huấn luyện các mô hình nếu chưa tồn tại
            self._check_and_train_initial_models()
            
            # Nếu đang trong giờ giao dịch, chạy dự đoán ngay lập tức
            if self.market_hours.is_trading_hours():
                self._run_predictions()
            
            # Hiển thị thông tin thị trường
            self._display_market_info()
            
            # Kiểm tra nếu là cuối tuần, có thể tiến hành huấn luyện và tiến hóa
            current_day = datetime.now().weekday()
            if current_day >= 5:  # 5 = Thứ 7, 6 = Chủ nhật
                logger.info("Đang ở cuối tuần, kiểm tra nếu cần tiến hóa mô hình...")
                self._check_weekend_evolution()
            
            logger.info("BondZiA AI đã khởi động thành công")
            
            # Giữ chạy chương trình
            while self.is_running:
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Nhận tín hiệu thoát, đang dừng BondZiA AI...")
            self.stop()
        
        except Exception as e:
            logger.error(f"Lỗi khi khởi động BondZiA AI: {str(e)}")
            logger.error(traceback.format_exc())
            self.discord.send_system_update(
                title="Lỗi khởi động BondZiA AI",
                message=f"Hệ thống BondZiA AI gặp lỗi khi khởi động: {str(e)}",
                is_error=True
            )
            self.stop()
    
    def _ensure_model_directories(self):
        """Tạo các thư mục cần thiết cho mô hình nếu chưa tồn tại"""
        try:
            # Đường dẫn gốc cho các mô hình
            models_dir = os.path.join(current_dir, "models")
            
            # Tạo thư mục models nếu chưa tồn tại
            os.makedirs(models_dir, exist_ok=True)
            
            # Tạo các thư mục con cho từng loại mô hình
            for model_type in ['intraday', 'five_day', 'monthly']:
                model_type_dir = os.path.join(models_dir, model_type)
                os.makedirs(model_type_dir, exist_ok=True)
                logger.info(f"Đã đảm bảo thư mục mô hình tồn tại: {model_type_dir}")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tạo thư mục mô hình: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _check_and_train_initial_models(self):
        """Kiểm tra và huấn luyện các mô hình ban đầu nếu chưa tồn tại"""
        try:
            # Đếm số mô hình cần huấn luyện
            models_needed = []
            for symbol in self.stocks:
                for timeframe in ['intraday', 'five_day', 'monthly']:
                    # Xác định lớp predictor phù hợp
                    if timeframe == 'intraday':
                        predictor_class = IntradayPredictor
                    elif timeframe == 'five_day':
                        predictor_class = FiveDayPredictor
                    elif timeframe == 'monthly':
                        predictor_class = MonthlyPredictor
                    
                    # Khởi tạo predictor
                    predictor = predictor_class(symbol, config_path=self.config_path)
                    
                    # Kiểm tra xem mô hình đã tồn tại chưa
                    if not predictor.check_model_exists():
                        models_needed.append((symbol, timeframe))
            
            # Nếu có mô hình cần huấn luyện
            if models_needed:
                total_models = len(models_needed)
                logger.info(f"Phát hiện {total_models} mô hình chưa tồn tại, cần huấn luyện ban đầu")
                
                # Thông báo cho người dùng
                self.discord.send_system_update(
                    title="Huấn luyện mô hình ban đầu",
                    message=f"Hệ thống phát hiện {total_models} mô hình chưa được huấn luyện. BondZiA AI đang bắt đầu quá trình huấn luyện ban đầu. Quá trình này có thể mất vài phút đến vài giờ tùy thuộc vào lượng dữ liệu.",
                    fields=[
                        {
                            "name": "Trạng thái",
                            "value": "Đang huấn luyện",
                            "inline": True
                        },
                        {
                            "name": "Số lượng mô hình",
                            "value": str(total_models),
                            "inline": True
                        }
                    ]
                )
                
                # Đánh dấu đang trong quá trình huấn luyện ban đầu
                self.is_initial_training = True
                
                # Huấn luyện các mô hình
                success_count = 0
                for idx, (symbol, timeframe) in enumerate(models_needed):
                    logger.info(f"Đang huấn luyện mô hình [{idx+1}/{total_models}]: {symbol} - {timeframe}")
                    
                    # Cập nhật tiến độ cho người dùng sau mỗi 25% mô hình
                    if (idx+1) % max(1, total_models // 4) == 0 or idx+1 == total_models:
                        progress_percent = round((idx+1) / total_models * 100)
                        self.discord.send_system_update(
                            title="Tiến độ huấn luyện mô hình ban đầu",
                            message=f"Đã huấn luyện {idx+1}/{total_models} mô hình ({progress_percent}%)",
                            fields=[
                                {
                                    "name": "Mô hình hiện tại",
                                    "value": f"{symbol} - {timeframe}",
                                    "inline": True
                                }
                            ]
                        )
                    
                    # Huấn luyện mô hình
                    if self._train_model_if_needed(symbol, timeframe):
                        success_count += 1
                
                # Thông báo hoàn thành
                self.discord.send_system_update(
                    title="Hoàn thành huấn luyện mô hình ban đầu",
                    message=f"BondZiA AI đã hoàn thành quá trình huấn luyện ban đầu: {success_count}/{total_models} mô hình đã được huấn luyện thành công.",
                    fields=[
                        {
                            "name": "Trạng thái",
                            "value": "Hoàn thành",
                            "inline": True
                        },
                        {
                            "name": "Tỷ lệ thành công",
                            "value": f"{success_count}/{total_models} ({round(success_count/total_models*100)}%)",
                            "inline": True
                        }
                    ]
                )
                
                # Đánh dấu đã hoàn thành quá trình huấn luyện ban đầu
                self.is_initial_training = False
                logger.info(f"Hoàn thành huấn luyện ban đầu: {success_count}/{total_models} mô hình")
                
                return success_count
            else:
                logger.info("Tất cả mô hình đã tồn tại, không cần huấn luyện ban đầu")
                return 0
        
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra và huấn luyện mô hình ban đầu: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Thông báo lỗi
            self.discord.send_system_update(
                title="Lỗi huấn luyện mô hình ban đầu",
                message=f"Quá trình huấn luyện mô hình ban đầu gặp lỗi: {str(e)}",
                is_error=True
            )
            
            # Đánh dấu đã hoàn thành (dù lỗi)
            self.is_initial_training = False
            return 0
    
    def _ensure_complete_predictions(self, predictions):
        """
        Đảm bảo mỗi cổ phiếu có đủ 3 dự đoán bằng cách sử dụng dự đoán trước đó nếu cần
        
        Args:
            predictions (dict): Dictionary dự đoán hiện tại
            
        Returns:
            dict: Dictionary dự đoán đã được bổ sung
        """
        # Đảm bảo mỗi cổ phiếu có đủ 3 khung thời gian dự đoán
        for symbol in list(predictions.keys()):
            if symbol in self.previous_predictions:
                # Bổ sung dự đoán 5 ngày nếu không có
                if 'five_day' not in predictions[symbol] and 'five_day' in self.previous_predictions[symbol]:
                    logger.info(f"Bổ sung dự đoán 5 ngày cũ cho {symbol}")
                    predictions[symbol]['five_day'] = self.previous_predictions[symbol]['five_day']
                
                # Bổ sung dự đoán 1 tháng nếu không có
                if 'monthly' not in predictions[symbol] and 'monthly' in self.previous_predictions[symbol]:
                    logger.info(f"Bổ sung dự đoán 1 tháng cũ cho {symbol}")
                    predictions[symbol]['monthly'] = self.previous_predictions[symbol]['monthly']
        
        return predictions

    def stop(self):
        """Dừng BondZiA"""
        try:
            logger.info("Đang dừng BondZiA AI...")
            
            # Đặt cờ dừng
            self.is_running = False
            
            # Dừng watchdog
            self.error_handler.stop_watchdog()
            
            # Thông báo dừng
            self.discord.send_system_update(
                title="BondZiA AI đã dừng",
                message="Hệ thống BondZiA AI đã dừng hoạt động."
            )
            
            logger.info("BondZiA AI đã dừng thành công")
        except Exception as e:
            logger.error(f"Lỗi khi dừng BondZiA AI: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _setup_schedule(self):
        """Thiết lập lịch trình chạy các tác vụ"""
        # Cập nhật dữ liệu mỗi 15 phút
        schedule.every(15).minutes.do(self._update_data)
        
        # Cập nhật trạng thái thị trường mỗi giờ
        schedule.every().hour.do(self._display_market_info)
        
        # Kiểm tra và chạy dự đoán mỗi 5 phút
        schedule.every(5).minutes.do(self._check_and_predict)
        
        # Thiết lập lịch trình tiến hóa cho các ngày trong tuần (từ 16:01 đến 6:59 sáng)
        schedule.every().monday.at("16:01").do(self._check_weekday_evolution)
        schedule.every().tuesday.at("16:01").do(self._check_weekday_evolution)
        schedule.every().wednesday.at("16:01").do(self._check_weekday_evolution)
        schedule.every().thursday.at("16:01").do(self._check_weekday_evolution)
        schedule.every().friday.at("16:01").do(self._check_weekday_evolution)
        
        # Kiểm tra định kỳ trong khoảng thời gian từ 16:01 đến 6:59 sáng
        schedule.every(1).hours.do(self._check_night_evolution_window)
        
        # Thiết lập kiểm tra tiến hóa cuối tuần (mỗi 3 giờ vào thứ 7 và chủ nhật)
        schedule.every(3).hours.do(self._check_weekend_evolution)
        
        # Đánh giá hiệu suất hàng ngày
        schedule.every().day.at("20:00").do(self._run_accuracy_evaluation)

        # Sao lưu phiên bản hiện tại mỗi ngày
        schedule.every().day.at("00:00").do(self.version_manager.backup_current_state)
    
    def _check_night_evolution_window(self):
        """Kiểm tra nếu đang trong cửa sổ thời gian tiến hóa buổi tối (16:01 - 06:59)"""
        current_time = datetime.now()
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_day = current_time.weekday()

        # Nếu không phải cuối tuần (0-4 là thứ 2 đến thứ 6)
        if current_day <= 4:
            # Nếu từ 16:01 đến 23:59
            if current_hour >= 16 and (current_hour > 16 or current_minute > 0):
                logger.info("Đang trong cửa sổ tiến hóa buổi tối, kiểm tra nếu cần tiến hóa mô hình...")
                return self._evolve_models()
            # Hoặc từ 00:00 đến 06:59
            elif 0 <= current_hour < 7:
                logger.info("Đang trong cửa sổ tiến hóa buổi sáng sớm, kiểm tra nếu cần tiến hóa mô hình...")
                return self._evolve_models()

        return False
    
    def _check_weekday_evolution(self):
        """Kiểm tra và bắt đầu tiến hóa vào buổi tối ngày thường"""
        current_day = datetime.now().weekday()
        
        # Nếu là ngày trong tuần (0-4 là thứ 2 đến thứ 6)
        if current_day <= 4:
            logger.info("Bắt đầu kiểm tra tiến hóa mô hình theo lịch trình buổi tối ngày thường...")
            return self._evolve_models()
        
        return False
    
    def _check_weekend_evolution(self):
        """Kiểm tra và bắt đầu tiến hóa vào cuối tuần"""
        current_day = datetime.now().weekday()
        
        # Nếu là cuối tuần (5-6 là thứ 7 và chủ nhật)
        if current_day >= 5:
            logger.info("Đang ở cuối tuần, bắt đầu tiến hóa mô hình...")
            return self._evolve_models()
        
        return False
    
    def _run_scheduler(self):
        """Chạy scheduler trong thread riêng"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)
    
    def _update_data(self):
        """Cập nhật dữ liệu"""
        try:
            logger.info("Đang cập nhật dữ liệu...")
            
            # Lấy dữ liệu cho tất cả cổ phiếu
            data = self.data_fetcher.get_batch_data_for_all_stocks()
            
            # Lưu vào biến instance
            self.current_data = data
            
            # Lưu dữ liệu vào đĩa (1 lần/giờ)
            current_hour = datetime.now().hour
            if not hasattr(self, 'last_save_hour') or self.last_save_hour != current_hour:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.data_fetcher.save_data_to_disk(data, filename=f"market_data_{timestamp}.json")
                self.last_save_hour = current_hour
            
            logger.info(f"Đã cập nhật dữ liệu cho {len(data)} cổ phiếu")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật dữ liệu: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _display_market_info(self):
        """Hiển thị thông tin thị trường"""
        try:
            market_status = self.market_hours.get_market_status()
            
            status_text = "mở cửa" if market_status['is_market_open'] else "đóng cửa"
            if market_status['is_pre_market']:
                status_text = "pre-market"
            elif market_status['is_after_market']:
                status_text = "after-market"
            
            logger.info(f"Thông tin thị trường: {market_status['time']}")
            logger.info(f"Trạng thái: {status_text}")
            
            if market_status['is_weekend']:
                logger.info("Thị trường đóng cửa (cuối tuần)")
            elif market_status['is_holiday']:
                logger.info("Thị trường đóng cửa (ngày lễ)")
            
            if not market_status['is_market_open'] and not market_status['is_pre_market']:
                next_open_time = market_status['next_pre_market_seconds'] // 60
                logger.info(f"Thời gian đến lần mở cửa pre-market tiếp theo: {next_open_time} phút")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi hiển thị thông tin thị trường: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _check_and_predict(self):
        """Kiểm tra và chạy dự đoán nếu đúng thời điểm"""
        try:
            # Kiểm tra xem có đang trong quá trình huấn luyện ban đầu không
            if self.is_initial_training:
                logger.info("Đang trong quá trình huấn luyện ban đầu, bỏ qua dự đoán")
                return False
                
            # Lấy thời gian hiện tại
            current_time = datetime.now()
            current_hour = current_time.hour
            current_day = current_time.weekday()
            
            # Chỉ chạy dự đoán vào ngày trong tuần (thứ 2 đến thứ 6) từ 7:00 đến 16:00
            if current_day > 4 or current_hour < 7 or current_hour >= 16:
                logger.info("Không trong giờ giao dịch (7:00-16:00, thứ 2-6), bỏ qua dự đoán")
                return False
            
            # Nếu đang dự đoán, bỏ qua
            if self.is_predicting:
                logger.info("Đang dự đoán, bỏ qua lần chạy này")
                return False
            
            # Chạy dự đoán
            self._run_predictions()
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra và chạy dự đoán: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _run_predictions(self):
        """Chạy dự đoán giá cổ phiếu"""
        try:
            logger.info("Bắt đầu dự đoán giá cổ phiếu...")
            self.is_predicting = True
            
            # Cập nhật dữ liệu trước khi dự đoán
            self._update_data()
            
            # Kết quả dự đoán
            all_predictions = {}
            
            # Lấy thời gian hiện tại
            now = datetime.now()
            
            # Dự đoán cho từng cổ phiếu
            for symbol in self.stocks:
                # Kiểm tra xem có dữ liệu không
                if symbol not in self.current_data:
                    logger.warning(f"Không có dữ liệu cho {symbol}, bỏ qua")
                    continue
                
                # Lấy dữ liệu
                stock_data = self.current_data[symbol]
                
                # Kiểm tra lỗi
                if 'error' in stock_data:
                    logger.error(f"Lỗi dữ liệu cho {symbol}: {stock_data['error']}")
                    continue
                
                # Dự đoán giá
                predictions = {}
                
                # Dự đoán intraday
                if symbol not in self.last_prediction_time or \
                    self.last_prediction_time[symbol]['intraday'] is None or \
                    (now - self.last_prediction_time[symbol]['intraday']).total_seconds() >= 900:  # 15 phút
                    
                    intraday_predictor = IntradayPredictor(symbol, config_path=self.config_path)
                    intraday_data = stock_data.get('intraday_data')
                    
                    # Huấn luyện mô hình nếu không tồn tại
                    if not intraday_predictor.check_model_exists() and intraday_data is not None and not intraday_data.empty:
                        logger.info(f"Mô hình intraday cho {symbol} không tồn tại, bắt đầu huấn luyện...")
                        intraday_predictor.train(intraday_data)
                    
                    if intraday_data is not None and not intraday_data.empty:
                        intraday_prediction = intraday_predictor.predict(intraday_data)
                        if intraday_prediction:
                            predictions['intraday'] = intraday_prediction
                    
                    # Cập nhật thời gian dự đoán
                    if symbol not in self.last_prediction_time:
                        self.last_prediction_time[symbol] = {'intraday': None, 'five_day': None, 'monthly': None}
                    self.last_prediction_time[symbol]['intraday'] = now
                
                # Dự đoán 5 ngày
                if symbol not in self.last_prediction_time or \
                    self.last_prediction_time[symbol]['five_day'] is None or \
                    (now - self.last_prediction_time[symbol]['five_day']).total_seconds() >= 14400:  # 4 giờ
                    
                    five_day_predictor = FiveDayPredictor(symbol, config_path=self.config_path)
                    daily_data = stock_data.get('daily_data')
                    
                    # Huấn luyện mô hình nếu không tồn tại
                    if not five_day_predictor.check_model_exists() and daily_data is not None and not daily_data.empty:
                        logger.info(f"Mô hình five_day cho {symbol} không tồn tại, bắt đầu huấn luyện...")
                        five_day_predictor.train(daily_data)
                    
                    if daily_data is not None and not daily_data.empty:
                        five_day_prediction = five_day_predictor.predict(daily_data)
                        if five_day_prediction:
                            predictions['five_day'] = five_day_prediction
                    
                    # Cập nhật thời gian dự đoán
                    if symbol not in self.last_prediction_time:
                        self.last_prediction_time[symbol] = {'intraday': None, 'five_day': None, 'monthly': None}
                    self.last_prediction_time[symbol]['five_day'] = now
                
                # Dự đoán 1 tháng
                if symbol not in self.last_prediction_time or \
                    self.last_prediction_time[symbol]['monthly'] is None or \
                    (now - self.last_prediction_time[symbol]['monthly']).total_seconds() >= 43200:  # 12 giờ
                    
                    monthly_predictor = MonthlyPredictor(symbol, config_path=self.config_path)
                    daily_data = stock_data.get('daily_data')
                    
                    # Huấn luyện mô hình nếu không tồn tại
                    if not monthly_predictor.check_model_exists() and daily_data is not None and not daily_data.empty:
                        logger.info(f"Mô hình monthly cho {symbol} không tồn tại, bắt đầu huấn luyện...")
                        monthly_predictor.train(daily_data)
                    
                    if daily_data is not None and not daily_data.empty:
                        monthly_prediction = monthly_predictor.predict(daily_data)
                        if monthly_prediction:
                            predictions['monthly'] = monthly_prediction
                    
                    # Cập nhật thời gian dự đoán
                    if symbol not in self.last_prediction_time:
                        self.last_prediction_time[symbol] = {'intraday': None, 'five_day': None, 'monthly': None}
                    self.last_prediction_time[symbol]['monthly'] = now
                    # Nếu có dự đoán
                if predictions:
                    # Lấy giá hiện tại
                    current_price = None
                    
                    if 'latest_quote' in stock_data and stock_data['latest_quote']:
                        if 'p' in stock_data['latest_quote']:  # Giá từ Polygon API
                            current_price = stock_data['latest_quote']['p']
                    
                    if current_price is None and 'intraday_data' in stock_data and not stock_data['intraday_data'].empty:
                        current_price = stock_data['intraday_data']['close'].iloc[-1]
                    
                    # Tạo biểu đồ
                    chart_path = None
                    if 'intraday_data' in stock_data and not stock_data['intraday_data'].empty:
                        chart_path = self.visualizer.create_price_prediction_chart(
                            symbol, 
                            stock_data['intraday_data'], 
                            predictions
                        )
                    
                    # Lưu vào kết quả
                    all_predictions[symbol] = {
                        **predictions,
                        'current_price': current_price,
                        'chart_path': chart_path
                    }
            
            # Lưu dự đoán gần nhất để sử dụng trong các thông báo tiếp theo
            for symbol, predictions in all_predictions.items():
                if symbol not in self.previous_predictions:
                    self.previous_predictions[symbol] = {}
                
                # Lưu các dự đoán hiện tại
                if 'intraday' in predictions:
                    self.previous_predictions[symbol]['intraday'] = predictions['intraday']
                if 'five_day' in predictions:
                    self.previous_predictions[symbol]['five_day'] = predictions['five_day']
                if 'monthly' in predictions:
                    self.previous_predictions[symbol]['monthly'] = predictions['monthly']

            # Gửi thông báo Discord
            if all_predictions:
                # Đảm bảo mỗi cổ phiếu có đủ 3 dự đoán
                complete_predictions = self._ensure_complete_predictions(all_predictions)
                
                # Gửi thông báo với dự đoán đã được bổ sung
                self.discord.send_prediction_message(complete_predictions)
            
            logger.info(f"Đã dự đoán giá cho {len(all_predictions)} cổ phiếu")
            
            # Đoạn code cần thêm vào:
            # Lưu dự đoán vào thư mục prediction_history
            prediction_history_dir = os.path.join(current_dir, "prediction_history")
            os.makedirs(prediction_history_dir, exist_ok=True)

            for symbol, predictions in all_predictions.items():
                # Thêm timestamp và symbol vào dữ liệu
                predictions['timestamp'] = now.isoformat()
                predictions['symbol'] = symbol
                
                # Tạo tên file
                filename = f"{symbol}_prediction_{now.strftime('%Y%m%d_%H%M%S')}.json"
                file_path = os.path.join(prediction_history_dir, filename)
                
                # Lưu dự đoán vào file
                with open(file_path, 'w') as f:
                    json.dump(predictions, f, indent=4)
                
                logger.info(f"Đã lưu dự đoán cho {symbol} vào {file_path}")

            # Đặt lại cờ
            self.is_predicting = False
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán giá cổ phiếu: {str(e)}")
            logger.error(traceback.format_exc())
            self.is_predicting = False
            return False
    
    def _train_model_if_needed(self, symbol, timeframe):
        """Huấn luyện mô hình nếu cần thiết"""
        try:
            logger.info(f"Kiểm tra và huấn luyện mô hình cho {symbol} - {timeframe}")
            
            # Xác định lớp predictor phù hợp
            if timeframe == 'intraday':
                predictor_class = IntradayPredictor
                data_key = 'intraday_data'
            elif timeframe == 'five_day':
                predictor_class = FiveDayPredictor
                data_key = 'daily_data'
            elif timeframe == 'monthly':
                predictor_class = MonthlyPredictor
                data_key = 'daily_data'
            else:
                logger.error(f"Timeframe không hợp lệ: {timeframe}")
                return False
            
            # Khởi tạo predictor
            predictor = predictor_class(symbol, config_path=self.config_path)
            
            # Kiểm tra xem mô hình đã tồn tại chưa
            if predictor.check_model_exists():
                logger.info(f"Mô hình cho {symbol} - {timeframe} đã tồn tại, không cần huấn luyện")
                return True
            
            # Nếu chưa tồn tại, huấn luyện mô hình mới
            logger.info(f"Bắt đầu huấn luyện mô hình mới cho {symbol} - {timeframe}")
            
            # Kiểm tra dữ liệu
            if symbol not in self.current_data:
                logger.warning(f"Không có dữ liệu cho {symbol}, không thể huấn luyện")
                return False
            
            stock_data = self.current_data[symbol]
            
            if data_key not in stock_data or stock_data[data_key] is None or stock_data[data_key].empty:
                logger.warning(f"Không có dữ liệu {data_key} cho {symbol}, không thể huấn luyện")
                return False
            
            # Huấn luyện mô hình
            history = predictor.train(stock_data[data_key])
            
            if history:
                logger.info(f"Đã huấn luyện thành công mô hình cho {symbol} - {timeframe}")
                return True
            else:
                logger.error(f"Huấn luyện mô hình thất bại cho {symbol} - {timeframe}")
                return False
                
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình {symbol} - {timeframe}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _train_all_models(self):
        """Huấn luyện tất cả các mô hình"""
        try:
            logger.info("Bắt đầu huấn luyện tất cả các mô hình...")
            
            # Đảm bảo có dữ liệu mới nhất
            self._update_data()
            
            # Đếm số mô hình đã huấn luyện thành công
            success_count = 0
            total_count = len(self.stocks) * 3  # 3 timeframes cho mỗi cổ phiếu
            
            # Huấn luyện cho từng cổ phiếu và từng timeframe
            for symbol in self.stocks:
                for timeframe in ['intraday', 'five_day', 'monthly']:
                    if self._train_model_if_needed(symbol, timeframe):
                        success_count += 1
            
            logger.info(f"Đã huấn luyện {success_count}/{total_count} mô hình")
            
            # Gửi thông báo Discord
            self.discord.send_system_update(
                title="Kết quả huấn luyện mô hình",
                message=f"Đã hoàn thành quá trình huấn luyện mô hình: {success_count}/{total_count} thành công."
            )
            
            return success_count > 0
        
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện tất cả mô hình: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Thông báo lỗi
            self.discord.send_system_update(
                title="Lỗi huấn luyện mô hình",
                message=f"Quá trình huấn luyện mô hình gặp lỗi: {str(e)}",
                is_error=True
            )
            
            return False
            
    def _evolve_models(self):
        """Tiến hóa mô hình"""
        try:
            # Kiểm tra xem có đang evolve không
            if self.is_evolving:
                logger.info("Đang tiến hóa mô hình, bỏ qua lần chạy này")
                return False
            
            # Lấy thời gian hiện tại
            current_time = datetime.now()
            current_hour = current_time.hour
            current_day = current_time.weekday()
            
            # Kiểm tra nếu đang trong giờ dự đoán (7:00-16:00) các ngày thứ 2-6
            if current_day <= 4 and 7 <= current_hour < 16:
                logger.info("Đang trong giờ dự đoán (7:00-16:00, thứ 2-6), bỏ qua tiến hóa")
                return False
            
            # Trước tiên, huấn luyện mô hình nếu cần
            # Kiểm tra xem có mô hình nào cần huấn luyện không
            logger.info("Kiểm tra và huấn luyện mô hình trước khi tiến hóa...")
            self._train_all_models()
            
            # Gửi thông báo bắt đầu tiến hóa
            self.discord.send_system_update(
                title="Bắt đầu tiến hóa mô hình",
                message=f"BondZiA AI đang tiến hóa để cải thiện khả năng dự đoán. Quá trình này có thể mất một thời gian. Thời điểm: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            logger.info("Bắt đầu tiến hóa mô hình...")
            self.is_evolving = True
            
            # Cập nhật dữ liệu
            self._update_data()
            
            # Tạo phiên bản mới
            new_version = self.version_manager.create_new_version()
            logger.info(f"Tạo phiên bản mới: {new_version}")
            
            # Khởi tạo evolution manager
            evolution_manager = ModelEvolutionManager(config_path=self.config_path)
            
            # Tiến hóa tất cả mô hình
            results = evolution_manager.evolve_all_models(self.current_data)
            
            # Lưu thời gian tiến hóa cuối cùng
            self.last_evolution_time = datetime.now()
            self._save_evolution_status()
            
            # Lấy thông tin cải thiện
            improvements = evolution_manager.get_evolution_improvements(results)
            
            # Tạo thông báo cải thiện
            improvements_dict = {}
            for symbol in improvements:
                for timeframe in improvements[symbol]:
                    if symbol not in improvements_dict:
                        improvements_dict[symbol] = 0
                    
                    # Trung bình cải thiện cho tất cả timeframes
                    improvements_dict[symbol] += improvements[symbol][timeframe] / len(improvements[symbol])
            
            # Hiệu suất trước và sau
            performance = {
                'before': f"RMSE: {results.get('avg_rmse_before', 'N/A')}, Accuracy: {results.get('avg_accuracy_before', 'N/A')}%",
                'after': f"RMSE: {results.get('avg_rmse_after', 'N/A')}, Accuracy: {results.get('avg_accuracy_after', 'N/A')}%"
            }
            
            # Gửi thông báo kết quả
            self.discord.send_evolution_update(
                version=results['version'],
                improvements=improvements_dict,
                params_changed=results['total_params_changed'],
                performance=performance
            )
            
            logger.info(f"Hoàn thành tiến hóa mô hình. Phiên bản mới: {results['version']}")
            
            # Đặt lại cờ
            self.is_evolving = False
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tiến hóa mô hình: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Gửi thông báo lỗi
            self.discord.send_system_update(
                title="Lỗi tiến hóa mô hình",
                message=f"Quá trình tiến hóa mô hình gặp lỗi: {str(e)}",
                is_error=True
            )
            
            self.is_evolving = False
            return False
    
    def _run_accuracy_evaluation(self):
        """Chạy đánh giá độ chính xác tự động"""
        try:
            logger.info("Bắt đầu đánh giá độ chính xác tự động")
            
            # Import module đánh giá
            from evolution.accuracy_evaluator import AccuracyEvaluator
            
            # Khởi tạo evaluator
            evaluator = AccuracyEvaluator()
            
            # Đánh giá tất cả cổ phiếu
            results = evaluator.evaluate_all_symbols(days=30)
            
            # Thông báo kết quả
            if results:
                self.discord.send_system_update(
                    title="Báo cáo đánh giá độ chính xác",
                    message=f"Đã hoàn thành đánh giá độ chính xác cho {len(results)} cổ phiếu.",
                    fields=[
                        {
                            "name": "Thời gian đánh giá",
                            "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "inline": True
                        },
                        {
                            "name": "Cổ phiếu đánh giá",
                            "value": ", ".join(results.keys()),
                            "inline": False
                        }
                    ]
                )
            
            logger.info(f"Đã hoàn thành đánh giá độ chính xác cho {len(results)} cổ phiếu")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá độ chính xác: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _save_evolution_status(self):
        """Lưu trạng thái tiến hóa gần nhất"""
        try:
            status_dir = os.path.join(current_dir, "status")
            os.makedirs(status_dir, exist_ok=True)
            
            status_file = os.path.join(status_dir, "evolution_status.json")
            
            status = {
                "last_evolution_time": self.last_evolution_time.isoformat() if hasattr(self, 'last_evolution_time') else None,
                "version": self.config['system']['version']
            }
            
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=4)
                
            logger.info(f"Đã lưu trạng thái tiến hóa: {status}")
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu trạng thái tiến hóa: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _load_evolution_status(self):
        """Tải trạng thái tiến hóa gần nhất"""
        try:
            status_file = os.path.join(current_dir, "status", "evolution_status.json")
            
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status = json.load(f)
                
                if 'last_evolution_time' in status and status['last_evolution_time']:
                    self.last_evolution_time = datetime.fromisoformat(status['last_evolution_time'])
                    logger.info(f"Lần tiến hóa cuối: {self.last_evolution_time}")
                    return True
            
            logger.info("Không tìm thấy thông tin về lần tiến hóa cuối")
            return False
        
        except Exception as e:
            logger.error(f"Lỗi khi tải trạng thái tiến hóa: {str(e)}")
            logger.error(traceback.format_exc())
            return False

def parse_arguments():
    """Parse các tham số dòng lệnh"""
    parser = argparse.ArgumentParser(description='BondZiA AI - Hệ thống dự đoán giá cổ phiếu')
    
    parser.add_argument('--evolve', action='store_true', help='Chạy tiến hóa mô hình ngay lập tức')
    parser.add_argument('--predict', action='store_true', help='Chạy dự đoán ngay lập tức')
    parser.add_argument('--train', action='store_true', help='Huấn luyện tất cả mô hình ngay lập tức')
    parser.add_argument('--config', type=str, default='config/system_config.json', help='Đường dẫn đến file cấu hình')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        default='INFO', help='Mức độ log')
    
    return parser.parse_args()

import utils.telegram_integration

if __name__ == "__main__":
    # Parse tham số dòng lệnh
    args = parse_arguments()
    
    # Khởi tạo BondZiA
    bondzia = BondZiA(config_path=args.config)
    
    # Xử lý các tùy chọn
    if args.evolve:
        logger.info("Chạy tiến hóa mô hình theo lệnh dòng lệnh")
        bondzia._evolve_models()
    elif args.train:
        logger.info("Chạy huấn luyện mô hình theo lệnh dòng lệnh")
        bondzia._train_all_models()
    elif args.predict:
        logger.info("Chạy dự đoán theo lệnh dòng lệnh")
        bondzia._run_predictions()
    else:
        # Chạy bình thường
        bondzia.start()