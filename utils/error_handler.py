import os
import sys
import traceback
import json
import time
import psutil
import signal
import threading
import subprocess
from datetime import datetime
import logging
from utils.logger_config import logger
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ErrorHandler:
    """Lớp xử lý lỗi và tự phục hồi cho BondZiA AI"""
    
    def __init__(self, config_path="../config/system_config.json", discord_notifier=None):
        """
        Khởi tạo Error Handler
        
        Args:
            config_path (str): Đường dẫn đến file cấu hình
            discord_notifier: Đối tượng DiscordNotifier để gửi thông báo lỗi
        """
        # Đọc cấu hình
        self.config_path = config_path
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Kiểm tra xem tự sửa lỗi có được bật không
        self.self_repair_enabled = self.config['system']['self_repair_enabled']
        
        # Đường dẫn đến script chính
        self.main_script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                          "main.py")
        
        # Đường dẫn log
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  "logs/errors")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Discord notifier
        self.discord_notifier = discord_notifier
        
        # Khởi tạo watchdog
        self.observer = None
        
        # Danh sách lỗi đã xử lý
        self.handled_errors = {}
        
        # Đếm số lần khởi động lại
        self.restart_count = 0
        self.restart_limit = 5  # Giới hạn số lần khởi động lại trong 1 giờ
        self.restart_window = datetime.now()
        
        # Trạng thái hiện tại
        self.status = "initialized"
        
        logger.info("Khởi tạo ErrorHandler thành công")
    
    def setup_watchdog(self):
        """Thiết lập theo dõi file log để phát hiện lỗi"""
        event_handler = LogFileHandler(self)
        self.observer = Observer()
        
        # Theo dõi thư mục log
        self.observer.schedule(event_handler, self.log_dir, recursive=True)
        self.observer.start()
        
        logger.info("Đã khởi động Watchdog để theo dõi file log")
    
    def stop_watchdog(self):
        """Dừng watchdog"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Đã dừng Watchdog")
    
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """
        Xử lý ngoại lệ không bắt được
        
        Args:
            exc_type: Loại ngoại lệ
            exc_value: Giá trị ngoại lệ
            exc_traceback: Traceback ngoại lệ
        """
        # Kiểm tra nếu là KeyboardInterrupt thì không xử lý
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Lấy thông tin lỗi
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        error_type = exc_type.__name__
        error_value = str(exc_value)
        
        # Log lỗi
        logger.error(f"Uncaught exception: {error_type}: {error_value}")
        logger.error(f"Traceback: {error_msg}")
        
        # Lưu lỗi vào file
        self._save_error_to_file(error_type, error_value, error_msg)
        
        # Gửi thông báo lỗi
        self._send_error_notification(error_type, error_value, error_msg)
        
        # Xử lý tự động
        self._handle_error(error_type, error_value, error_msg)
    
    def handle_error(self, error_type, error_value, traceback_str=None):
        """
        Xử lý lỗi được bắt
        
        Args:
            error_type (str): Loại lỗi
            error_value (str): Thông báo lỗi
            traceback_str (str, optional): Chuỗi traceback
        """
        # Log lỗi
        logger.error(f"Caught error: {error_type}: {error_value}")
        if traceback_str:
            logger.error(f"Traceback: {traceback_str}")
        
        # Lưu lỗi vào file
        self._save_error_to_file(error_type, error_value, traceback_str)
        
        # Gửi thông báo lỗi
        self._send_error_notification(error_type, error_value, traceback_str)
        
        # Xử lý tự động
        self._handle_error(error_type, error_value, traceback_str)
    
    def _save_error_to_file(self, error_type, error_value, traceback_str):
        """
        Lưu thông tin lỗi vào file
        
        Args:
            error_type (str): Loại lỗi
            error_value (str): Giá trị lỗi
            traceback_str (str): Chuỗi traceback
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"error_{timestamp}.log"
            filepath = os.path.join(self.log_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Error Type: {error_type}\n")
                f.write(f"Error Value: {error_value}\n")
                f.write(f"Traceback:\n{traceback_str}\n")
            
            logger.info(f"Đã lưu thông tin lỗi vào {filepath}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu thông tin lỗi: {str(e)}")
    
    def _send_error_notification(self, error_type, error_value, traceback_str):
        """
        Gửi thông báo lỗi qua Discord
        
        Args:
            error_type (str): Loại lỗi
            error_value (str): Giá trị lỗi
            traceback_str (str): Chuỗi traceback
        """
        if self.discord_notifier:
            # Cắt ngắn traceback nếu quá dài
            if traceback_str and len(traceback_str) > 500:
                traceback_short = traceback_str[:500] + "...[truncated]"
            else:
                traceback_short = traceback_str
            
            # Tạo thông báo
            title = f"Lỗi hệ thống: {error_type}"
            message = f"BondZiA AI đã gặp lỗi và đang cố gắng tự khắc phục.\n\nLỗi: {error_value}"
            
            fields = [
                {
                    "name": "Trạng thái tự sửa lỗi",
                    "value": "Đang kích hoạt" if self.self_repair_enabled else "Đã tắt",
                    "inline": True
                },
                {
                    "name": "Thời gian xảy ra",
                    "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "inline": True
                }
            ]
            
            if traceback_short:
                fields.append({
                    "name": "Chi tiết lỗi",
                    "value": f"```{traceback_short}```",
                    "inline": False
                })
            
            self.discord_notifier.send_system_update(title, message, is_error=True, fields=fields)
    
    def _handle_error(self, error_type, error_value, traceback_str):
        """
        Xử lý lỗi tự động
        
        Args:
            error_type (str): Loại lỗi
            error_value (str): Giá trị lỗi
            traceback_str (str): Chuỗi traceback
        """
        # Nếu tự sửa lỗi không được bật, bỏ qua
        if not self.self_repair_enabled:
            logger.info("Tự sửa lỗi đã bị tắt. Bỏ qua xử lý lỗi tự động.")
            return
        
        # Đánh dấu ID lỗi dựa trên loại và giá trị
        error_id = f"{error_type}:{error_value}"
        
        # Kiểm tra xem lỗi này đã được xử lý gần đây chưa
        if error_id in self.handled_errors:
            last_handled_time, handle_count = self.handled_errors[error_id]
            time_diff = (datetime.now() - last_handled_time).total_seconds() / 60  # Phút
            
            if time_diff < 30 and handle_count >= 3:
                logger.warning(f"Lỗi {error_id} đã xảy ra {handle_count} lần trong 30 phút. Không xử lý tiếp.")
                
                # Thông báo lỗi lặp lại
                if self.discord_notifier:
                    title = f"Lỗi lặp lại: {error_type}"
                    message = f"Lỗi đã xảy ra {handle_count} lần trong 30 phút. Cần kiểm tra thủ công."
                    self.discord_notifier.send_system_update(title, message, is_error=True)
                
                return
            
            # Cập nhật số lần xử lý
            if time_diff < 30:
                self.handled_errors[error_id] = (last_handled_time, handle_count + 1)
            else:
                self.handled_errors[error_id] = (datetime.now(), 1)
        else:
            self.handled_errors[error_id] = (datetime.now(), 1)
        
        # Xử lý dựa vào loại lỗi
        if "ConnectionError" in error_type or "Timeout" in error_type or "ApiError" in error_type:
            logger.info("Phát hiện lỗi kết nối. Đợi 60 giây trước khi thử lại.")
            time.sleep(60)
            self.restart_system()
        elif "JSONDecodeError" in error_type or "ValueError" in error_type:
            logger.info("Phát hiện lỗi dữ liệu. Đợi 30 giây trước khi thử lại.")
            time.sleep(30)
            self.restart_system()
        elif "MemoryError" in error_type:
            logger.info("Phát hiện lỗi bộ nhớ. Dọn dẹp bộ nhớ trước khi khởi động lại.")
            self._clean_memory()
            time.sleep(10)
            self.restart_system()
        elif "FileNotFoundError" in error_type or "PermissionError" in error_type:
            logger.info("Phát hiện lỗi file. Kiểm tra quyền truy cập và tệp tin.")
            # Tự động tạo file nếu cần
            if "No such file or directory" in error_value:
                file_path = self._extract_file_path_from_error(error_value)
                if file_path:
                    self._ensure_file_exists(file_path)
            time.sleep(5)
            self.restart_system()
        else:
            # Lỗi chưa biết, khởi động lại hệ thống
            logger.info(f"Phát hiện lỗi chưa biết: {error_type}. Khởi động lại hệ thống.")
            self.restart_system()
    
    def _extract_file_path_from_error(self, error_value):
        """
        Trích xuất đường dẫn file từ thông báo lỗi
        
        Args:
            error_value (str): Thông báo lỗi
            
        Returns:
            str: Đường dẫn file hoặc None nếu không tìm thấy
        """
        # Tìm chuỗi đường dẫn trong thông báo lỗi
        import re
        path_matches = re.findall(r"'([^']*\.(?:json|py|csv|txt|log))'", error_value)
        if path_matches:
            return path_matches[0]
        return None
    
    def _ensure_file_exists(self, file_path):
        """
        Đảm bảo file tồn tại, tạo nếu cần
        
        Args:
            file_path (str): Đường dẫn file
        """
        try:
            # Tạo thư mục nếu cần
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Đã tạo thư mục: {directory}")
            
            # Tạo file trống nếu không tồn tại
            if not os.path.exists(file_path):
                # Xác định loại file
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump({}, f)
                else:
                    with open(file_path, 'w') as f:
                        pass
                logger.info(f"Đã tạo file: {file_path}")
        except Exception as e:
            logger.error(f"Lỗi khi tạo file {file_path}: {str(e)}")
    
    def _clean_memory(self):
        """Dọn dẹp bộ nhớ"""
        # Gọi garbage collector
        import gc
        gc.collect()
        
        # Ghi log
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"Bộ nhớ đã sử dụng sau khi dọn dẹp: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    def restart_system(self):
        """Khởi động lại hệ thống"""
        # Kiểm tra số lần khởi động lại
        current_time = datetime.now()
        time_diff = (current_time - self.restart_window).total_seconds() / 3600  # Giờ
        
        if time_diff > 1:
            # Reset bộ đếm nếu đã qua 1 giờ
            self.restart_count = 0
            self.restart_window = current_time
        
        self.restart_count += 1
        
        if self.restart_count > self.restart_limit:
            logger.critical(f"Đã khởi động lại {self.restart_count} lần trong 1 giờ. Dừng hệ thống để tránh vòng lặp lỗi.")
            
            # Thông báo
            if self.discord_notifier:
                title = "Dừng tự động khởi động lại"
                message = f"BondZiA AI đã khởi động lại {self.restart_count} lần trong 1 giờ. Hệ thống đã dừng để tránh vòng lặp lỗi. Cần kiểm tra thủ công."
                self.discord_notifier.send_system_update(title, message, is_error=True)
            
            # Ghi log
            logger.info("Hệ thống sẽ dừng lại sau 5 giây...")
            time.sleep(5)
            self.status = "stopped"
            return
        
        logger.info(f"Khởi động lại hệ thống (lần {self.restart_count}/{self.restart_limit} trong 1 giờ)")
        
        # Thông báo khởi động lại
        if self.discord_notifier:
            title = "Đang khởi động lại hệ thống"
            message = f"BondZiA AI đang khởi động lại để khắc phục lỗi (lần {self.restart_count}/{self.restart_limit} trong 1 giờ)."
            self.discord_notifier.send_system_update(title, message, is_error=False)
        
        # Khởi động lại bằng cách chạy script mới
        try:
            # Lấy đường dẫn Python hiện tại
            python_executable = sys.executable
            
            # Mở tiến trình mới
            subprocess.Popen([python_executable, self.main_script_path])
            
            # Ghi log
            logger.info(f"Đã khởi động tiến trình mới: {python_executable} {self.main_script_path}")
            logger.info("Tiến trình hiện tại sẽ thoát sau 5 giây...")
            
            # Đợi một chút rồi thoát
            time.sleep(5)
            
            # Thoát tiến trình hiện tại
            os._exit(0)
        except Exception as e:
            logger.error(f"Lỗi khi khởi động lại hệ thống: {str(e)}")
            self.status = "error"

class LogFileHandler(FileSystemEventHandler):
    """Lớp xử lý sự kiện file log"""
    
    def __init__(self, error_handler):
        """
        Khởi tạo Log File Handler
        
        Args:
            error_handler: Đối tượng ErrorHandler
        """
        self.error_handler = error_handler
    
    def on_created(self, event):
        """
        Xử lý khi file mới được tạo
        
        Args:
            event: Sự kiện tạo file
        """
        # Chỉ xử lý file log
        if not event.is_directory and event.src_path.endswith('.log'):
            # Đọc file log
            try:
                time.sleep(1)  # Đợi file được ghi hoàn tất
                with open(event.src_path, 'r') as f:
                    log_content = f.read()
                
                # Phân tích nội dung log
                error_type = None
                error_value = None
                traceback_str = None
                
                for line in log_content.split('\n'):
                    if line.startswith('Error Type:'):
                        error_type = line.replace('Error Type:', '').strip()
                    elif line.startswith('Error Value:'):
                        error_value = line.replace('Error Value:', '').strip()
                    elif line.startswith('Traceback:'):
                        traceback_str = '\n'.join(log_content.split('\n')[log_content.split('\n').index(line) + 1:])
                
                if error_type and error_value:
                    # Gửi thông tin lỗi đến error handler
                    self.error_handler.handle_error(error_type, error_value, traceback_str)
            except Exception as e:
                logger.error(f"Lỗi khi xử lý file log {event.src_path}: {str(e)}")

# Đăng ký exception hook
def setup_exception_handler(error_handler):
    """
    Đăng ký exception handler toàn cục
    
    Args:
        error_handler: Đối tượng ErrorHandler
    """
    sys.excepthook = error_handler.handle_exception
    
    # Đăng ký xử lý cho các signal
    signal.signal(signal.SIGTERM, lambda signum, frame: handle_signal(signum, frame, error_handler))
    signal.signal(signal.SIGINT, lambda signum, frame: handle_signal(signum, frame, error_handler))

def handle_signal(signum, frame, error_handler):
    """
    Xử lý signal hệ thống
    
    Args:
        signum: Số hiệu signal
        frame: Stack frame
        error_handler: Đối tượng ErrorHandler
    """
    signal_names = {
        signal.SIGTERM: "SIGTERM",
        signal.SIGINT: "SIGINT"
    }
    
    signal_name = signal_names.get(signum, f"Signal {signum}")
    logger.info(f"Nhận tín hiệu {signal_name}")
    
    # Dừng watchdog
    error_handler.stop_watchdog()
    
    # Thông báo dừng hệ thống
    if error_handler.discord_notifier:
        title = "Hệ thống đang dừng"
        message = f"BondZiA AI đang dừng hoạt động do nhận tín hiệu {signal_name}."
        error_handler.discord_notifier.send_system_update(title, message, is_error=False)
    
    # Thoát sau 5 giây
    logger.info("Hệ thống sẽ thoát sau 5 giây...")
    time.sleep(5)
    sys.exit(0)

if __name__ == "__main__":
    # Test module
    logger.info("Kiểm tra module ErrorHandler")
    
    # Tạo ErrorHandler
    error_handler = ErrorHandler()
    
    # Đăng ký exception hook
    setup_exception_handler(error_handler)
    
    # Khởi động watchdog
    error_handler.setup_watchdog()
    
    # Tạo lỗi để test
    logger.info("Tạo lỗi để test...")
    try:
        1 / 0
    except Exception as e:
        error_handler.handle_error(type(e).__name__, str(e), traceback.format_exc())
    
    # Đợi một chút trước khi thoát
    logger.info("Đợi 5 giây trước khi thoát...")
    time.sleep(5)
    
    # Dừng watchdog
    error_handler.stop_watchdog()
