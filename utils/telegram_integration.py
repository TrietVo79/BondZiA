# Tạo file mới này trong thư mục utils/
import os
import json
from utils.telegram_notifier import TelegramNotifier
from utils.logger_config import logger

# Lấy đường dẫn hiện tại
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Patch cho BondZiA
def patch_bondzia():
    from main import BondZiA
    
    # Lưu các phương thức gốc
    original_init = BondZiA.__init__
    
    # Override phương thức __init__
    def patched_init(self, config_path="config/system_config.json"):
        # Gọi phương thức gốc
        original_init(self, config_path)
        
        # Thêm Telegram Notifier
        self.telegram = TelegramNotifier(config_path=config_path)
        logger.info("Đã tích hợp Telegram Notifier")
    
    # Áp dụng patch cho __init__
    BondZiA.__init__ = patched_init
    
    # Patch các phương thức của DiscordNotifier
    from utils.discord_notifier import DiscordNotifier
    
    # Lưu các phương thức gốc
    original_discord_send_prediction = DiscordNotifier.send_prediction_message
    original_discord_send_system_update = DiscordNotifier.send_system_update
    original_discord_send_evolution_update = DiscordNotifier.send_evolution_update
    
    # Patch phương thức send_prediction_message
    def patched_send_prediction(self, predictions_data):
        # Gọi phương thức gốc
        result = original_discord_send_prediction(self, predictions_data)
        
        # Gửi thông qua Telegram
        try:
            # Tạo Telegram notifier mỗi khi cần gửi
            telegram = TelegramNotifier(config_path=os.path.join(current_dir, "config/system_config.json"))
            telegram.send_prediction_message(predictions_data)
            logger.info("Đã gửi thông báo dự đoán qua Telegram")
        except Exception as e:
            logger.error(f"Lỗi khi gửi thông báo dự đoán qua Telegram: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result
        
    # Patch phương thức send_system_update
    def patched_send_system_update(self, title, message, is_error=False, fields=None):
        # Gọi phương thức gốc
        result = original_discord_send_system_update(self, title, message, is_error, fields)
        
        # Gửi thông qua Telegram
        try:
            # Tạo Telegram notifier mỗi khi cần gửi
            telegram = TelegramNotifier(config_path=os.path.join(current_dir, "config/system_config.json"))
            telegram.send_system_update(title, message, is_error, fields)
            logger.info("Đã gửi thông báo hệ thống qua Telegram")
        except Exception as e:
            logger.error(f"Lỗi khi gửi thông báo hệ thống qua Telegram: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result
        
    # Patch phương thức send_evolution_update
    def patched_send_evolution_update(self, version, improvements, params_changed, performance):
        # Gọi phương thức gốc
        result = original_discord_send_evolution_update(self, version, improvements, params_changed, performance)
        
        # Gửi thông qua Telegram
        try:
            # Tạo Telegram notifier mỗi khi cần gửi
            telegram = TelegramNotifier(config_path=os.path.join(current_dir, "config/system_config.json"))
            telegram.send_evolution_update(version, improvements, params_changed, performance)
            logger.info("Đã gửi thông báo tiến hóa qua Telegram")
        except Exception as e:
            logger.error(f"Lỗi khi gửi thông báo tiến hóa qua Telegram: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result
    
    # Áp dụng các patch cho DiscordNotifier
    DiscordNotifier.send_prediction_message = patched_send_prediction
    DiscordNotifier.send_system_update = patched_send_system_update
    DiscordNotifier.send_evolution_update = patched_send_evolution_update
    
    logger.info("Đã patch thành công các phương thức gửi thông báo Discord để gửi qua Telegram")

# Tự động thực hiện patch khi import
try:
    patch_bondzia()
    logger.info("Đã tích hợp thành công Telegram vào BondZiA")
except Exception as e:
    logger.error(f"Không thể tích hợp Telegram: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())