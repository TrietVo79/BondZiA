import os
import json
import requests
import time
from datetime import datetime
import pytz
from utils.logger_config import logger
import traceback
from io import BytesIO

class TelegramNotifier:
    """Lớp quản lý việc gửi thông báo đến Telegram"""
    
    def __init__(self, config_path="../config/system_config.json"):
        """
        Khởi tạo Telegram Notifier
        
        Args:
            config_path (str): Đường dẫn đến file cấu hình
        """
        # Đọc cấu hình
        self.config_path = config_path
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Đọc token và chat_id
        api_keys_path = os.path.join(os.path.dirname(self.config_path), "api_keys.json")
        with open(api_keys_path, 'r') as f:
            api_keys = json.load(f)
        
        self.token = api_keys['telegram']['bot_token']
        self.chat_id = api_keys['telegram']['chat_id']
        
        # Thiết lập cấu hình thông báo
        self.notification_config = self.config['notification']
        self.max_retries = self.notification_config.get('telegram_max_retries', 3)
        self.retry_delay = self.notification_config.get('telegram_retry_delay_seconds', 3)
        
        # Timezone
        self.market_timezone = pytz.timezone(self.config['market']['timezone'])
        
        # Khởi tạo các màu sắc (emoji thay cho màu)
        self.up_emoji = "🔼"
        self.down_emoji = "🔽"
        self.neutral_emoji = "➡️"
        
        logger.info("Khởi tạo Telegram Notifier thành công")
    
    def _clean_html_content(self, text):
        """
        Làm sạch nội dung HTML để tránh lỗi khi gửi tin nhắn Telegram
        """
        # Loại bỏ các ký tự đặc biệt, thay thế các thẻ không hợp lệ
        text = text.replace("<", "&lt;").replace(">", "&gt;")
        # Chỉ giữ lại các thẻ HTML hợp lệ với Telegram
        allowed_tags = ["b", "i", "u", "s", "code", "pre"]
        for tag in allowed_tags:
            text = text.replace(f"&lt;{tag}&gt;", f"<{tag}>")
            text = text.replace(f"&lt;/{tag}&gt;", f"</{tag}>")
        return text

    def _send_message(self, text, parse_mode="HTML", retries=None):
        """
        Gửi thông báo văn bản đến Telegram
        
        Args:
            text (str): Nội dung thông báo
            parse_mode (str): Chế độ parse ('HTML' hoặc 'Markdown')
            retries (int, optional): Số lần thử lại nếu gặp lỗi
            
        Returns:
            bool: True nếu gửi thành công, False nếu thất bại
        """
        if retries is None:
            retries = self.max_retries
        
        # Sử dụng hàm làm sạch HTML nếu parse_mode là HTML
        if parse_mode == "HTML":
            text = self._clean_html_content(text)

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': parse_mode
        }
        
        for attempt in range(retries + 1):
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                return True
            except requests.exceptions.RequestException as err:
                logger.error(f"Lỗi khi gửi tin nhắn Telegram: {err}")
                if attempt < retries:
                    logger.info(f"Thử lại sau {self.retry_delay} giây (lần {attempt + 1}/{retries})")
                    time.sleep(self.retry_delay)
                else:
                    return False
        
        return False
    
    def _send_photo(self, photo_path, caption, parse_mode="HTML", retries=None):
        """
        Gửi ảnh đến Telegram
        
        Args:
            photo_path (str): Đường dẫn đến file ảnh
            caption (str): Chú thích ảnh
            parse_mode (str): Chế độ parse ('HTML' hoặc 'Markdown')
            retries (int, optional): Số lần thử lại nếu gặp lỗi
            
        Returns:
            bool: True nếu gửi thành công, False nếu thất bại
        """
        if retries is None:
            retries = self.max_retries
        
        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
        
        for attempt in range(retries + 1):
            try:
                with open(photo_path, 'rb') as photo_file:
                    files = {'photo': photo_file}
                    data = {
                        'chat_id': self.chat_id,
                        'caption': caption,
                        'parse_mode': parse_mode
                    }
                    response = requests.post(url, files=files, data=data)
                    response.raise_for_status()
                    return True
            except requests.exceptions.RequestException as err:
                logger.error(f"Lỗi khi gửi ảnh Telegram: {err}")
                if attempt < retries:
                    logger.info(f"Thử lại sau {self.retry_delay} giây (lần {attempt + 1}/{retries})")
                    time.sleep(self.retry_delay)
                else:
                    return False
            except Exception as e:
                logger.error(f"Lỗi không xác định khi gửi ảnh: {str(e)}")
                return False
        
        return False
    
    def _format_prediction_text(self, symbol, prediction_data):
        """
        Định dạng dữ liệu dự đoán để hiển thị trong Telegram
        
        Args:
            symbol (str): Mã cổ phiếu
            prediction_data (dict): Dữ liệu dự đoán
            
        Returns:
            str: Văn bản đã định dạng cho Telegram
        """
        # Lấy dữ liệu dự đoán
        intraday = prediction_data.get('intraday', {})
        five_day = prediction_data.get('five_day', {})
        monthly = prediction_data.get('monthly', {})
        
        # Lấy giá hiện tại
        current_price = prediction_data.get('current_price', 'N/A')
        if isinstance(current_price, (int, float)):
            current_price = f"${current_price:.2f}"
            
        # Tạo văn bản dự đoán
        timestamp = datetime.now(self.market_timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
        
        # Tạo tiêu đề
        text = f"<b>BondZiA AI - Dự đoán giá {symbol}</b>\n"
        text += f"<i>({timestamp})</i>\n\n"
        text += f"Giá hiện tại: <b>{current_price}</b>\n\n"
        
        # Thêm thông tin intraday
        if intraday:
            direction = intraday.get('direction', 'neutral')
            emoji = self.up_emoji if direction == 'up' else self.down_emoji if direction == 'down' else self.neutral_emoji
            
            price = intraday.get('predicted_price', intraday.get('price', intraday.get('predicted_value', 'N/A')))
            if isinstance(price, (int, float)):
                price = f"${price:.2f}"
                
            confidence = intraday.get('confidence', 0)
            if isinstance(confidence, (int, float)):
                confidence = f"{confidence:.1f}%"
            
            reason = intraday.get('reason', "Dự đoán dựa trên mô hình ML")
            
            text += f"{emoji} <b>Dự đoán Intraday:</b>\n"
            text += f"Giá: {price}\n"
            text += f"Độ tin cậy: {confidence}\n"
            text += f"Lý do: {reason}\n\n"
        
        # Thêm thông tin 5 ngày
        if five_day:
            direction = five_day.get('direction', 'neutral')
            emoji = self.up_emoji if direction == 'up' else self.down_emoji if direction == 'down' else self.neutral_emoji
            
            price = five_day.get('predicted_price', five_day.get('price', five_day.get('predicted_value', 'N/A')))
            if isinstance(price, (int, float)):
                price = f"${price:.2f}"
                
            confidence = five_day.get('confidence', 0)
            if isinstance(confidence, (int, float)):
                confidence = f"{confidence:.1f}%"
            
            reason = five_day.get('reason', "Dự đoán dựa trên mô hình ML")
            
            text += f"{emoji} <b>Dự đoán 5 ngày:</b>\n"
            text += f"Giá: {price}\n"
            text += f"Độ tin cậy: {confidence}\n"
            text += f"Lý do: {reason}\n\n"
        
        # Thêm thông tin 1 tháng
        if monthly:
            direction = monthly.get('direction', 'neutral')
            emoji = self.up_emoji if direction == 'up' else self.down_emoji if direction == 'down' else self.neutral_emoji
            
            price = monthly.get('predicted_price', monthly.get('price', monthly.get('predicted_value', 'N/A')))
            if isinstance(price, (int, float)):
                price = f"${price:.2f}"
                
            confidence = monthly.get('confidence', 0)
            if isinstance(confidence, (int, float)):
                confidence = f"{confidence:.1f}%"
            
            reason = monthly.get('reason', "Dự đoán dựa trên mô hình ML")
            
            text += f"{emoji} <b>Dự đoán 1 tháng:</b>\n"
            text += f"Giá: {price}\n"
            text += f"Độ tin cậy: {confidence}\n"
            text += f"Lý do: {reason}\n"
        
        return text
    
    def send_prediction_message(self, predictions_data):
        """
        Gửi thông báo dự đoán giá cổ phiếu
        
        Args:
            predictions_data (dict): Dữ liệu dự đoán theo cấu trúc {symbol: prediction_data}
            
        Returns:
            bool: True nếu gửi thành công, False nếu thất bại
        """
        if not self.notification_config.get('send_telegram_predictions', True):
            logger.info("Bỏ qua thông báo dự đoán Telegram (đã tắt trong cấu hình)")
            return True
        
        try:
            success = True
            
            # Gửi thông báo cho từng cổ phiếu
            for symbol, prediction_data in predictions_data.items():
                # Tạo văn bản dự đoán
                prediction_text = self._format_prediction_text(symbol, prediction_data)
                
                # Gửi văn bản
                if not self._send_message(prediction_text):
                    logger.error(f"Không thể gửi thông báo dự đoán Telegram cho: {symbol}")
                    success = False
                else:
                    logger.info(f"Đã gửi thông báo dự đoán Telegram cho: {symbol}")
                
                # Nếu có biểu đồ, gửi biểu đồ
                if 'chart_path' in prediction_data and prediction_data['chart_path']:
                    chart_path = prediction_data['chart_path']
                    caption = f"Biểu đồ dự đoán giá {symbol}"
                    
                    if not self._send_photo(chart_path, caption):
                        logger.error(f"Không thể gửi biểu đồ dự đoán Telegram cho: {symbol}")
                        success = False
                    else:
                        logger.info(f"Đã gửi biểu đồ dự đoán Telegram cho: {symbol}")
                
                # Đợi giữa các lần gửi để tránh rate limit
                time.sleep(1)
            
            return success
        except Exception as e:
            logger.error(f"Lỗi khi gửi thông báo dự đoán Telegram: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def send_system_update(self, title, message, is_error=False, fields=None):
        """
        Gửi thông báo cập nhật hệ thống
        
        Args:
            title (str): Tiêu đề thông báo
            message (str): Nội dung thông báo
            is_error (bool): True nếu là thông báo lỗi
            fields (list, optional): Danh sách các trường bổ sung
            
        Returns:
            bool: True nếu gửi thành công, False nếu thất bại
        """
        if is_error and not self.notification_config.get('send_telegram_errors', True):
            logger.info("Bỏ qua thông báo lỗi Telegram (đã tắt trong cấu hình)")
            return True
        
        if not is_error and not self.notification_config.get('send_telegram_system_updates', True):
            logger.info("Bỏ qua thông báo cập nhật Telegram (đã tắt trong cấu hình)")
            return True
        
        try:
            # Xác định emoji dựa trên loại thông báo
            emoji = "⚠️" if is_error else "📢"
            
            # Tạo timestamp
            timestamp = datetime.now(self.market_timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
            
            # Tạo văn bản thông báo
            text = f"<b>{emoji} {title}</b>\n"
            text += f"<i>({timestamp})</i>\n\n"
            text += f"{message}\n"
            
            # Thêm các trường bổ sung
            if fields:
                text += "\n"
                for field in fields:
                    name = field.get("name", "")
                    value = field.get("value", "")
                    text += f"<b>{name}:</b> {value}\n"
            
            # Gửi thông báo
            return self._send_message(text)
        except Exception as e:
            logger.error(f"Lỗi khi gửi thông báo cập nhật hệ thống qua Telegram: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def send_evolution_update(self, version, improvements, params_changed, performance):
        """
        Gửi thông báo cập nhật về tiến hóa của AI
        
        Args:
            version (str): Phiên bản mới
            improvements (dict): Cải thiện theo cổ phiếu {symbol: percentage}
            params_changed (int): Số lượng siêu tham số đã thay đổi
            performance (dict): Hiệu suất trước và sau khi tiến hóa
            
        Returns:
            bool: True nếu gửi thành công, False nếu thất bại
        """
        if not self.notification_config.get('send_telegram_system_updates', True):
            logger.info("Bỏ qua thông báo tiến hóa Telegram (đã tắt trong cấu hình)")
            return True
        
        try:
            # Tạo danh sách cải thiện
            improvements_list = []
            for symbol, percentage in improvements.items():
                if percentage > 0:
                    improvements_list.append(f"{symbol}: +{percentage:.2f}%")
                else:
                    improvements_list.append(f"{symbol}: {percentage:.2f}%")
            
            improvements_text = "\n".join(improvements_list)
            
            # Tạo thông tin về hiệu suất
            performance_text = f"Trước tiến hóa: {performance.get('before', 'N/A')}\nSau tiến hóa: {performance.get('after', 'N/A')}"
            
            # Tạo các trường
            fields = [
                {
                    "name": "Cải thiện dự đoán",
                    "value": improvements_text
                },
                {
                    "name": "Số lượng siêu tham số thay đổi",
                    "value": str(params_changed)
                },
                {
                    "name": "Hiệu suất",
                    "value": performance_text
                }
            ]
            
            # Gửi thông báo
            return self.send_system_update(
                title=f"Tiến hóa AI thành công - Phiên bản {version}",
                message="BondZiA AI đã hoàn thành quá trình tiến hóa và nâng cấp lên phiên bản mới.",
                fields=fields
            )
        except Exception as e:
            logger.error(f"Lỗi khi gửi thông báo tiến hóa qua Telegram: {str(e)}")
            logger.error(traceback.format_exc())
            return False

if __name__ == "__main__":
    # Test module
    logger.info("Kiểm tra module Telegram Notifier")
    notifier = TelegramNotifier()
    
    # Test gửi thông báo cập nhật hệ thống
    success = notifier.send_system_update(
        title="Khởi động BondZiA AI",
        message="Hệ thống BondZiA AI đã khởi động và sẵn sàng dự đoán giá cổ phiếu.",
        fields=[
            {
                "name": "Phiên bản",
                "value": "1.0.0"
            },
            {
                "name": "Cổ phiếu theo dõi",
                "value": "TSLA, NVDA, PLTR, AGX, META, AMZN, AAPL, IBM, BABA"
            }
        ]
    )
    
    logger.info(f"Gửi thông báo cập nhật Telegram: {'Thành công' if success else 'Thất bại'}")