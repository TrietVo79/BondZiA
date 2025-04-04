import os
import json
import time
import requests
from datetime import datetime
import pytz
from utils.logger_config import logger
import traceback
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.io as pio
import pandas as pd

class DiscordNotifier:
    """Lớp quản lý việc gửi thông báo đến Discord"""
    
    def __init__(self, config_path="../config/system_config.json"):
        """
        Khởi tạo Discord Notifier
        
        Args:
            config_path (str): Đường dẫn đến file cấu hình
        """
        # Đọc cấu hình
        self.config_path = config_path
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Đọc webhook URLs
        api_keys_path = os.path.join(os.path.dirname(self.config_path), "api_keys.json")
        with open(api_keys_path, 'r') as f:
            api_keys = json.load(f)
        
        self.prediction_webhook = api_keys['discord']['prediction_webhook']
        self.update_webhook = api_keys['discord']['update_webhook']
        
        # Thiết lập cấu hình thông báo
        self.notification_config = self.config['notification']
        self.max_retries = self.notification_config['max_retries']
        self.retry_delay = self.notification_config['retry_delay_seconds']
        
        # Timezone
        self.market_timezone = pytz.timezone(self.config['market']['timezone'])
        
        # Khởi tạo các màu sắc theo cấu hình
        self.viz_config = self.config['visualization']
        self.up_color = self.viz_config['up_color']
        self.down_color = self.viz_config['down_color']
        self.neutral_color = self.viz_config['neutral_color']
        
        logger.info("Khởi tạo Discord Notifier thành công")
    
    def _send_webhook(self, webhook_url, payload, retries=None):
        """
        Gửi thông báo đến webhook Discord
        
        Args:
            webhook_url (str): URL webhook Discord
            payload (dict): Nội dung thông báo
            retries (int, optional): Số lần thử lại nếu gặp lỗi
            
        Returns:
            bool: True nếu gửi thành công, False nếu thất bại
        """
        if retries is None:
            retries = self.max_retries
        
        headers = {
            "Content-Type": "application/json"
        }
        
        for attempt in range(retries + 1):
            try:
                response = requests.post(webhook_url, json=payload, headers=headers)
                
                if response.status_code == 429:  # Rate limit
                    # Lấy thời gian cần đợi từ phản hồi
                    retry_after = response.json().get('retry_after', self.retry_delay) / 1000.0
                    logger.warning(f"Discord rate limit, đợi {retry_after} giây")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                return True
            except requests.exceptions.HTTPError as http_err:
                logger.error(f"HTTP Error: {http_err}")
                if attempt < retries:
                    logger.info(f"Thử lại sau {self.retry_delay} giây (lần {attempt + 1}/{retries})")
                    time.sleep(self.retry_delay)
                else:
                    return False
            except Exception as err:
                logger.error(f"Lỗi khi gửi webhook: {err}")
                if attempt < retries:
                    logger.info(f"Thử lại sau {self.retry_delay} giây (lần {attempt + 1}/{retries})")
                    time.sleep(self.retry_delay)
                else:
                    return False
        
        return False
    
    def _format_prediction(self, prediction_data, prediction_type):
        """
        Định dạng dữ liệu dự đoán để hiển thị trong Discord
        
        Args:
            prediction_data (dict): Dữ liệu dự đoán
            prediction_type (str): Loại dự đoán ('intraday', 'five_day', 'monthly')
            
        Returns:
            dict: Dữ liệu đã định dạng
        """
        if prediction_type not in prediction_data:
            logger.debug(f"Không tìm thấy dự đoán loại {prediction_type} trong dữ liệu")
            return None
        
        pred = prediction_data[prediction_type]
        if not pred or not isinstance(pred, dict):
            logger.debug(f"Dự đoán loại {prediction_type} không hợp lệ hoặc rỗng")
            return None
        
        # Kiểm tra xem có đủ thông tin không
        if 'direction' not in pred or 'confidence' not in pred:
            logger.debug(f"Dự đoán loại {prediction_type} thiếu thông tin direction hoặc confidence")
            return None
        
        # Log chi tiết
        logger.info(f"DEBUG - Format prediction {prediction_type}: original_direction={pred['direction']}, confidence={pred['confidence']}")

        # Tạo emoji hướng
        direction_emoji = "🔼" if pred['direction'] == 'up' else "🔽" if pred['direction'] == 'down' else "➡️"
        
        # Định dạng giá
        price = pred.get('predicted_price', pred.get('price', pred.get('predicted_value', 'N/A')))
        if isinstance(price, (int, float)):
            price = f"${price:.2f}"
        
        # Định dạng độ tin cậy
        confidence = pred.get('confidence', 0)
        if isinstance(confidence, (int, float)):
            confidence = f"{confidence:.1f}%"
        else:
            confidence = "N/A"
        
        # Tạo lý do cho dự đoán
        reason = pred.get('reason', "")
        if not reason or reason == "":
            # Đảm bảo luôn có lý do phù hợp với hướng dự đoán
            if pred['direction'] == 'up':
                if prediction_type == 'intraday':
                    price_change = pred.get('price_change_percent', 0)
                    reason = f"Mô hình dự đoán giá tăng {abs(price_change):.1f}%"
                elif prediction_type == 'five_day':
                    price_change = pred.get('price_change_percent', 0)
                    reason = f"Mô hình dự đoán giá tăng {abs(price_change):.1f}% trong 5 ngày tới"
                else:  # monthly
                    price_change = pred.get('price_change_percent', 0)
                    reason = f"Mô hình dự đoán xu hướng tăng giá {abs(price_change):.1f}% trong tháng tới"
            elif pred['direction'] == 'down':
                if prediction_type == 'intraday':
                    price_change = pred.get('price_change_percent', 0)
                    reason = f"Mô hình dự đoán giá giảm {abs(price_change):.1f}%"
                elif prediction_type == 'five_day':
                    price_change = pred.get('price_change_percent', 0)
                    reason = f"Mô hình dự đoán giá giảm {abs(price_change):.1f}% trong 5 ngày tới"
                else:  # monthly
                    price_change = pred.get('price_change_percent', 0)
                    reason = f"Mô hình dự đoán xu hướng giảm giá {abs(price_change):.1f}% trong tháng tới"
            else:
                reason = "Dự đoán dựa trên mô hình ML"
        
        return {
            "emoji": direction_emoji,
            "price": price,
            "confidence": confidence,
            "reason": reason,
            "direction": pred['direction']
        }
    
    def _create_stock_embed(self, symbol, prediction_data):
        """
        Tạo embed thông tin dự đoán cho một cổ phiếu
        
        Args:
            symbol (str): Mã cổ phiếu
            prediction_data (dict): Dữ liệu dự đoán
            
        Returns:
            dict: Discord embed object
        """
        # Định dạng các dự đoán
        intraday_formatted = self._format_prediction(prediction_data, 'intraday')
        five_day_formatted = self._format_prediction(prediction_data, 'five_day')
        monthly_formatted = self._format_prediction(prediction_data, 'monthly')
        
        # Log để kiểm tra
        logger.info(f"DEBUG - {symbol} predictions: intraday={intraday_formatted is not None}, five_day={five_day_formatted is not None}, monthly={monthly_formatted is not None}")
    
        if intraday_formatted:
            logger.info(f"DEBUG - {symbol} intraday: direction={intraday_formatted['direction']}, price={intraday_formatted['price']}, confidence={intraday_formatted['confidence']}")

        # Xác định màu sắc embed dựa trên dự đoán intraday hoặc mặc định
        if intraday_formatted and intraday_formatted['direction'] == 'up':
            color = int(self.up_color.replace('#', ''), 16)
            direction_emoji = "🔼"
        elif intraday_formatted and intraday_formatted['direction'] == 'down':
            color = int(self.down_color.replace('#', ''), 16)
            direction_emoji = "🔽"
        else:
            color = int(self.neutral_color.replace('#', ''), 16)
            direction_emoji = "➡️"
        
        # Tạo tiêu đề
        title = f"{direction_emoji} Dự đoán giá {symbol}"
        
        # Tạo thông tin về giá hiện tại
        current_price = prediction_data.get('current_price', 'N/A')
        if isinstance(current_price, (int, float)):
            current_price = f"${current_price:.2f}"
        
        # Tạo nội dung cho từng mốc thời gian
        fields = []
        
        # Thêm thông tin intraday
        if intraday_formatted:
            intraday_field = {
                "name": f"{intraday_formatted['emoji']} Dự đoán Intraday",
                "value": f"Giá: {intraday_formatted['price']}\nĐộ tin cậy: {intraday_formatted['confidence']}\nLý do: {intraday_formatted['reason']}",
                "inline": True
            }
            fields.append(intraday_field)
        
        # Thêm thông tin 5 ngày
        if five_day_formatted:
            five_day_field = {
                "name": f"{five_day_formatted['emoji']} Dự đoán 5 ngày",
                "value": f"Giá: {five_day_formatted['price']}\nĐộ tin cậy: {five_day_formatted['confidence']}\nLý do: {five_day_formatted['reason']}",
                "inline": True
            }
            fields.append(five_day_field)
        
        # Thêm thông tin 1 tháng
        if monthly_formatted:
            monthly_field = {
                "name": f"{monthly_formatted['emoji']} Dự đoán 1 tháng",
                "value": f"Giá: {monthly_formatted['price']}\nĐộ tin cậy: {monthly_formatted['confidence']}\nLý do: {monthly_formatted['reason']}",
                "inline": True
            }
            fields.append(monthly_field)
        
        # Tạo footer với timestamp
        timestamp = datetime.now(self.market_timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
        
        # Tạo embed object
        embed = {
            "title": title,
            "description": f"Giá hiện tại: {current_price}",
            "color": color,
            "fields": fields,
            "footer": {
                "text": f"BondZiA AI • {timestamp}"
            }
        }
        
        # Thêm thumbnail nếu có ảnh biểu đồ
        if prediction_data.get('chart_url'):
            embed["thumbnail"] = {
                "url": prediction_data['chart_url']
            }
        
        return embed
    
    def send_prediction_message(self, predictions_data, group_size=3):
        """
        Gửi thông báo dự đoán giá cổ phiếu
        
        Args:
            predictions_data (dict): Dữ liệu dự đoán theo cấu trúc {symbol: prediction_data}
            group_size (int): Số lượng cổ phiếu gửi trong mỗi tin nhắn
            
        Returns:
            bool: True nếu gửi thành công, False nếu thất bại
        """
        if not self.notification_config['send_predictions']:
            logger.info("Bỏ qua thông báo dự đoán (đã tắt trong cấu hình)")
            return True
        
        # Log thông tin về các loại dự đoán có trong thông báo
        prediction_types_summary = {}
        for symbol, data in predictions_data.items():
            types = []
            if 'intraday' in data:
                types.append('intraday')
            if 'five_day' in data:
                types.append('five_day')
            if 'monthly' in data:
                types.append('monthly')
            prediction_types_summary[symbol] = types
        
        logger.info(f"Chuẩn bị gửi dự đoán với các khung thời gian: {prediction_types_summary}")

        try:
            # Chia nhỏ danh sách cổ phiếu thành các nhóm
            symbols = list(predictions_data.keys())
            symbol_groups = [symbols[i:i + group_size] for i in range(0, len(symbols), group_size)]
            
            success = True
            
            for group in symbol_groups:
                embeds = []
                
                for symbol in group:
                    prediction_data = predictions_data[symbol]
                    embed = self._create_stock_embed(symbol, prediction_data)
                    embeds.append(embed)
                
                # Tạo payload
                timestamp = datetime.now(self.market_timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
                payload = {
                    "content": f"**BondZiA AI - Dự đoán giá cổ phiếu ({timestamp})**",
                    "embeds": embeds
                }
                
                # Thêm log debug để kiểm tra dự đoán
                logger.info(f"Đang gửi dự đoán cho nhóm cổ phiếu: {group}")
                logger.info(f"Số lượng embeds: {len(embeds)}")
                for idx, embed in enumerate(embeds):
                    logger.info(f"Embed {idx+1}: {embed['title']}")

                # Gửi webhook
                if not self._send_webhook(self.prediction_webhook, payload):
                    logger.error(f"Không thể gửi thông báo dự đoán cho nhóm: {group}")
                    success = False
                else:
                    logger.info(f"Đã gửi thông báo dự đoán cho: {group}")
                
                # Đợi giữa các lần gửi để tránh rate limit
                time.sleep(1)
            
            return success
        except Exception as e:
            logger.error(f"Lỗi khi gửi thông báo dự đoán: {str(e)}")
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
        if is_error and not self.notification_config['send_errors']:
            logger.info("Bỏ qua thông báo lỗi (đã tắt trong cấu hình)")
            return True
        
        if not is_error and not self.notification_config['send_system_updates']:
            logger.info("Bỏ qua thông báo cập nhật (đã tắt trong cấu hình)")
            return True
        
        try:
            # Xác định màu sắc dựa trên loại thông báo
            if is_error:
                color = 0xFF0000  # Red for errors
                title = f"⚠️ {title}"
            else:
                color = 0x00FF00  # Green for updates
                title = f"📢 {title}"
            
            # Chuẩn bị các trường bổ sung
            embed_fields = []
            if fields:
                for field in fields:
                    embed_fields.append({
                        "name": field.get("name", ""),
                        "value": field.get("value", ""),
                        "inline": field.get("inline", False)
                    })
            
            # Tạo timestamp
            timestamp = datetime.now(self.market_timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
            
            # Tạo embed
            embed = {
                "title": title,
                "description": message,
                "color": color,
                "fields": embed_fields,
                "footer": {
                    "text": f"BondZiA AI • {timestamp}"
                }
            }
            
            # Tạo payload
            payload = {
                "embeds": [embed]
            }
            
            # Gửi webhook
            return self._send_webhook(self.update_webhook, payload)
        except Exception as e:
            logger.error(f"Lỗi khi gửi thông báo cập nhật hệ thống: {str(e)}")
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
        if not self.notification_config['send_system_updates']:
            logger.info("Bỏ qua thông báo tiến hóa (đã tắt trong cấu hình)")
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
                    "value": improvements_text,
                    "inline": False
                },
                {
                    "name": "Số lượng siêu tham số thay đổi",
                    "value": str(params_changed),
                    "inline": True
                },
                {
                    "name": "Hiệu suất",
                    "value": performance_text,
                    "inline": True
                }
            ]
            
            # Gửi thông báo
            return self.send_system_update(
                title=f"Tiến hóa AI thành công - Phiên bản {version}",
                message="BondZiA AI đã hoàn thành quá trình tiến hóa và nâng cấp lên phiên bản mới.",
                fields=fields
            )
        except Exception as e:
            logger.error(f"Lỗi khi gửi thông báo tiến hóa: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def send_chart_with_prediction(self, symbol, chart_data, prediction_data):
        """
        Tạo và gửi biểu đồ với dự đoán
        
        Args:
            symbol (str): Mã cổ phiếu
            chart_data (DataFrame): Dữ liệu biểu đồ
            prediction_data (dict): Dữ liệu dự đoán
            
        Returns:
            bool: True nếu gửi thành công, False nếu thất bại
        """
        try:
            # Tạo biểu đồ
            plt.figure(figsize=(10, 6))
            
            # Thiết lập style theo cấu hình
            if self.viz_config['theme'] == 'dark':
                plt.style.use('dark_background')
                
            # Vẽ dữ liệu lịch sử
            plt.plot(chart_data.index, chart_data['close'], label='Giá đóng cửa', color='white')
            
            # Thêm dự đoán
            predictions = []
            labels = []
            colors = []
            
            dates = []
            last_date = chart_data.index[-1]
            current_date = datetime.now().date()
            
            if 'intraday' in prediction_data:
                intraday = prediction_data['intraday']
                if 'direction' in intraday and 'confidence' in intraday:
                    price = intraday.get('predicted_price', intraday.get('price', intraday.get('predicted_value', 0)))
                    predictions.append(price)
                    labels.append(f"Intraday ({intraday['confidence']:.1f}%)")
                    colors.append(self.up_color if intraday['direction'] == 'up' else 
                                self.down_color if intraday['direction'] == 'down' else self.neutral_color)
                    # Thêm 4 giờ để dự đoán intraday
                    dates.append(pd.Timestamp(current_date) + pd.Timedelta(hours=4))
            
            if 'five_day' in prediction_data:
                five_day = prediction_data['five_day']
                if 'direction' in five_day and 'confidence' in five_day:
                    price = five_day.get('predicted_price', five_day.get('price', five_day.get('predicted_value', 0)))
                    predictions.append(price)
                    labels.append(f"5 ngày ({five_day['confidence']:.1f}%)")
                    colors.append(self.up_color if five_day['direction'] == 'up' else 
                                self.down_color if five_day['direction'] == 'down' else self.neutral_color)
                    # Thêm 5 ngày cho dự đoán 5 ngày
                    dates.append(pd.Timestamp(current_date) + pd.Timedelta(days=5))
            
            if 'monthly' in prediction_data:
                monthly = prediction_data['monthly']
                if 'direction' in monthly and 'confidence' in monthly:
                    price = monthly.get('predicted_price', monthly.get('price', monthly.get('predicted_value', 0)))
                    predictions.append(price)
                    labels.append(f"1 tháng ({monthly['confidence']:.1f}%)")
                    colors.append(self.up_color if monthly['direction'] == 'up' else 
                                self.down_color if monthly['direction'] == 'down' else self.neutral_color)
                    # Thêm 30 ngày cho dự đoán 1 tháng
                    dates.append(pd.Timestamp(current_date) + pd.Timedelta(days=30))
            
            # Vẽ các dự đoán
            for i in range(len(predictions)):
                plt.scatter(dates[i], predictions[i], color=colors[i], s=100, zorder=5)
                plt.annotate(labels[i], (dates[i], predictions[i]), 
                           textcoords="offset points", xytext=(0,10), ha='center')
            
            # Thêm đường từ điểm cuối đến dự đoán
            for i in range(len(predictions)):
                plt.plot([last_date, dates[i]], [chart_data['close'].iloc[-1], predictions[i]], 
                        '--', color=colors[i], alpha=0.7)
            
            # Thiết lập nhãn và tiêu đề
            current_date_str = datetime.now().strftime("%d/%m/%Y")
            plt.title(f"Dự đoán giá {symbol} (Ngày: {current_date_str})", fontsize=14)
            plt.xlabel("Thời gian")
            plt.ylabel("Giá ($)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Lưu biểu đồ
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Đảm bảo thư mục tồn tại
            charts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    "charts")
            os.makedirs(charts_dir, exist_ok=True)
            
            chart_path = os.path.join(charts_dir, f"{symbol}_prediction_{timestamp}.png")
            plt.savefig(chart_path)
            plt.close()
            
            # Gửi biểu đồ lên Discord
            with open(chart_path, 'rb') as f:
                chart_data = f.read()
            
            files = {
                'file': (f"{symbol}_prediction.png", chart_data)
            }
            
            # Tạo thông tin dự đoán
            message = f"**Biểu đồ dự đoán giá {symbol} (Ngày: {current_date_str})**"
            
            payload = {
                'content': message
            }

            try:    
                # Gửi biểu đồ lên Discord
                response = requests.post(
                    self.prediction_webhook, 
                    data=payload,
                    files=files
                )
            
                response.raise_for_status()
                logger.info(f"Đã gửi biểu đồ dự đoán cho {symbol} - HTTP Status: {response.status_code}")
                logger.debug(f"Discord response: {response.text[:100]}")  # Log 100 ký tự đầu tiên
            except Exception as chart_err:
                logger.error(f"Lỗi khi gửi biểu đồ lên Discord: {chart_err}")
                # Vẫn trả về True để không làm gián đoạn quá trình khác
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tạo và gửi biểu đồ dự đoán cho {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return False

if __name__ == "__main__":
    # Test module
    logger.info("Kiểm tra module Discord Notifier")
    notifier = DiscordNotifier()
    
    # Test gửi thông báo cập nhật hệ thống
    success = notifier.send_system_update(
        title="Khởi động BondZiA AI",
        message="Hệ thống BondZiA AI đã khởi động và sẵn sàng dự đoán giá cổ phiếu.",
        fields=[
            {
                "name": "Phiên bản",
                "value": "1.0.0",
                "inline": True
            },
            {
                "name": "Cổ phiếu theo dõi",
                "value": "TSLA, NVDA, PLTR, AGX, META, AMZN, AAPL, IBM, BABA",
                "inline": True
            }
        ]
    )
    
    logger.info(f"Gửi thông báo cập nhật: {'Thành công' if success else 'Thất bại'}")