import os
from loguru import logger

# Xóa cấu hình logger mặc định
logger.remove()

# Thêm cấu hình mới với rotation
log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(log_path, exist_ok=True)
logger.add(
    os.path.join(log_path, "bondzia_{time:YYYY-MM-DD}.log"),
    rotation="00:00",  # Xoay file mỗi ngày vào lúc 00:00
    retention="14 days",  # Giữ log trong 14 ngày
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)