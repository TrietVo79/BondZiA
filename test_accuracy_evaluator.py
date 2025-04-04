# test_accuracy_evaluator.py
import sys
import os
import logging
from datetime import datetime

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_accuracy')

# Thêm đường dẫn để có thể import từ thư mục gốc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import module cần kiểm thử
from evolution.accuracy_evaluator import AccuracyEvaluator

def main():
    """Kiểm thử module AccuracyEvaluator"""
    
    # Chọn một cổ phiếu có dữ liệu để kiểm thử
    symbols = ["NVDA", "TSLA", "MSFT", "AMZN", "PLTR"]
    
    for symbol in symbols:
        logger.info(f"===== Kiểm thử đánh giá cho {symbol} =====")
        evaluator = AccuracyEvaluator()
        results = evaluator.evaluate_symbol(symbol)  # Thay đổi từ evaluate() sang evaluate_symbol()
        
        logger.info(f"Kết quả: {results}")
        logger.info(f"===== Kết thúc kiểm thử cho {symbol} =====\n")

if __name__ == "__main__":
    logger.info("Bắt đầu kiểm thử AccuracyEvaluator")
    main()
    logger.info("Kết thúc kiểm thử")