import os
import sys
import json
import pandas as pd
from datetime import datetime
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("csv_to_json_converter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CSVtoJSONConverter")

# Đường dẫn thư mục gốc
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# Thư mục dữ liệu
data_raw_dir = os.path.join(root_dir, "data/raw")

def convert_csv_to_json():
    """Chuyển đổi các file CSV trong data/raw sang định dạng JSON chuẩn"""
    # Tìm tất cả file CSV
    csv_files = [f for f in os.listdir(data_raw_dir) if f.endswith('.csv')]
    logger.info(f"Tìm thấy {len(csv_files)} file CSV")

    if not csv_files:
        logger.warning("Không tìm thấy file CSV nào trong data/raw")
        return

    # Nhóm theo symbol
    symbol_files = {}
    for filename in csv_files:
        parts = filename.split('_')
        if len(parts) > 1:
            symbol = parts[0]
            if symbol not in symbol_files:
                symbol_files[symbol] = []
            symbol_files[symbol].append(filename)

    logger.info(f"Đã tìm thấy dữ liệu cho {len(symbol_files)} cổ phiếu")

    # Tạo file market_data JSON cho tất cả symbol
    market_data = {}
    
    for symbol, files in symbol_files.items():
        # Đọc file CSV mới nhất cho mỗi symbol
        latest_file = sorted(files)[-1]
        file_path = os.path.join(data_raw_dir, latest_file)
        
        try:
            logger.info(f"Đang đọc {file_path}")
            df = pd.read_csv(file_path)
            
            # Đóng gói dữ liệu theo định dạng mong muốn
            daily_data = {}
            for col in df.columns:
                if col in ['open', 'high', 'low', 'close', 'volume', 'date', 'timestamp']:
                    # Nếu là cột date hoặc timestamp, bỏ qua
                    if col in ['date', 'timestamp']:
                        continue
                    daily_data[col] = df[col].tolist()
            
            market_data[symbol] = {
                'symbol': symbol,
                'daily_data': daily_data,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Đã xử lý dữ liệu cho {symbol}")
        
        except Exception as e:
            logger.error(f"Lỗi khi xử lý {file_path}: {str(e)}")
            continue
    
    # Lưu file JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_filename = f"market_data_{timestamp}.json"
    json_path = os.path.join(data_raw_dir, json_filename)
    
    try:
        with open(json_path, 'w') as f:
            json.dump(market_data, f, indent=2)
        
        logger.info(f"Đã tạo file JSON: {json_path}")
        print(f"Đã chuyển đổi dữ liệu CSV sang JSON: {json_path}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu file JSON: {str(e)}")

if __name__ == "__main__":
    convert_csv_to_json()