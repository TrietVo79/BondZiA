# utils/generate_test_predictions.py
import os
import json
import random
from datetime import datetime, timedelta
import argparse
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_prediction_generator')

def generate_test_predictions(symbol, days=30, include_all_timeframes=True):
    """
    Tạo dữ liệu dự đoán giả để kiểm thử
    
    Args:
        symbol (str): Mã cổ phiếu
        days (int): Số ngày để tạo dự đoán (ngược về quá khứ từ hôm nay)
        include_all_timeframes (bool): Tạo dự đoán cho tất cả khung thời gian
    
    Returns:
        dict: Thông tin về dự đoán đã tạo
    """
    timeframes = ["intraday", "five_day", "monthly"] if include_all_timeframes else ["intraday"]
    
    # Tải dữ liệu giá để có giá thực tế
    data_file = f"data/raw/{symbol}_daily_2025-01-06_2025-04-04.csv"
    if not os.path.exists(data_file):
        logger.error(f"Không tìm thấy file dữ liệu giá: {data_file}")
        logger.info("Sẽ tạo dữ liệu ngẫu nhiên")
        use_real_prices = False
    else:
        import pandas as pd
        try:
            price_data = pd.read_csv(data_file)
            price_data['date'] = pd.to_datetime(price_data['date'])
            price_data = price_data.sort_values('date')
            use_real_prices = True
            logger.info(f"Đã tải dữ liệu giá thực tế từ {data_file}")
        except Exception as e:
            logger.error(f"Lỗi khi đọc file dữ liệu giá: {e}")
            use_real_prices = False
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    all_predictions = {}
    prediction_counts = {}
    
    for timeframe in timeframes:
        predictions = []
        prediction_counts[timeframe] = 0
        
        # Tính số ngày dự đoán
        skip_days = 1
        if timeframe == "five_day":
            skip_days = 2
        elif timeframe == "monthly":
            skip_days = 7
        
        # Tạo các dự đoán trong khoảng thời gian
        current_date = start_date
        while current_date <= end_date:
            try:
                # Dự đoán thực hiện vào ngày hiện tại
                prediction_date = current_date
                
                # Ngày mục tiêu dựa vào timeframe
                if timeframe == "intraday":
                    target_date = prediction_date
                elif timeframe == "five_day":
                    target_date = prediction_date + timedelta(days=5)
                else:  # monthly
                    target_date = prediction_date + timedelta(days=30)
                
                # Đảm bảo ngày mục tiêu không vượt quá hôm nay
                if target_date > end_date:
                    current_date += timedelta(days=skip_days)
                    continue
                
                # Lấy giá hiện tại và giá mục tiêu từ dữ liệu thực tế nếu có
                if use_real_prices:
                    try:
                        pred_day_data = price_data[price_data['date'] == prediction_date.strftime('%Y-%m-%d')]
                        target_day_data = price_data[price_data['date'] == target_date.strftime('%Y-%m-%d')]
                        
                        if len(pred_day_data) > 0 and len(target_day_data) > 0:
                            current_price = pred_day_data['close'].values[0]
                            actual_target_price = target_day_data['close'].values[0]
                            
                            # Tạo một dự đoán với độ chính xác ngẫu nhiên
                            accuracy = random.uniform(0.9, 1.1)  # 90-110% của giá thực tế
                            predicted_price = actual_target_price * accuracy
                            
                            # Đặt độ tin cậy dựa vào mức độ chính xác
                            error_percent = abs((predicted_price - actual_target_price) / actual_target_price) * 100
                            confidence = max(30, 90 - error_percent * 2)  # Độ tin cậy giảm khi sai số tăng
                        else:
                            # Nếu không tìm thấy dữ liệu, sinh ngẫu nhiên
                            current_price = random.uniform(100, 300)
                            direction = random.choice([0.9, 1.1])  # Giảm hoặc tăng 10%
                            predicted_price = current_price * direction
                            confidence = random.randint(40, 80)
                    except Exception as e:
                        logger.warning(f"Lỗi khi lấy giá từ dữ liệu: {e}")
                        current_price = random.uniform(100, 300)
                        direction = random.choice([0.9, 1.1])
                        predicted_price = current_price * direction
                        confidence = random.randint(40, 80)
                else:
                    # Sinh ngẫu nhiên nếu không có dữ liệu thực tế
                    current_price = random.uniform(100, 300)
                    direction = random.choice([0.9, 1.1])
                    predicted_price = current_price * direction
                    confidence = random.randint(40, 80)
                
                # Tạo dự đoán
                prediction = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "prediction_date": prediction_date.isoformat(),
                    "target_date": target_date.isoformat(),
                    "current_price": current_price,
                    "predicted_price": predicted_price,
                    "confidence": confidence,
                    "direction": "up" if predicted_price > current_price else "down",
                    "prediction_id": f"{symbol}_{timeframe}_{prediction_date.strftime('%Y%m%d%H%M%S')}"
                }
                
                predictions.append(prediction)
                prediction_counts[timeframe] += 1
            except Exception as e:
                logger.error(f"Lỗi khi tạo dự đoán: {e}")
                
            # Tăng ngày hiện tại
            current_date += timedelta(days=skip_days)
        
        # Lưu file dự đoán
        os.makedirs("prediction_history", exist_ok=True)
        
        # Tạo file cho từng ngày dự đoán
        prediction_dates = {}
        for pred in predictions:
            pred_date = pred["prediction_date"].split('T')[0]
            if pred_date not in prediction_dates:
                prediction_dates[pred_date] = []
            prediction_dates[pred_date].append(pred)
        
        # Lưu file cho mỗi ngày dự đoán
        for date, preds in prediction_dates.items():
            date_str = date.replace('-', '')
            output_file = f"prediction_history/{symbol}_prediction_{date_str}_120000.json"
            
            with open(output_file, 'w') as f:
                json.dump(preds, f, indent=4)
            
            logger.info(f"Đã lưu {len(preds)} dự đoán {timeframe} cho {symbol} vào {output_file}")
        
        all_predictions[timeframe] = predictions
    
    # Log thông tin
    logger.info(f"Đã tạo dự đoán cho {symbol}:")
    for tf, count in prediction_counts.items():
        logger.info(f"- {tf}: {count} dự đoán")
    
    return {
        "symbol": symbol,
        "prediction_counts": prediction_counts,
        "total_files": sum(1 for _ in os.listdir("prediction_history") if _.startswith(f"{symbol}_prediction_"))
    }

def main():
   parser = argparse.ArgumentParser(description='Tạo dữ liệu dự đoán giả để kiểm thử')
   parser.add_argument('--symbol', required=True, help='Mã cổ phiếu (vd: NVDA)')
   parser.add_argument('--days', type=int, default=30, help='Số ngày cần tạo dự đoán (mặc định: 30)')
   parser.add_argument('--timeframe-only', choices=['intraday', 'five_day', 'monthly'], 
                     help='Chỉ tạo dự đoán cho một khung thời gian cụ thể')
   
   args = parser.parse_args()
   
   include_all = args.timeframe_only is None
   result = generate_test_predictions(args.symbol, args.days, include_all)
   
   logger.info(f"Hoàn thành tạo dữ liệu dự đoán giả cho {args.symbol}")
   logger.info(f"Tổng số file: {result['total_files']}")
   
   return 0

if __name__ == "__main__":
   main()