import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import argparse
import traceback

# Đường dẫn thư mục gốc
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# Import các module cần thiết
try:
    from utils.logger_config import logger
except ImportError:
    # Thiết lập logging cơ bản nếu không import được
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("accuracy_evaluator.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("AccuracyEvaluator")

class AccuracyEvaluator:
    """Module đánh giá độ chính xác và điều chỉnh mô hình tự động"""
    
    def __init__(self, prediction_history_dir="prediction_history", 
                 data_dir="data/raw", 
                 accuracy_stats_dir="prediction_stats",
                 config_path="config/system_config.json"):
        
        self.prediction_history_dir = os.path.join(root_dir, prediction_history_dir)
        self.data_dir = os.path.join(root_dir, data_dir)
        self.accuracy_stats_dir = os.path.join(root_dir, accuracy_stats_dir)
        self.config_path = os.path.join(root_dir, config_path)
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.accuracy_stats_dir, exist_ok=True)
        
        # Tải cấu hình nếu tồn tại
        self.config = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.error(f"Lỗi khi tải cấu hình: {str(e)}")
        
        logger.info("Khởi tạo AccuracyEvaluator thành công")
    
    def evaluate_symbol(self, symbol, timeframe='all', days=30):
        """Đánh giá độ chính xác cho một mã cổ phiếu"""
        logger.info(f"Đánh giá độ chính xác cho {symbol} trong {days} ngày qua")
        
        # Lấy dữ liệu dự đoán và giá thực tế
        predictions = self._load_historical_predictions(symbol, days)
        actual_data = self._load_actual_prices(symbol, days)
        
        if not predictions:
            logger.warning(f"Không tìm thấy dự đoán cho {symbol}")
            return None
        
        if not actual_data:
            logger.warning(f"Không tìm thấy giá thực tế cho {symbol}")
            return None
        
        # Đánh giá từng timeframe
        results = {}
        
        timeframes = ['intraday', 'five_day', 'monthly'] if timeframe == 'all' else [timeframe]
        
        for tf in timeframes:
            logger.info(f"Đang đánh giá {symbol} - {tf}")
            
            # Khớp dự đoán với giá thực tế
            matched_data = self._match_predictions_with_actual(predictions, actual_data, tf)
            
            if not matched_data:
                logger.warning(f"Không đủ dữ liệu để đánh giá {symbol} - {tf}")
                continue
            
            # Tính các metrics
            metrics = self._calculate_accuracy_metrics(matched_data)
            
            if metrics:
                results[tf] = metrics
        
        if not results:
            logger.warning(f"Không thể đánh giá bất kỳ timeframe nào cho {symbol}")
            return None
        
        # Lưu kết quả đánh giá
        self._save_evaluation_results(symbol, results)
        
        # Áp dụng việc tự động điều chỉnh
        self._auto_adjust_models(symbol, results)
        
        return results
    
    def _load_historical_predictions(self, symbol, days=30):
        """Tải lịch sử dự đoán cho một mã cổ phiếu"""
        predictions = []
        
        # Xác định khoảng thời gian
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Tìm tất cả file lịch sử dự đoán
        for filename in os.listdir(self.prediction_history_dir):
            if symbol in filename:
                file_path = os.path.join(self.prediction_history_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        pred_data = json.load(f)
                    
                    # Kiểm tra xem dự đoán có trong khoảng thời gian không
                    if 'timestamp' in pred_data:
                        pred_time = datetime.fromisoformat(pred_data['timestamp'])
                        if start_date <= pred_time <= end_date:
                            predictions.append(pred_data)
                except Exception as e:
                    logger.error(f"Lỗi khi đọc file {file_path}: {str(e)}")
        
        logger.info(f"Đã tải {len(predictions)} dự đoán cho {symbol}")
        return predictions
    

    def _load_actual_prices(self, symbol, days=30):
        """Tải giá thực tế từ dữ liệu thị trường"""
        actual_data = []
        
        # Tính thời gian bắt đầu
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Tìm kiếm giá thực tế cho {symbol} từ {start_date.date()} đến {end_date.date()}")
        
        # Kiểm tra file CSV trước
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv') and filename.startswith(f"{symbol}_daily"):
                file_path = os.path.join(self.data_dir, filename)
                try:
                    logger.info(f"Đang đọc file CSV: {filename}")
                    # Đọc file CSV
                    df = pd.read_csv(file_path)
                    
                    # Chuyển đổi cột 'date' hoặc 'timestamp' thành datetime
                    date_column = None
                    for col in ['date', 'timestamp']:
                        if col in df.columns:
                            date_column = col
                            break
                    
                    if date_column and 'close' in df.columns:
                        # Chuyển đổi thành datetime nếu cần
                        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                            df[date_column] = pd.to_datetime(df[date_column])
                        
                        # Lọc theo khoảng thời gian
                        if start_date and end_date:
                            df_filtered = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
                        else:
                            df_filtered = df
                        
                        # Thêm vào actual_data
                        for _, row in df_filtered.iterrows():
                            actual_data.append({
                                'timestamp': row[date_column].isoformat() if isinstance(row[date_column], datetime) else row[date_column],
                                'price': row['close']
                            })
                        
                        logger.info(f"Đã tìm thấy {len(df_filtered)} record trong file {filename}")
                    else:
                        logger.warning(f"File {filename} không có cột date/timestamp hoặc close")
                except Exception as e:
                    logger.error(f"Lỗi khi đọc file CSV {file_path}: {str(e)}")
                    logger.error(traceback.format_exc())
        
        # Nếu không tìm thấy dữ liệu từ CSV, thử tìm trong JSON
        if not actual_data:
            logger.info("Không tìm thấy dữ liệu từ CSV, đang tìm trong JSON...")
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.json') and filename.startswith('market_data_'):
                    file_path = os.path.join(self.data_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            market_data = json.load(f)
                        
                        # Lấy timestamp từ tên file
                        time_str = filename.replace('market_data_', '').replace('.json', '')
                        try:
                            time_obj = datetime.strptime(time_str, '%Y%m%d_%H%M%S')
                        except ValueError:
                            logger.warning(f"Không thể parse timestamp từ tên file: {filename}")
                            continue
                        
                        # Chỉ xem xét file trong khoảng thời gian cần đánh giá
                        if time_obj < start_date or time_obj > end_date:
                            continue
                        
                        if symbol not in market_data:
                            continue
                        
                        logger.info(f"Tìm thấy dữ liệu cho {symbol} trong file {filename}")
                        
                        # Kiểm tra cấu trúc dữ liệu (debug)
                        symbol_data = market_data[symbol]
                        data_keys = list(symbol_data.keys())
                        logger.info(f"Các khóa dữ liệu có sẵn: {data_keys}")
                        
                        # Trích xuất giá đóng cửa từ nhiều cấu trúc dữ liệu khả thi
                        price = None
                        
                        # Cấu trúc 1: daily_data là dict có chứa close
                        if 'daily_data' in symbol_data and isinstance(symbol_data['daily_data'], dict) and 'close' in symbol_data['daily_data']:
                            close_data = symbol_data['daily_data']['close']
                            if isinstance(close_data, list) and close_data:
                                price = close_data[-1]  # Lấy giá đóng cửa mới nhất
                            else:
                                price = close_data
                            logger.info(f"Tìm thấy giá từ daily_data[close]: {price}")
                        
                        # Cấu trúc 2: daily_data là list các record
                        elif 'daily_data' in symbol_data and isinstance(symbol_data['daily_data'], list) and symbol_data['daily_data']:
                            latest_record = symbol_data['daily_data'][-1]
                            if isinstance(latest_record, dict) and 'close' in latest_record:
                                price = latest_record['close']
                                logger.info(f"Tìm thấy giá từ daily_data[-1][close]: {price}")
                        
                        # Cấu trúc 3: intraday_data
                        elif 'intraday_data' in symbol_data:
                            intraday_data = symbol_data['intraday_data']
                            if isinstance(intraday_data, dict) and 'close' in intraday_data:
                                close_data = intraday_data['close']
                                price = close_data[-1] if isinstance(close_data, list) and close_data else close_data
                                logger.info(f"Tìm thấy giá từ intraday_data[close]: {price}")
                            elif isinstance(intraday_data, list) and intraday_data:
                                latest_record = intraday_data[-1]
                                if isinstance(latest_record, dict) and 'close' in latest_record:
                                    price = latest_record['close']
                                    logger.info(f"Tìm thấy giá từ intraday_data[-1][close]: {price}")
                        
                        # Cấu trúc 4: latest_quote
                        elif 'latest_quote' in symbol_data:
                            latest_quote = symbol_data['latest_quote']
                            if isinstance(latest_quote, dict):
                                # Polygon API sử dụng 'p' cho price
                                if 'p' in latest_quote:
                                    price = latest_quote['p']
                                    logger.info(f"Tìm thấy giá từ latest_quote[p]: {price}")
                                # Cũng kiểm tra 'close' cho trường hợp định dạng khác
                                elif 'close' in latest_quote:
                                    price = latest_quote['close']
                                    logger.info(f"Tìm thấy giá từ latest_quote[close]: {price}")
                        
                        # Nếu tìm thấy giá, thêm vào danh sách
                        if price is not None:
                            actual_data.append({
                                'timestamp': time_obj.isoformat(),
                                'price': float(price)
                            })
                            logger.info(f"Đã thêm giá {price} vào actual_data")
                        else:
                            logger.warning(f"Không tìm thấy giá trong file {filename}")
                    
                    except Exception as e:
                        logger.error(f"Lỗi khi đọc file JSON {file_path}: {str(e)}")
                        logger.error(traceback.format_exc())
        
        # Sắp xếp theo thời gian
        actual_data.sort(key=lambda x: x['timestamp'])
        
        logger.info(f"Đã tải {len(actual_data)} giá thực tế cho {symbol}")
        
        # Nếu vẫn không tìm thấy dữ liệu, log rõ ràng và liệt kê các file đã tìm kiếm
        if not actual_data:
            logger.error(f"KHÔNG TÌM THẤY DỮ LIỆU GIÁ cho {symbol}. Các file trong thư mục:")
            for filename in os.listdir(self.data_dir):
                logger.info(f" - {filename}")
        
        return actual_data
    
    def _match_predictions_with_actual(self, predictions, actual_data, timeframe):
        """Khớp dự đoán với giá thực tế"""
        matched_results = []
        
        # Chuyển actual_data thành dict để tìm kiếm nhanh hơn
        actual_by_date = {}
        for actual in actual_data:
            date_str = actual['timestamp'].split('T')[0]  # Lấy phần ngày
            actual_by_date[date_str] = actual
        
        logger.info(f"Đã chuyển đổi {len(actual_data)} giá thực tế thành dict theo ngày")
        
        for pred in predictions:
            if timeframe not in pred:
                continue
                
            pred_time = datetime.fromisoformat(pred['timestamp'])
            pred_price = pred[timeframe].get('price', pred[timeframe].get('predicted_price'))
            pred_confidence = pred[timeframe].get('confidence')
            
            if pred_price is None:
                continue
            
            # Xác định thời điểm cần tìm giá thực tế
            target_time = None
            if timeframe == 'intraday':
                # Giá đóng cửa cùng ngày
                target_time = pred_time.replace(hour=16, minute=0, second=0)
            elif timeframe == 'five_day':
                # Giá sau 5 ngày giao dịch (~ 7 ngày lịch)
                target_time = pred_time + timedelta(days=7)
            elif timeframe == 'monthly':
                # Giá sau 30 ngày
                target_time = pred_time + timedelta(days=30)
            
            logger.info(f"Dự đoán {timeframe} từ {pred_time} mục tiêu tại {target_time}")
            
            # Tìm giá thực tế gần nhất cho ngày dự đoán (thử chính xác ngày trước)
            target_date_str = target_time.strftime('%Y-%m-%d')
            found_exact = False
            
            if target_date_str in actual_by_date:
                closest_actual = actual_by_date[target_date_str]
                found_exact = True
            else:
                # Nếu không tìm thấy chính xác ngày, tìm ngày gần nhất
                closest_actual = None
                min_diff = timedelta(days=999)
                
                for actual in actual_data:
                    actual_time = datetime.fromisoformat(actual['timestamp'])
                    diff = abs(actual_time - target_time)
                    
                    if diff < min_diff:
                        min_diff = diff
                        closest_actual = actual
            
            # Kiểm tra xem có tìm thấy giá và chênh lệch thời gian có hợp lý không
            if closest_actual is not None:
                actual_time = datetime.fromisoformat(closest_actual['timestamp'])
                time_diff = abs(actual_time - target_time)
                
                matched_results.append({
                    'pred_time': pred_time,
                    'target_time': target_time,
                    'actual_time': actual_time,
                    'pred_price': pred_price,
                    'actual_price': closest_actual['price'],
                    'confidence': pred_confidence,
                    'is_future_prediction': target_time > datetime.now(),
                    'time_diff_days': time_diff.days,
                    'exact_date_match': found_exact
                })
                
                log_msg = f"Khớp: dự đoán ${pred_price:.2f} với giá thực tế ${closest_actual['price']:.2f}"
                if found_exact:
                    log_msg += " (khớp chính xác ngày)"
                else:
                    log_msg += f" (chênh lệch {time_diff.days} ngày)"
                logger.info(log_msg)
        
        logger.info(f"Đã khớp {len(matched_results)} dự đoán với giá thực tế")
        return matched_results
    
    def _calculate_accuracy_metrics(self, matched_data):
        """Tính toán các metrics đánh giá độ chính xác"""
        if not matched_data:
            return None
        
        metrics = {
            'sample_count': len(matched_data),
            'mae': 0,  # Mean Absolute Error
            'mape': 0,  # Mean Absolute Percentage Error
            'rmse': 0,  # Root Mean Squared Error
            'direction_accuracy': 0,  # Accuracy of price direction
            'high_confidence_accuracy': 0,  # Accuracy of high confidence predictions
            'confidence_correlation': 0  # Correlation between confidence and accuracy
        }
        
        # Tính các metrics cơ bản
        errors = []
        pct_errors = []
        direction_correct = 0
        
        for i, data in enumerate(matched_data):
            # Lỗi tuyệt đối
            error = abs(data['actual_price'] - data['pred_price'])
            errors.append(error)
            
            # Lỗi phần trăm
            pct_error = error / data['actual_price'] * 100
            pct_errors.append(pct_error)
            
            # Kiểm tra hướng
            if i > 0:
                prev_price = matched_data[i-1]['actual_price']
                actual_direction = data['actual_price'] > prev_price
                pred_direction = data['pred_price'] > prev_price
                
                if actual_direction == pred_direction:
                    direction_correct += 1
        
        # Tính MAE
        metrics['mae'] = sum(errors) / len(errors)
        
        # Tính MAPE
        metrics['mape'] = sum(pct_errors) / len(pct_errors)
        
        # Tính RMSE
        metrics['rmse'] = np.sqrt(sum(e**2 for e in errors) / len(errors))
        
        # Tính độ chính xác hướng
        if len(matched_data) > 1:
            metrics['direction_accuracy'] = direction_correct / (len(matched_data) - 1) * 100
        
        # Đánh giá dự đoán có độ tin cậy cao
        high_conf_indices = [i for i, data in enumerate(matched_data) 
                            if data.get('confidence') is not None and data['confidence'] > 70]
        
        if high_conf_indices and len(high_conf_indices) > 1:
            # Tính độ chính xác theo hướng cho các dự đoán có độ tin cậy cao
            high_conf_correct = 0
            
            for i in high_conf_indices:
                if i > 0:
                    prev_price = matched_data[i-1]['actual_price']
                    actual_direction = matched_data[i]['actual_price'] > prev_price
                    pred_direction = matched_data[i]['pred_price'] > prev_price
                    
                    if actual_direction == pred_direction:
                        high_conf_correct += 1
            
            metrics['high_confidence_accuracy'] = high_conf_correct / (len(high_conf_indices) - 1) * 100
        
        # Tính tương quan giữa độ tin cậy và độ chính xác (biểu thị bằng inverse của sai số)
        confidences = [data.get('confidence', 0) for data in matched_data]
        if any(confidences) and len(confidences) == len(errors):
            # Sử dụng inverse của lỗi (1/error) cho tương quan
            inverse_errors = [1/(e+0.001) for e in errors]  # Thêm 0.001 để tránh chia cho 0
            
            # Tính hệ số tương quan
            confidence_mean = sum(confidences) / len(confidences)
            inv_error_mean = sum(inverse_errors) / len(inverse_errors)
            
            numerator = sum((c - confidence_mean) * (e - inv_error_mean) for c, e in zip(confidences, inverse_errors))
            denominator = (sum((c - confidence_mean)**2 for c in confidences) * 
                          sum((e - inv_error_mean)**2 for e in inverse_errors))**0.5
            
            if denominator != 0:
                metrics['confidence_correlation'] = numerator / denominator
        
        return metrics
    
    def _save_evaluation_results(self, symbol, results):
        """Lưu kết quả đánh giá vào file"""
        # Thêm timestamp
        results['timestamp'] = datetime.now().isoformat()
        results['symbol'] = symbol
        
        # Tạo tên file
        filename = f"{symbol}_accuracy_{datetime.now().strftime('%Y%m%d')}.json"
        file_path = os.path.join(self.accuracy_stats_dir, filename)
        
        # Lưu file
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Đã lưu kết quả đánh giá vào {file_path}")
        
        # Cập nhật tệp tổng hợp
        summary_file = os.path.join(self.accuracy_stats_dir, f"{symbol}_accuracy_summary.json")
        
        try:
            # Tải dữ liệu cũ nếu có
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
            else:
                summary_data = {'history': []}
            
            # Thêm kết quả mới
            summary_data['history'].append(results)
            
            # Giới hạn lịch sử (giữ 30 đánh giá gần nhất)
            if len(summary_data['history']) > 30:
                summary_data['history'] = summary_data['history'][-30:]
            
            # Tính trung bình
            summary_data['latest'] = results
            
            # Lưu file
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=4)
            
            logger.info(f"Đã cập nhật tệp tổng hợp {summary_file}")
        
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật tệp tổng hợp: {str(e)}")
    
    def _auto_adjust_models(self, symbol, results):
        """Điều chỉnh mô hình dựa trên kết quả đánh giá"""
        try:
            # Xử lý trường hợp results không phải là dict
            if not isinstance(results, dict):
                logger.warning(f"Kết quả đánh giá không phải là dictionary: {results}")
                return

            # Loại bỏ các trường không phải là dict trước khi xử lý
            timeframes_dict = {}
            for key, value in results.items():
                if isinstance(value, dict) and key in ['intraday', 'five_day', 'monthly']:
                    timeframes_dict[key] = value
            
            if not timeframes_dict:
                logger.warning(f"Không tìm thấy dữ liệu timeframe hợp lệ trong kết quả")
                return
                
            # Kiểm tra cấu hình cho việc tự động điều chỉnh
            if 'confidence' not in self.config:
                logger.warning("Không tìm thấy cấu hình cho việc tự động điều chỉnh")
                return
            
            conf_config = self.config.get('confidence', {})
            
            # Xác định xem có nên điều chỉnh không
            if not conf_config.get('auto_adjust', True):
                logger.info("Tự động điều chỉnh mô hình bị tắt trong cấu hình")
                return
            
            # Kiểm tra từng timeframe
            for timeframe, metrics in timeframes_dict.items():
                # Bỏ qua nếu metrics không phải là dictionary
                if not isinstance(metrics, dict):
                    logger.warning(f"Metrics cho {symbol} - {timeframe} không phải là dictionary: {metrics}")
                    continue
                    
                # Bỏ qua nếu không có sample_count
                if 'sample_count' not in metrics:
                    logger.warning(f"Không tìm thấy sample_count trong metrics cho {symbol} - {timeframe}")
                    continue
                    
                # Bỏ qua nếu không đủ dữ liệu
                if metrics['sample_count'] < conf_config.get('min_predictions_for_stats', 10):
                    logger.info(f"Không đủ dữ liệu cho {symbol} - {timeframe}")
                    continue
                
                # Kiểm tra độ chính xác
                direction_accuracy = metrics.get('direction_accuracy', 0)
                
                # Ngưỡng điều chỉnh
                threshold = conf_config.get('adjustment_threshold', 60)
                
                # Đánh giá cần điều chỉnh
                needs_adjustment = False
                adjustment_reason = ""
                
                if direction_accuracy < threshold:
                    needs_adjustment = True
                    adjustment_reason = f"Độ chính xác theo hướng thấp ({direction_accuracy:.1f}%)"
                
                # Kiểm tra tương quan độ tin cậy
                conf_corr = metrics.get('confidence_correlation', 0)
                if conf_corr < 0.2:  # Tương quan yếu
                    needs_adjustment = True
                    adjustment_reason += f"; Tương quan độ tin cậy yếu ({conf_corr:.2f})"
                
                # Thực hiện điều chỉnh nếu cần
                if needs_adjustment:
                    # Tạo lệnh tiến hóa mô hình
                    adjustment_file = os.path.join(self.accuracy_stats_dir, "adjustment_needed.json")
                    
                    adjustment_data = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'reason': adjustment_reason,
                        'metrics': metrics,
                        'status': 'pending'
                    }
                    
                    # Lưu file
                    with open(adjustment_file, 'a') as f:
                        f.write(json.dumps(adjustment_data) + '\n')
                    
                    logger.info(f"Đánh dấu {symbol} - {timeframe} cần điều chỉnh: {adjustment_reason}")
                else:
                    logger.info(f"{symbol} - {timeframe} không cần điều chỉnh (độ chính xác: {direction_accuracy:.1f}%)")
        
        except Exception as e:
            logger.error(f"Lỗi khi tự động điều chỉnh mô hình: {str(e)}")
            logger.error(traceback.format_exc())
    
    def evaluate_all_symbols(self, days=30):
        """Đánh giá tất cả các cổ phiếu"""
        # Lấy danh sách cổ phiếu từ prediction_history
        symbols = set()
        
        for filename in os.listdir(self.prediction_history_dir):
            if '_' in filename:
                symbol = filename.split('_')[0]
                symbols.add(symbol)
        
        logger.info(f"Tìm thấy {len(symbols)} cổ phiếu cần đánh giá: {', '.join(symbols)}")
        
        results = {}
        for symbol in symbols:
            symbol_results = self.evaluate_symbol(symbol, days=days)
            if symbol_results:
                results[symbol] = symbol_results
        
        return results


    def inspect_market_data_files(self, symbol=None):
        """
        Kiểm tra cấu trúc các file dữ liệu thị trường trong thư mục data/raw
        
        Args:
            symbol (str, optional): Mã cổ phiếu cần kiểm tra. Nếu None, kiểm tra tất cả.
            
        Returns:
            dict: Thông tin về cấu trúc các file
        """
        results = {'csv_files': [], 'json_files': []}
        
        # Kiểm tra các file CSV
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        logger.info(f"Tìm thấy {len(csv_files)} file CSV")
        
        for filename in csv_files:
            file_path = os.path.join(self.data_dir, filename)
            try:
                if symbol is None or symbol in filename:
                    df = pd.read_csv(file_path, nrows=1)
                    results['csv_files'].append({
                        'filename': filename,
                        'columns': list(df.columns),
                        'records': len(pd.read_csv(file_path))
                    })
            except Exception as e:
                logger.error(f"Lỗi khi kiểm tra file CSV {file_path}: {str(e)}")
        
        # Kiểm tra các file JSON
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        logger.info(f"Tìm thấy {len(json_files)} file JSON")
        
        for filename in json_files:
            file_path = os.path.join(self.data_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                file_info = {'filename': filename, 'symbols': list(data.keys())}
                
                # Kiểm tra chi tiết cho symbol cụ thể
                if symbol is not None and symbol in data:
                    symbol_data = data[symbol]
                    file_info['structure'] = {k: type(v).__name__ for k, v in symbol_data.items()}
                    
                    # Kiểm tra cấu trúc chi tiết hơn
                    if 'daily_data' in symbol_data:
                        daily_data = symbol_data['daily_data']
                        if isinstance(daily_data, dict):
                            file_info['daily_data_keys'] = list(daily_data.keys())
                        elif isinstance(daily_data, list) and daily_data:
                            file_info['daily_data_sample'] = daily_data[0]
                    
                    if 'intraday_data' in symbol_data:
                        intraday_data = symbol_data['intraday_data']
                        if isinstance(intraday_data, dict):
                            file_info['intraday_data_keys'] = list(intraday_data.keys())
                        elif isinstance(intraday_data, list) and intraday_data:
                            file_info['intraday_data_sample'] = intraday_data[0]
                
                results['json_files'].append(file_info)
            except Exception as e:
                logger.error(f"Lỗi khi kiểm tra file JSON {file_path}: {str(e)}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='BondZiA Accuracy Evaluator')
    parser.add_argument('--symbol', type=str, help='Mã cổ phiếu cần đánh giá (bỏ trống để đánh giá tất cả)')
    parser.add_argument('--timeframe', type=str, default='all', 
                        choices=['all', 'intraday', 'five_day', 'monthly'], 
                        help='Khung thời gian cần đánh giá')
    parser.add_argument('--days', type=int, default=30, help='Số ngày cần đánh giá')
    parser.add_argument('--inspect', action='store_true', help='Kiểm tra cấu trúc file dữ liệu')
    
    args = parser.parse_args()
    
    evaluator = AccuracyEvaluator()
    
    if args.inspect:
        # Kiểm tra cấu trúc file
        results = evaluator.inspect_market_data_files(args.symbol)
        print(json.dumps(results, indent=2, default=str))
        return
    
    if args.symbol:
        evaluator.evaluate_symbol(args.symbol, args.timeframe, args.days)
    else:
        evaluator.evaluate_all_symbols(args.days)

if __name__ == "__main__":
    main()