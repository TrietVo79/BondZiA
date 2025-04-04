import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import logging
import glob

# Lấy đường dẫn tuyệt đối
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # Đi lên một cấp từ thư mục evolution

# Xây dựng đường dẫn tuyệt đối
prediction_history_dir = os.path.join(root_dir, "prediction_history")
data_dir = os.path.join(root_dir, "data/raw")
output_dir = os.path.join(root_dir, "reports")

class PerformanceReporter:
    def __init__(self, prediction_history_dir=prediction_history_dir, 
                 data_dir=data_dir, output_dir=output_dir):
        self.prediction_history_dir = prediction_history_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(output_dir, exist_ok=True)
        
        # Thiết lập logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{output_dir}/performance_reporter.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("PerformanceReporter")
    
    def load_historical_predictions(self, symbol, days=30):
        """Tải lịch sử dự đoán cho một mã cổ phiếu"""
        self.logger.info(f"Đang tải lịch sử dự đoán cho {symbol} trong {days} ngày qua")
        
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
                    self.logger.error(f"Lỗi khi đọc file {file_path}: {str(e)}")
        
        self.logger.info(f"Đã tải {len(predictions)} dự đoán cho {symbol}")
        return predictions
    
    def load_actual_prices(self, symbol, days=30):
        """
        Hàm tải giá thực tế từ nhiều nguồn dữ liệu khác nhau.
        Hỗ trợ cả file CSV và JSON.
        """
        # Xác định khoảng thời gian
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Mảng lưu dữ liệu giá từ các nguồn
        actual_prices = []
        
        # 1. Kiểm tra các file market_data JSON gần đây nhất
        json_files = sorted(glob.glob(os.path.join(self.data_dir, "market_data_*.json")), reverse=True)
        
        for json_file in json_files[:10]:  # Chỉ xem 10 file gần nhất để tối ưu hiệu suất
            try:
                with open(json_file, 'r') as f:
                    market_data = json.load(f)
                    
                # Kiểm tra cấu trúc JSON và trích xuất giá
                # Thử nhiều format có thể có
                try:
                    # Format cũ
                    timestamp = datetime.strptime(json_file.split('_')[-1].split('.')[0], '%Y%m%d%H%M%S')
                except ValueError:
                    try:
                        # Format mới: nếu tên file là market_data_20250404_160104.json
                        parts = json_file.split('_')
                        date_part = parts[-2]  # 20250404
                        time_part = parts[-1].split('.')[0]  # 160104
                        timestamp_str = f"{date_part}{time_part}"
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
                    except Exception as e:
                        self.logger.error(f"Không thể đọc timestamp từ file {json_file}: {str(e)}")
                        continue
                
                # Format 1: Dạng danh sách symbols
                if isinstance(market_data, list):
                    for item in market_data:
                        if item.get('symbol') == symbol:
                            actual_prices.append({
                                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                'price': item.get('price', item.get('current_price', 0)),
                                'source': 'market_data_json_list'
                            })
                            break
                
                # Format 2: Dạng dictionary với key là symbol
                elif isinstance(market_data, dict):
                    if symbol in market_data:
                        actual_prices.append({
                            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            'price': market_data[symbol].get('price', market_data[symbol].get('current_price', 0)),
                            'source': 'market_data_json_dict'
                        })
                    # Format 3: Dạng dictionary với symbols là một trường
                    elif 'symbols' in market_data and symbol in market_data['symbols']:
                        actual_prices.append({
                            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            'price': market_data['symbols'][symbol].get('price', market_data['symbols'][symbol].get('current_price', 0)),
                            'source': 'market_data_json_symbols'
                        })
                    # Format 4: Dạng dictionary với trường data
                    elif 'data' in market_data:
                        if isinstance(market_data['data'], list):
                            for item in market_data['data']:
                                if item.get('symbol') == symbol:
                                    actual_prices.append({
                                        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                        'price': item.get('price', item.get('current_price', 0)),
                                        'source': 'market_data_json_data_list'
                                    })
                                    break
                        elif isinstance(market_data['data'], dict) and symbol in market_data['data']:
                            actual_prices.append({
                                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                'price': market_data['data'][symbol].get('price', market_data['data'][symbol].get('current_price', 0)),
                                'source': 'market_data_json_data_dict'
                            })
            except Exception as e:
                self.logger.error(f"Lỗi khi đọc file {json_file}: {str(e)}")
                continue
        
        # 2. Kiểm tra các file CSV của symbol
        csv_pattern = os.path.join(self.data_dir, f"{symbol}_daily_*.csv")
        csv_files = sorted(glob.glob(csv_pattern), reverse=True)
        
        if csv_files:
            try:
                # Lấy file CSV mới nhất
                latest_csv = csv_files[0]
                df = pd.read_csv(latest_csv)
                
                # Kiểm tra các định dạng CSV khác nhau
                date_column = None
                price_column = None
                
                # Tìm column chứa ngày tháng
                for col in df.columns:
                    if col.lower() in ['date', 'timestamp', 'time', 'datetime']:
                        date_column = col
                        break
                
                # Tìm column chứa giá
                for col in df.columns:
                    if col.lower() in ['close', 'price', 'adjusted_close', 'adj close', 'close_price', 'current_price']:
                        price_column = col
                        break
                
                if date_column and price_column:
                    # Chuyển đổi định dạng ngày tháng
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df[date_column])
                    
                    # Lọc theo ngày
                    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                    filtered_df = df.loc[mask]
                    
                    for _, row in filtered_df.iterrows():
                        actual_prices.append({
                            'timestamp': str(row[date_column]),
                            'price': row[price_column],
                            'source': 'csv'
                        })
            except Exception as e:
                self.logger.error(f"Lỗi khi đọc file CSV {csv_files[0]}: {str(e)}")
        
        # Chuyển sang DataFrame và sắp xếp theo ngày
        if len(actual_prices) > 0:
            return actual_prices
        else:
            self.logger.warning(f"Không tìm thấy dữ liệu giá thực tế cho {symbol} trong khoảng thời gian từ {start_date} đến {end_date}")
            return []
    
    def calculate_metrics(self, predictions, actual_data, timeframe='intraday'):
        """Tính toán các metrics đánh giá"""
        self.logger.info(f"Đang tính toán metrics cho {timeframe}")
        
        results = {
            'total_predictions': 0,
            'direction_accuracy': 0,
            'mae': 0,  # Mean Absolute Error
            'mape': 0,  # Mean Absolute Percentage Error
            'confidence_correlation': 0,
            'high_confidence_accuracy': 0
        }
        
        # Kiểm tra nếu không có dữ liệu
        if len(predictions) == 0 or len(actual_data) == 0:
            self.logger.warning(f"Không đủ dữ liệu để tính toán metrics cho {timeframe}")
            return results
        
        prediction_prices = []
        actual_prices = []
        confidence_values = []
        
        # Kết hợp dự đoán với giá thực tế
        for pred in predictions:
            if timeframe not in pred:
                continue
                
            pred_time = datetime.fromisoformat(pred['timestamp'])
            
            # Tìm giá thực tế tương ứng dựa vào timeframe
            target_time = None
            if timeframe == 'intraday':
                # Tìm giá đóng cửa cùng ngày
                target_time = pred_time.replace(hour=16, minute=0, second=0)
            elif timeframe == 'five_day':
                # Tìm giá sau 5 ngày giao dịch
                target_time = pred_time + timedelta(days=7)  # Ước tính 5 ngày giao dịch ~ 7 ngày lịch
            elif timeframe == 'monthly':
                # Tìm giá sau 30 ngày
                target_time = pred_time + timedelta(days=30)
            
            # Tìm giá thực tế gần nhất với target_time
            closest_actual = None
            min_diff = timedelta(days=999)
            
            for actual in actual_data:
                actual_time = datetime.fromisoformat(actual['timestamp'])
                diff = abs(actual_time - target_time)
                
                if diff < min_diff:
                    min_diff = diff
                    closest_actual = actual
            
            # Nếu tìm thấy giá thực tế tương ứng
            if closest_actual:
                pred_price = pred[timeframe]['price']
                actual_price = closest_actual['price']
                
                prediction_prices.append(pred_price)
                actual_prices.append(actual_price)
                
                if 'confidence' in pred[timeframe]:
                    confidence_values.append(pred[timeframe]['confidence'])
        
        # Tính toán metrics
        if len(prediction_prices) > 0:
            results['total_predictions'] = len(prediction_prices)
            
            # Direction accuracy
            correct_direction = 0
            for i in range(1, len(prediction_prices)):
                if (prediction_prices[i] > prediction_prices[i-1] and actual_prices[i] > actual_prices[i-1]) or \
                   (prediction_prices[i] < prediction_prices[i-1] and actual_prices[i] < actual_prices[i-1]):
                    correct_direction += 1
            
            if len(prediction_prices) > 1:
                results['direction_accuracy'] = (correct_direction / (len(prediction_prices) - 1)) * 100
            
            # MAE
            results['mae'] = np.mean(np.abs(np.array(prediction_prices) - np.array(actual_prices)))
            
            # MAPE
            results['mape'] = np.mean(np.abs((np.array(actual_prices) - np.array(prediction_prices)) / np.array(actual_prices))) * 100
            
            # Confidence correlation
            if len(confidence_values) > 0:
                # Tính absolute error
                abs_errors = np.abs(np.array(prediction_prices) - np.array(actual_prices))
                
                # Tính correlation between confidence and inverse of error
                inverse_errors = 1 / (abs_errors + 1e-10)  # Thêm epsilon để tránh chia cho 0
                results['confidence_correlation'] = np.corrcoef(confidence_values, inverse_errors)[0, 1]
                
                # Đánh giá dự đoán có độ tin cậy cao
                high_conf_indices = [i for i, conf in enumerate(confidence_values) if conf > 70]
                if high_conf_indices:
                    high_conf_correct = 0
                    for i in high_conf_indices:
                        if i > 0:  # Đảm bảo có điểm trước đó để so sánh
                            if (prediction_prices[i] > prediction_prices[i-1] and actual_prices[i] > actual_prices[i-1]) or \
                               (prediction_prices[i] < prediction_prices[i-1] and actual_prices[i] < actual_prices[i-1]):
                                high_conf_correct += 1
                    
                    if len(high_conf_indices) > 0:
                        results['high_confidence_accuracy'] = high_conf_correct / len(high_conf_indices) * 100
        
        return results
    
    def generate_report(self, symbol, days=30):
        """Tạo báo cáo đánh giá hiệu suất"""
        self.logger.info(f"Đang tạo báo cáo hiệu suất cho {symbol} trong {days} ngày qua")
        
        # Tải dữ liệu
        predictions = self.load_historical_predictions(symbol, days)
        actual_data = self.load_actual_prices(symbol, days)
        
        if len(predictions) == 0:
            self.logger.warning(f"Không tìm thấy dự đoán cho {symbol}")
            return None
            
        if len(actual_data) == 0:
            self.logger.warning(f"Không tìm thấy dữ liệu giá thực tế cho {symbol}")
            return None
        
        # Tính metrics cho từng timeframe
        report = {
            'symbol': symbol,
            'date_range': f"{datetime.now() - timedelta(days=days)} - {datetime.now()}",
            'intraday': self.calculate_metrics(predictions, actual_data, 'intraday'),
            'five_day': self.calculate_metrics(predictions, actual_data, 'five_day'),
            'monthly': self.calculate_metrics(predictions, actual_data, 'monthly')
        }
        
        # Lưu báo cáo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"{symbol}_performance_{timestamp}.json")
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Tạo báo cáo PDF/HTML
        self._create_visualization(report, symbol, days)
        
        self.logger.info(f"Đã tạo báo cáo hiệu suất cho {symbol} tại {report_path}")
        return report
    
    def _create_visualization(self, report, symbol, days):
        """Tạo trực quan hóa từ báo cáo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = os.path.join(self.output_dir, f"{symbol}_performance_viz_{timestamp}.png")
        
        # Tạo biểu đồ
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Báo cáo hiệu suất - {symbol} ({days} ngày)', fontsize=16)
        
        # 1. So sánh độ chính xác theo timeframe
        timeframes = ['intraday', 'five_day', 'monthly']
        accuracy_values = [report[tf]['direction_accuracy'] for tf in timeframes]
        
        axs[0, 0].bar(timeframes, accuracy_values)
        axs[0, 0].set_title('Độ chính xác theo hướng')
        axs[0, 0].set_ylabel('Độ chính xác (%)')
        axs[0, 0].set_ylim(0, 100)
        
        # 2. So sánh MAPE theo timeframe
        mape_values = [report[tf]['mape'] for tf in timeframes]
        
        axs[0, 1].bar(timeframes, mape_values)
        axs[0, 1].set_title('MAPE theo timeframe')
        axs[0, 1].set_ylabel('MAPE (%)')
        
        # 3. So sánh độ tin cậy với độ chính xác
        confidence_correlation = [report[tf]['confidence_correlation'] for tf in timeframes]
        
        axs[1, 0].bar(timeframes, confidence_correlation)
        axs[1, 0].set_title('Tương quan độ tin cậy-chính xác')
        axs[1, 0].set_ylabel('Hệ số tương quan')
        axs[1, 0].set_ylim(-1, 1)
        
        # 4. So sánh độ chính xác của dự đoán độ tin cậy cao
        high_conf_accuracy = [report[tf].get('high_confidence_accuracy', 0) for tf in timeframes]
        
        axs[1, 1].bar(timeframes, high_conf_accuracy)
        axs[1, 1].set_title('Độ chính xác khi độ tin cậy > 70%')
        axs[1, 1].set_ylabel('Độ chính xác (%)')
        axs[1, 1].set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(viz_path)
        
        self.logger.info(f"Đã tạo biểu đồ hiệu suất tại {viz_path}")
        return viz_path

def main():
    parser = argparse.ArgumentParser(description='BondZiA Performance Reporter')
    parser.add_argument('--symbol', type=str, required=True, help='Mã cổ phiếu cần phân tích')
    parser.add_argument('--days', type=int, default=30, help='Số ngày cần phân tích')
    parser.add_argument('--prediction-dir', type=str, default=prediction_history_dir, 
                        help='Thư mục chứa lịch sử dự đoán')
    parser.add_argument('--data-dir', type=str, default=data_dir, 
                        help='Thư mục chứa dữ liệu thị trường')
    parser.add_argument('--output-dir', type=str, default=output_dir, 
                        help='Thư mục lưu báo cáo')
    
    args = parser.parse_args()
    
    reporter = PerformanceReporter(
        prediction_history_dir=args.prediction_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    reporter.generate_report(args.symbol, args.days)

if __name__ == "__main__":
    main()