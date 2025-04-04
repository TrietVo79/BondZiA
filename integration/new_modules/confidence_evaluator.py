import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import traceback

# Thêm thư mục gốc vào PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger_config import logger

class ConfidenceEvaluator:
    """
    Lớp đánh giá và quản lý độ tin cậy của các dự đoán
    """
    
    def __init__(self, config_path="../config/system_config.json"):
        """
        Khởi tạo ConfidenceEvaluator
        
        Args:
            config_path (str): Đường dẫn đến file cấu hình
        """
        # Đọc cấu hình
        self.config_path = config_path
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Đường dẫn đến thư mục lưu trữ lịch sử dự đoán
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.history_dir = os.path.join(root_dir, "prediction_history")
        os.makedirs(self.history_dir, exist_ok=True)
        
        # Đường dẫn đến thư mục lưu trữ thống kê
        self.stats_dir = os.path.join(root_dir, "prediction_stats")
        os.makedirs(self.stats_dir, exist_ok=True)
        
        # Tải thống kê nếu có
        self.stats = self._load_stats()
        
        logger.info("Khởi tạo ConfidenceEvaluator")
    
    def _load_stats(self):
        """
        Tải thống kê dự đoán từ file
        
        Returns:
            dict: Thống kê dự đoán hoặc dict rỗng nếu không có file
        """
        try:
            stats_file = os.path.join(self.stats_dir, "prediction_stats.json")
            
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    return json.load(f)
            
            return {
                'symbols': {},
                'timeframes': {
                    'intraday': {
                        'total_predictions': 0,
                        'correct_directions': 0,
                        'avg_error': 0,
                        'confidence_accuracy': []
                    },
                    'five_day': {
                        'total_predictions': 0,
                        'correct_directions': 0,
                        'avg_error': 0,
                        'confidence_accuracy': []
                    },
                    'monthly': {
                        'total_predictions': 0,
                        'correct_directions': 0,
                        'avg_error': 0,
                        'confidence_accuracy': []
                    }
                },
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"Lỗi khi tải thống kê dự đoán: {str(e)}")
            return {
                'symbols': {},
                'timeframes': {
                    'intraday': {
                        'total_predictions': 0,
                        'correct_directions': 0,
                        'avg_error': 0,
                        'confidence_accuracy': []
                    },
                    'five_day': {
                        'total_predictions': 0,
                        'correct_directions': 0,
                        'avg_error': 0,
                        'confidence_accuracy': []
                    },
                    'monthly': {
                        'total_predictions': 0,
                        'correct_directions': 0,
                        'avg_error': 0,
                        'confidence_accuracy': []
                    }
                },
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
    def _save_stats(self):
        """
        Lưu thống kê dự đoán vào file
        
        Returns:
            bool: True nếu lưu thành công, False nếu không
        """
        try:
            stats_file = os.path.join(self.stats_dir, "prediction_stats.json")
            
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=4)
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu thống kê dự đoán: {str(e)}")
            return False
    
    def log_prediction(self, symbol, timeframe, prediction):
        """
        Ghi nhận dự đoán mới
        
        Args:
            symbol (str): Mã cổ phiếu
            timeframe (str): Khung thời gian
            prediction (dict): Dự đoán
            
        Returns:
            bool: True nếu ghi nhận thành công, False nếu không
        """
        try:
            # Tạo thư mục lưu trữ lịch sử dự đoán cho symbol nếu chưa có
            symbol_dir = os.path.join(self.history_dir, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Đường dẫn đến file lịch sử dự đoán
            history_file = os.path.join(symbol_dir, f"{timeframe}_predictions.json")
            
            # Lấy lịch sử dự đoán nếu có
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            # Chuẩn bị dự đoán mới
            new_prediction = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'price': prediction.get('price'),
                'current_price': prediction.get('current_price'),
                'percent_change': prediction.get('percent_change'),
                'confidence': prediction.get('confidence'),
                'reason': prediction.get('reason'),
                'actual_price': None,  # Sẽ được cập nhật sau
                'actual_change': None,  # Sẽ được cập nhật sau
                'error': None,          # Sẽ được cập nhật sau
                'direction_correct': None, # Sẽ được cập nhật sau
                'verified': False       # Đánh dấu chưa được xác minh
            }
            
            # Thêm vào lịch sử
            history.append(new_prediction)
            
            # Lưu lịch sử
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=4)
            
            logger.info(f"Đã ghi nhận dự đoán mới cho {symbol} - {timeframe}")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi ghi nhận dự đoán: {str(e)}")
            return False

    def verify_predictions(self, symbols=None, data_fetcher=None):
        """
        Xác minh các dự đoán trước đó bằng cách so sánh với giá thực tế
        
        Args:
            symbols (list, optional): Danh sách các mã cổ phiếu. Nếu None, xác minh tất cả.
            data_fetcher (object, optional): Đối tượng để lấy dữ liệu thực tế. Nếu None, tạo mới.
            
        Returns:
            dict: Kết quả xác minh
        """
        try:
            if symbols is None:
                # Lấy tất cả symbols có trong thư mục history
                symbols = [d for d in os.listdir(self.history_dir) if os.path.isdir(os.path.join(self.history_dir, d))]
            
            if data_fetcher is None:
                # Tạo data_fetcher mới nếu không được cung cấp
                from utils.data_fetcher import EnhancedDataFetcher
                data_fetcher = EnhancedDataFetcher(config_path=self.config_path)
            
            results = {
                'total_verified': 0,
                'correct_directions': 0,
                'symbols': {}
            }
            
            for symbol in symbols:
                symbol_dir = os.path.join(self.history_dir, symbol)
                
                if not os.path.isdir(symbol_dir):
                    logger.warning(f"Không tìm thấy thư mục lịch sử cho {symbol}")
                    continue
                
                results['symbols'][symbol] = {
                    'timeframes': {},
                    'total_verified': 0,
                    'correct_directions': 0
                }
                
                # Lấy dữ liệu thực tế
                current_data = data_fetcher.get_enhanced_stock_data(symbol)
                
                # Xác minh cho từng timeframe
                for timeframe in ['intraday', 'five_day', 'monthly']:
                    history_file = os.path.join(symbol_dir, f"{timeframe}_predictions.json")
                    
                    if not os.path.exists(history_file):
                        continue
                    
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                    
                    # Khởi tạo thống kê cho timeframe này
                    results['symbols'][symbol]['timeframes'][timeframe] = {
                        'verified': 0,
                        'correct_directions': 0,
                        'avg_error': 0
                    }
                    
                    # Số dự đoán đã được xác minh cho timeframe này
                    verified_count = 0
                    correct_directions = 0
                    total_error = 0
                    
                    # Cập nhật từng dự đoán
                    for i, prediction in enumerate(history):
                        # Bỏ qua các dự đoán đã được xác minh
                        if prediction.get('verified', False):
                            verified_count += 1
                            if prediction.get('direction_correct', False):
                                correct_directions += 1
                            if prediction.get('error') is not None:
                                total_error += abs(prediction.get('error'))
                            continue
                        
                        # Kiểm tra xem dự đoán đã đủ thời gian để xác minh chưa
                        prediction_time = datetime.strptime(prediction['timestamp'], '%Y-%m-%d %H:%M:%S')
                        
                        # Xác định thời gian cần để xác minh
                        if timeframe == 'intraday':
                            verification_time = timedelta(hours=6)  # 6 giờ cho intraday
                        elif timeframe == 'five_day':
                            verification_time = timedelta(days=5)   # 5 ngày
                        else:  # monthly
                            verification_time = timedelta(days=30)  # 30 ngày
                        
                        # Kiểm tra xem đã đủ thời gian chưa
                        if datetime.now() - prediction_time < verification_time:
                            continue
                        
                        # Lấy giá thực tế
                        if timeframe == 'intraday':
                            if 'intraday_data' in current_data and not current_data['intraday_data'].empty:
                                actual_price = current_data['intraday_data']['close'].iloc[-1]
                            else:
                                continue
                        else:
                            if 'daily_data' in current_data and not current_data['daily_data'].empty:
                                actual_price = current_data['daily_data']['close'].iloc[-1]
                            else:
                                continue
                        
                        # Cập nhật dự đoán
                        predicted_change = prediction['percent_change']
                        actual_change = ((actual_price - prediction['current_price']) / prediction['current_price']) * 100
                        error = predicted_change - actual_change
                        
                        # Kiểm tra hướng
                        direction_correct = (predicted_change > 0 and actual_change > 0) or (predicted_change < 0 and actual_change < 0)
                        
                        # Cập nhật dự đoán
                        history[i]['actual_price'] = actual_price
                        history[i]['actual_change'] = actual_change
                        history[i]['error'] = error
                        history[i]['direction_correct'] = direction_correct
                        history[i]['verified'] = True
                        
                        # Cập nhật thống kê
                        verified_count += 1
                        total_error += abs(error)
                        
                        if direction_correct:
                            correct_directions += 1
                    
                    # Lưu lịch sử đã cập nhật
                    with open(history_file, 'w') as f:
                        json.dump(history, f, indent=4)
                    
                    # Cập nhật thống kê
                    results['symbols'][symbol]['timeframes'][timeframe]['verified'] = verified_count
                    
                    if verified_count > 0:
                        results['symbols'][symbol]['timeframes'][timeframe]['correct_directions'] = correct_directions
                        results['symbols'][symbol]['timeframes'][timeframe]['avg_error'] = total_error / verified_count
                    
                    # Cập nhật tổng số
                    results['symbols'][symbol]['total_verified'] += verified_count
                    results['symbols'][symbol]['correct_directions'] += correct_directions
                    results['total_verified'] += verified_count
                    results['correct_directions'] += correct_directions
            
            # Cập nhật thống kê tổng thể
            self._update_stats(results)
            
            return results
        except Exception as e:
            logger.error(f"Lỗi khi xác minh dự đoán: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}

    def _update_stats(self, verification_results):
        """
        Cập nhật thống kê dự đoán dựa trên kết quả xác minh
        
        Args:
            verification_results (dict): Kết quả xác minh
            
        Returns:
            bool: True nếu cập nhật thành công, False nếu không
        """
        try:
            # Cập nhật thống kê cho từng symbol
            for symbol, symbol_results in verification_results['symbols'].items():
                if symbol not in self.stats['symbols']:
                    self.stats['symbols'][symbol] = {
                        'timeframes': {
                            'intraday': {
                                'total_predictions': 0,
                                'correct_directions': 0,
                                'avg_error': 0
                            },
                            'five_day': {
                                'total_predictions': 0,
                                'correct_directions': 0,
                                'avg_error': 0
                            },
                            'monthly': {
                                'total_predictions': 0,
                                'correct_directions': 0,
                                'avg_error': 0
                            }
                        },
                        'total_predictions': 0,
                        'correct_directions': 0,
                        'avg_error': 0
                    }
                
                # Cập nhật thống kê cho từng timeframe
                for timeframe, timeframe_results in symbol_results.get('timeframes', {}).items():
                    verified = timeframe_results.get('verified', 0)
                    correct_directions = timeframe_results.get('correct_directions', 0)
                    avg_error = timeframe_results.get('avg_error', 0)
                    
                    if verified > 0:
                        # Cập nhật thống kê cho symbol và timeframe
                        self.stats['symbols'][symbol]['timeframes'][timeframe]['total_predictions'] += verified
                        self.stats['symbols'][symbol]['timeframes'][timeframe]['correct_directions'] += correct_directions
                        
                        # Cập nhật avg_error (trung bình có trọng số)
                        old_total = self.stats['symbols'][symbol]['timeframes'][timeframe]['total_predictions'] - verified
                        old_avg = self.stats['symbols'][symbol]['timeframes'][timeframe]['avg_error']
                        new_avg = ((old_total * old_avg) + (verified * avg_error)) / (old_total + verified)
                        self.stats['symbols'][symbol]['timeframes'][timeframe]['avg_error'] = new_avg
                        
                        # Cập nhật thống kê tổng cho symbol
                        self.stats['symbols'][symbol]['total_predictions'] += verified
                        self.stats['symbols'][symbol]['correct_directions'] += correct_directions
                        
                        # Cập nhật avg_error tổng cho symbol
                        old_total = self.stats['symbols'][symbol]['total_predictions'] - verified
                        old_avg = self.stats['symbols'][symbol]['avg_error']
                        new_avg = ((old_total * old_avg) + (verified * avg_error)) / (old_total + verified)
                        self.stats['symbols'][symbol]['avg_error'] = new_avg
                        
                        # Cập nhật thống kê tổng cho timeframe
                        self.stats['timeframes'][timeframe]['total_predictions'] += verified
                        self.stats['timeframes'][timeframe]['correct_directions'] += correct_directions
                        
                        # Cập nhật avg_error tổng cho timeframe
                        old_total = self.stats['timeframes'][timeframe]['total_predictions'] - verified
                        old_avg = self.stats['timeframes'][timeframe]['avg_error']
                        new_avg = ((old_total * old_avg) + (verified * avg_error)) / (old_total + verified)
                        self.stats['timeframes'][timeframe]['avg_error'] = new_avg
            
            # Cập nhật thời gian cập nhật
            self.stats['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Lưu thống kê
            self._save_stats()
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật thống kê dự đoán: {str(e)}")
            return False

    def evaluate_market_conditions(self, data_fetcher=None):
        """
        Đánh giá điều kiện thị trường hiện tại để điều chỉnh độ tin cậy
        
        Args:
            data_fetcher (object, optional): Đối tượng để lấy dữ liệu thị trường. Nếu None, tạo mới.
            
        Returns:
            dict: Đánh giá điều kiện thị trường
        """
        try:
            # Nếu data_fetcher không được cung cấp, tạo mới
            if data_fetcher is None:
                from utils.data_fetcher import EnhancedDataFetcher
                data_fetcher = EnhancedDataFetcher(config_path=self.config_path)
            
            # Lấy chỉ số kinh tế
            economic_indicators = data_fetcher.get_economic_indicators()
            
            # Đánh giá biến động thị trường
            market_volatility = self._assess_market_volatility(economic_indicators)
            
            # Đánh giá tâm lý thị trường
            market_sentiment = self._assess_market_sentiment(economic_indicators)
            
            # Đánh giá các điều kiện thị trường khác
            market_conditions = {
                'volatility': market_volatility,
                'sentiment': market_sentiment,
                'confidence_modifier': self._calculate_confidence_modifier(market_volatility, market_sentiment),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return market_conditions
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá điều kiện thị trường: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'volatility': 'medium',
                'sentiment': 'neutral',
                'confidence_modifier': 0,
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def _assess_market_volatility(self, economic_indicators):
        """
        Đánh giá mức độ biến động của thị trường
        
        Args:
            economic_indicators (dict): Các chỉ số kinh tế
            
        Returns:
            str: Mức độ biến động ('low', 'medium', 'high', 'extreme')
        """
        try:
            # Lấy chỉ số VIX nếu có
            vix_value = None
            if 'CBOE Volatility Index' in economic_indicators:
                vix_value = economic_indicators['CBOE Volatility Index'].get('value')
            
            # Nếu có VIX, đánh giá dựa trên nó
            if vix_value is not None:
                if vix_value < 15:
                    return 'low'
                elif vix_value < 25:
                    return 'medium'
                elif vix_value < 35:
                    return 'high'
                else:
                    return 'extreme'
            
            # Nếu không có VIX, đánh giá dựa trên biến động của S&P 500
            if 'S&P 500' in economic_indicators:
                sp500_change = abs(economic_indicators['S&P 500'].get('change_percent', 0))
                
                if sp500_change < 0.5:
                    return 'low'
                elif sp500_change < 1.5:
                    return 'medium'
                elif sp500_change < 3:
                    return 'high'
                else:
                    return 'extreme'
            
            # Nếu không có dữ liệu, trả về giá trị trung bình
            return 'medium'
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá biến động thị trường: {str(e)}")
            return 'medium'

    def _assess_market_sentiment(self, economic_indicators):
        """
        Đánh giá tâm lý thị trường
        
        Args:
            economic_indicators (dict): Các chỉ số kinh tế
            
        Returns:
            str: Tâm lý thị trường ('bearish', 'slightly_bearish', 'neutral', 'slightly_bullish', 'bullish')
        """
        try:
            # Tính điểm tâm lý dựa trên các chỉ số
            sentiment_score = 0
            indicators_count = 0
            
            # Các chỉ số chính để đánh giá tâm lý
            major_indices = ['S&P 500', 'Dow Jones Industrial Average', 'NASDAQ Composite']
            
            for index_name in major_indices:
                if index_name in economic_indicators:
                    change_percent = economic_indicators[index_name].get('change_percent', 0)
                    
                    # Tính điểm dựa trên phần trăm thay đổi
                    if change_percent < -2:
                        sentiment_score -= 2
                    elif change_percent < -0.5:
                        sentiment_score -= 1
                    elif change_percent > 2:
                        sentiment_score += 2
                    elif change_percent > 0.5:
                        sentiment_score += 1
                    
                    indicators_count += 1
            
            # Nếu không có đủ dữ liệu, trả về giá trị trung tính
            if indicators_count == 0:
                return 'neutral'
            
            # Tính trung bình
            avg_score = sentiment_score / indicators_count
            
            # Phân loại tâm lý
            if avg_score < -1.5:
                return 'bearish'
            elif avg_score < -0.5:
                return 'slightly_bearish'
            elif avg_score > 1.5:
                return 'bullish'
            elif avg_score > 0.5:
                return 'slightly_bullish'
            else:
                return 'neutral'
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá tâm lý thị trường: {str(e)}")
            return 'neutral'
    
    def _calculate_confidence_modifier(self, volatility, sentiment):
        """
        Tính toán hệ số điều chỉnh độ tin cậy dựa trên điều kiện thị trường
        
        Args:
            volatility (str): Mức độ biến động
            sentiment (str): Tâm lý thị trường
            
        Returns:
            float: Hệ số điều chỉnh (từ -30 đến +10)
        """
        try:
            # Điều chỉnh dựa trên biến động
            volatility_modifier = {
                'low': 10,       # Biến động thấp -> tăng độ tin cậy
                'medium': 0,     # Biến động trung bình -> không thay đổi
                'high': -15,     # Biến động cao -> giảm độ tin cậy
                'extreme': -30   # Biến động cực cao -> giảm mạnh độ tin cậy
            }.get(volatility, 0)
            
            # Điều chỉnh dựa trên tâm lý
            sentiment_modifier = {
                'bearish': -5,           # Thị trường giảm mạnh -> giảm độ tin cậy
                'slightly_bearish': -2,  # Thị trường giảm nhẹ -> giảm nhẹ độ tin cậy
                'neutral': 0,            # Thị trường trung tính -> không thay đổi
                'slightly_bullish': 2,   # Thị trường tăng nhẹ -> tăng nhẹ độ tin cậy
                'bullish': 5             # Thị trường tăng mạnh -> tăng độ tin cậy
            }.get(sentiment, 0)
            
            # Tổng hợp điều chỉnh
            return volatility_modifier + sentiment_modifier
        except Exception as e:
            logger.error(f"Lỗi khi tính toán hệ số điều chỉnh độ tin cậy: {str(e)}")
            return 0
    
    def adjust_prediction_confidence(self, symbol, timeframe, prediction, market_conditions=None):
        """
        Điều chỉnh độ tin cậy của dự đoán dựa trên điều kiện thị trường và lịch sử dự đoán
        
        Args:
            symbol (str): Mã cổ phiếu
            timeframe (str): Khung thời gian
            prediction (dict): Dự đoán
            market_conditions (dict, optional): Điều kiện thị trường. Nếu None, sẽ được đánh giá.
            
        Returns:
            dict: Dự đoán đã được điều chỉnh độ tin cậy
        """
        try:
            # Nếu dự đoán không có độ tin cậy, không cần điều chỉnh
            if 'confidence' not in prediction:
                return prediction
            
            # Lấy điều kiện thị trường nếu chưa được cung cấp
            if market_conditions is None:
                market_conditions = self.evaluate_market_conditions()
            
            # Lấy hệ số điều chỉnh từ điều kiện thị trường
            market_modifier = market_conditions.get('confidence_modifier', 0)
            
            # Điều chỉnh dựa trên lịch sử dự đoán của symbol và timeframe
            symbol_modifier = self._get_symbol_confidence_modifier(symbol, timeframe)
            
            # Điều chỉnh dựa trên khung thời gian
            timeframe_modifier = {
                'intraday': -5,    # Intraday khó dự đoán hơn -> giảm độ tin cậy
                'five_day': 0,     # Five_day là cơ sở -> không thay đổi
                'monthly': -10     # Monthly chịu ảnh hưởng bởi nhiều yếu tố không lường trước -> giảm độ tin cậy
            }.get(timeframe, 0)
            
            # Tổng hợp các điều chỉnh
            total_modifier = market_modifier + symbol_modifier + timeframe_modifier
            
            # Điều chỉnh độ tin cậy
            adjusted_confidence = prediction['confidence'] + total_modifier
            
            # Giới hạn trong khoảng 0-100
            adjusted_confidence = max(0.1, min(99.9, adjusted_confidence))
            
            # Làm tròn
            adjusted_confidence = round(adjusted_confidence, 1)
            
            # Cập nhật dự đoán
            adjusted_prediction = prediction.copy()
            adjusted_prediction['confidence'] = adjusted_confidence
            adjusted_prediction['confidence_adjustments'] = {
                'market': market_modifier,
                'symbol_history': symbol_modifier,
                'timeframe': timeframe_modifier,
                'total': total_modifier
            }
            
            return adjusted_prediction
        except Exception as e:
            logger.error(f"Lỗi khi điều chỉnh độ tin cậy: {str(e)}")
            return prediction
    
    def _get_symbol_confidence_modifier(self, symbol, timeframe):
        """
        Lấy hệ số điều chỉnh độ tin cậy dựa trên lịch sử dự đoán của symbol và timeframe
        
        Args:
            symbol (str): Mã cổ phiếu
            timeframe (str): Khung thời gian
            
        Returns:
            float: Hệ số điều chỉnh (từ -20 đến +20)
        """
        try:
            # Nếu không có thống kê cho symbol này, trả về 0
            if symbol not in self.stats['symbols']:
                return 0
            
            # Lấy thống kê của symbol và timeframe
            symbol_stats = self.stats['symbols'][symbol]
            timeframe_stats = symbol_stats['timeframes'].get(timeframe, {})
            
            # Nếu không có đủ dữ liệu, trả về 0
            total_predictions = timeframe_stats.get('total_predictions', 0)
            if total_predictions < 5:
                return 0
            
            # Tính direction accuracy
            correct_directions = timeframe_stats.get('correct_directions', 0)
            direction_accuracy = (correct_directions / total_predictions) * 100 if total_predictions > 0 else 50
            
            # Điều chỉnh dựa trên direction accuracy
            if direction_accuracy > 70:
                return 20  # Nếu accuracy > 70%, tăng mạnh độ tin cậy
            elif direction_accuracy > 60:
                return 10  # Nếu accuracy > 60%, tăng độ tin cậy
            elif direction_accuracy < 40:
                return -20  # Nếu accuracy < 40%, giảm mạnh độ tin cậy
            elif direction_accuracy < 50:
                return -10  # Nếu accuracy < 50%, giảm độ tin cậy
            else:
                return 0  # Nếu accuracy trong khoảng 50-60%, không thay đổi
        except Exception as e:
            logger.error(f"Lỗi khi lấy hệ số điều chỉnh từ lịch sử: {str(e)}")
            return 0
    
    def get_confidence_stats(self):
        """
        Lấy thống kê về độ tin cậy của các dự đoán
        
        Returns:
            dict: Thống kê về độ tin cậy
        """
        try:
            # Cập nhật thống kê nếu cần
            verification_results = self.verify_predictions()
            
            # Tính direction accuracy cho mỗi timeframe
            timeframe_accuracy = {}
            
            for timeframe, stats in self.stats['timeframes'].items():
                total = stats.get('total_predictions', 0)
                correct = stats.get('correct_directions', 0)
                
                if total > 0:
                    accuracy = (correct / total) * 100
                    timeframe_accuracy[timeframe] = round(accuracy, 2)
                else:
                    timeframe_accuracy[timeframe] = 0
            
            # Lấy top 3 symbols có độ chính xác cao nhất
            top_symbols = []
            
            for symbol, stats in self.stats['symbols'].items():
                total = stats.get('total_predictions', 0)
                correct = stats.get('correct_directions', 0)
                
                if total >= 10:  # Chỉ xét các symbol có ít nhất 10 dự đoán
                    accuracy = (correct / total) * 100
                    top_symbols.append({
                        'symbol': symbol,
                        'accuracy': accuracy,
                        'total_predictions': total
                    })
            
            # Sắp xếp theo độ chính xác giảm dần
            top_symbols.sort(key=lambda x: x['accuracy'], reverse=True)
            
            # Lấy top 3
            top_symbols = top_symbols[:3]
            
            # Kết quả
            result = {
                'timeframe_accuracy': timeframe_accuracy,
                'top_symbols': top_symbols,
                'last_update': self.stats.get('last_updated'),
                'total_verified_predictions': sum(stats.get('total_predictions', 0) for stats in self.stats['timeframes'].values())
            }
            
            return result
        except Exception as e:
            logger.error(f"Lỗi khi lấy thống kê độ tin cậy: {str(e)}")
            return {'error': str(e)}                    