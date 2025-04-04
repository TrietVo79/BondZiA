import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import traceback

# Thêm thư mục gốc vào PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from utils.logger_config import logger

# Import lớp EnsemblePredictor và ConfidenceEvaluator nếu tồn tại
try:
    from models.ensemble_predictor import EnsemblePredictor
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    logger.warning("EnsemblePredictor không khả dụng, sử dụng mô hình đơn")

try:
    from utils.confidence_evaluator import ConfidenceEvaluator
    CONFIDENCE_EVALUATOR_AVAILABLE = True
except ImportError:
    CONFIDENCE_EVALUATOR_AVAILABLE = False
    logger.warning("ConfidenceEvaluator không khả dụng, sử dụng phương pháp tính độ tin cậy đơn giản")

# Lớp dự đoán cơ sở
class BasePredictor:
    """Lớp cơ sở cho các bộ dự đoán giá"""
    
    def __init__(self, symbol, timeframe, config_path="../config/system_config.json"):
        """
        Khởi tạo BasePredictor
        
        Args:
            symbol (str): Mã cổ phiếu
            timeframe (str): Khung thời gian dự đoán ('intraday', 'five_day', 'monthly')
            config_path (str): Đường dẫn đến file cấu hình
        """
        # Lưu thông tin cơ bản
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Đọc cấu hình
        self.config_path = config_path
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Lấy cấu hình cho khung thời gian
        self.prediction_config = self.config['prediction'][timeframe]
        self.lookback_window = self.prediction_config['lookback_window']
        
        # Đường dẫn lưu trữ mô hình
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               f"models/{timeframe}")
        os.makedirs(models_dir, exist_ok=True)
        
        self.model_path = os.path.join(models_dir, f"{symbol}_{timeframe}_model.h5")
        
        # Đường dẫn lưu trữ scaler
        self.scalers_dir = os.path.join(models_dir, "scalers")
        os.makedirs(self.scalers_dir, exist_ok=True)
        
        self.feature_scaler_path = os.path.join(self.scalers_dir, f"{symbol}_{timeframe}_feature_scaler.pkl")
        self.price_scaler_path = os.path.join(self.scalers_dir, f"{symbol}_{timeframe}_price_scaler.pkl")
        
        # Khởi tạo mô hình
        self.model = None
        self.feature_scaler = None
        self.price_scaler = None

        # Khởi tạo EnsemblePredictor nếu có
        self.ensemble_predictor = None
        if ENSEMBLE_AVAILABLE:
            try:
                self.ensemble_predictor = EnsemblePredictor(symbol, timeframe, config_path=config_path)
            except Exception as e:
                logger.error(f"Lỗi khi khởi tạo EnsemblePredictor: {str(e)}")
                logger.error(traceback.format_exc())

        # Khởi tạo ConfidenceEvaluator nếu có
        self.confidence_evaluator = None
        if CONFIDENCE_EVALUATOR_AVAILABLE:
            try:
                self.confidence_evaluator = ConfidenceEvaluator(config_path=config_path)
            except Exception as e:
                logger.error(f"Lỗi khi khởi tạo ConfidenceEvaluator: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Các tính năng mặc định
        self.default_features = [
            'open', 'high', 'low', 'close', 'volume', 
            'rsi_14', 'macd', 'sma_20', 'ema_9'
        ]
        
        logger.info(f"Khởi tạo BasePredictor cho {symbol} - {timeframe}")
    
    def check_model_exists(self):
        """
        Kiểm tra xem mô hình đã tồn tại chưa
        
        Returns:
            bool: True nếu mô hình đã tồn tại, False nếu chưa
        """
        # Kiểm tra mô hình ensemble trước (nếu có)
        if self.ensemble_predictor and self.ensemble_predictor.check_model_exists():
            return True
        
        # Kiểm tra mô hình đơn
        return os.path.exists(self.model_path)
    
    def _prepare_data(self, data, features=None):
        """
        Chuẩn bị dữ liệu cho mô hình
        
        Args:
            data (DataFrame): Dữ liệu đầu vào
            features (list, optional): Danh sách các tính năng sử dụng
            
        Returns:
            tuple: X, y, feature_scaler, price_scaler
        """
        try:
            if features is None:
                features = self.default_features
            
            # Đảm bảo các tính năng tồn tại trong dữ liệu
            available_features = [f for f in features if f in data.columns]
            
            # Nếu thiếu tính năng, log cảnh báo
            if len(available_features) < len(features):
                missing_features = set(features) - set(available_features)
                logger.warning(f"Thiếu các tính năng: {missing_features}. Sẽ sử dụng các tính năng có sẵn.")
            
            # Nếu không có đủ tính năng, sử dụng giá đóng cửa
            if len(available_features) < 3:
                logger.warning(f"Không đủ tính năng. Sẽ chỉ sử dụng giá đóng cửa.")
                available_features = ['close']
            
            # Đảm bảo có giá đóng cửa
            if 'close' not in available_features:
                logger.error(f"Không có dữ liệu giá đóng cửa, không thể chuẩn bị dữ liệu")
                return None, None, None, None
            
            # Lọc dữ liệu
            data = data[available_features].copy()
            
            # Xử lý missing values
            data.fillna(method='ffill', inplace=True)
            data.fillna(method='bfill', inplace=True)
            
            # Nếu vẫn còn missing values, drop hàng
            if data.isnull().any().any():
                logger.warning("Vẫn còn missing values sau khi xử lý")
                data.dropna(inplace=True)
            
            # Chuẩn hóa dữ liệu
            if os.path.exists(self.feature_scaler_path) and os.path.exists(self.price_scaler_path):
                # Tải scalers đã lưu
                feature_scaler = joblib.load(self.feature_scaler_path)
                price_scaler = joblib.load(self.price_scaler_path)
            else:
                # Tạo scalers mới
                feature_scaler = StandardScaler()
                feature_scaler.fit(data[available_features].values)
                
                price_scaler = MinMaxScaler(feature_range=(0, 1))
                price_scaler.fit(data[['close']].values)
            
            # Chuẩn hóa dữ liệu
            X_values = feature_scaler.transform(data[available_features].values)
            y_values = price_scaler.transform(data[['close']].values)
            
            # Tạo chuỗi thời gian
            X = []
            y = []
            
            for i in range(self.lookback_window, len(data)):
                X.append(X_values[i-self.lookback_window:i])
                y.append(y_values[i])
            
            X = np.array(X)
            y = np.array(y)
            
            return X, y, feature_scaler, price_scaler
        
        except Exception as e:
            logger.error(f"Lỗi khi chuẩn bị dữ liệu: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None, None, None
    
    def _build_model(self, input_shape):
        """
        Xây dựng mô hình dự đoán
        
        Args:
            input_shape (tuple): Kích thước đầu vào
            
        Returns:
            Model: Mô hình Keras
        """
        # Xây dựng mô hình cơ bản với cải tiến
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        # Biên dịch mô hình
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def train(self, data, epochs=100, batch_size=32, validation_split=0.2):
        """
        Huấn luyện mô hình
        
        Args:
            data (DataFrame): Dữ liệu huấn luyện
            epochs (int): Số epoch
            batch_size (int): Kích thước batch
            validation_split (float): Tỷ lệ dữ liệu validation
            
        Returns:
            History: Lịch sử huấn luyện
        """
        try:
            # Ưu tiên sử dụng mô hình ensemble nếu có
            if self.ensemble_predictor and self.config.get('use_ensemble', True):
                logger.info(f"Huấn luyện mô hình ensemble cho {self.symbol} - {self.timeframe}")
                return self.ensemble_predictor.train(data)
            
            logger.info(f"Huấn luyện mô hình đơn cho {self.symbol} - {self.timeframe}")
            
            # Chuẩn bị dữ liệu
            X, y, feature_scaler, price_scaler = self._prepare_data(data)
            
            if X is None or y is None:
                logger.error("Không thể chuẩn bị dữ liệu huấn luyện")
                return None
            
            # Xây dựng mô hình
            self.model = self._build_model((X.shape[1], X.shape[2]))
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                ModelCheckpoint(self.model_path, save_best_only=True)
            ]
            
            # Huấn luyện mô hình
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # Lưu scalers
            joblib.dump(feature_scaler, self.feature_scaler_path)
            joblib.dump(price_scaler, self.price_scaler_path)
            
            # Lưu tham chiếu
            self.feature_scaler = feature_scaler
            self.price_scaler = price_scaler
            
            logger.info(f"Đã huấn luyện xong mô hình cho {self.symbol} - {self.timeframe}")
            
            return history
        
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def predict(self, data):
        """
        Dự đoán giá cổ phiếu
        
        Args:
            data (DataFrame): Dữ liệu đầu vào
            
        Returns:
            dict: Kết quả dự đoán
        """
        try:
            # Dùng mô hình ensemble nếu có và đã được huấn luyện
            if self.ensemble_predictor and self.ensemble_predictor.check_model_exists() and self.config.get('use_ensemble', True):
                logger.info(f"Dự đoán bằng mô hình ensemble cho {self.symbol} - {self.timeframe}")
                prediction = self.ensemble_predictor.predict(data)
                
                # Điều chỉnh độ tin cậy nếu có ConfidenceEvaluator
                if self.confidence_evaluator and 'confidence' in prediction:
                    prediction = self.confidence_evaluator.adjust_prediction_confidence(
                        self.symbol, self.timeframe, prediction
                    )
                    
                    # Ghi nhận dự đoán để theo dõi
                    self.confidence_evaluator.log_prediction(self.symbol, self.timeframe, prediction)
                
                return prediction
            
            logger.info(f"Dự đoán bằng mô hình đơn cho {self.symbol} - {self.timeframe}")
            
            # Chuẩn bị dữ liệu
            X, _, feature_scaler, price_scaler = self._prepare_data(data)
            
            if X is None:
                logger.error("Không thể chuẩn bị dữ liệu dự đoán")
                return None
            
            # Tải mô hình nếu chưa có
            if self.model is None:
                if os.path.exists(self.model_path):
                    self.model = load_model(self.model_path)
                else:
                    logger.error(f"Không tìm thấy mô hình tại {self.model_path}")
                    return None
            
            # Tải scalers nếu chưa có
            if self.feature_scaler is None or self.price_scaler is None:
                if os.path.exists(self.feature_scaler_path) and os.path.exists(self.price_scaler_path):
                    self.feature_scaler = joblib.load(self.feature_scaler_path)
                    self.price_scaler = joblib.load(self.price_scaler_path)
                else:
                    logger.error("Không tìm thấy scalers")
                    return None
            
            # Dự đoán
            y_pred = self.model.predict(X)
            
            # Chuyển về giá thực
            price_pred = self.price_scaler.inverse_transform(y_pred)
            
            # Lấy giá hiện tại
            current_price = data['close'].iloc[-1]
            
            # Tính toán các giá trị dự đoán
            if self.timeframe == 'intraday':
                # Intraday: dự đoán giá cuối ngày
                prediction_price = float(price_pred[-1][0])
                
                # Tính % thay đổi
                percent_change = ((prediction_price - current_price) / current_price) * 100
                
                # Xây dựng kết quả dự đoán
                result = {
                    'price': prediction_price,
                    'current_price': float(current_price),
                    'percent_change': float(percent_change)
                }
                
            elif self.timeframe == 'five_day':
                # Five_day: dự đoán giá sau 5 ngày
                prediction_price = float(price_pred[-1][0])
                
                # Tính % thay đổi
                percent_change = ((prediction_price - current_price) / current_price) * 100
                
                # Xây dựng kết quả dự đoán
                result = {
                    'price': prediction_price,
                    'current_price': float(current_price),
                    'percent_change': float(percent_change)
                }
                
            elif self.timeframe == 'monthly':
                # Monthly: dự đoán giá sau 1 tháng
                prediction_price = float(price_pred[-1][0])
                
                # Tính % thay đổi
                percent_change = ((prediction_price - current_price) / current_price) * 100
                
                # Xây dựng kết quả dự đoán
                result = {
                    'price': prediction_price,
                    'current_price': float(current_price),
                    'percent_change': float(percent_change)
                }
            
            # Tính độ tin cậy
            confidence = self._calculate_confidence(data, price_pred, current_price)
            result['confidence'] = confidence
            
            # Thêm lý do
            result['reason'] = self._generate_prediction_reason(prediction_price, current_price, data)
            
            # Điều chỉnh độ tin cậy nếu có ConfidenceEvaluator
            if self.confidence_evaluator:
                result = self.confidence_evaluator.adjust_prediction_confidence(
                    self.symbol, self.timeframe, result
                )
                
                # Ghi nhận dự đoán để theo dõi
                self.confidence_evaluator.log_prediction(self.symbol, self.timeframe, result)
            
            return result
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _calculate_confidence(self, data, predictions, current_price):
        """
        Tính độ tin cậy của dự đoán
        
        Args:
            data (DataFrame): Dữ liệu đầu vào
            predictions (array): Dự đoán từ mô hình
            current_price (float): Giá hiện tại
            
        Returns:
            float: Độ tin cậy (từ 0 đến 100)
        """
        try:
            # Phân tích trên dữ liệu gần đây (20 ngày)
            recent_data = data.tail(20)
            
            # Tính độ biến động (volatility)
            if len(recent_data) >= 5:
                volatility = recent_data['close'].pct_change().std() * 100
                
                # Volatility cao -> độ tin cậy thấp
                volatility_factor = max(0, min(1, 2 - volatility / 3))  # Scale to 0-1
            else:
                volatility_factor = 0.5  # Default
            
            # Consistency của các chỉ báo kỹ thuật
            indicator_consistency = 0.5  # Default
            
            # Kiểm tra sự nhất quán giữa các chỉ báo
            if all(col in data.columns for col in ['rsi_14', 'macd', 'ema_9', 'sma_20']):
                # RSI > 70 hoặc < 30 là tín hiệu mạnh
                rsi = recent_data['rsi_14'].iloc[-1] if len(recent_data) > 0 else 50
                
                # MACD > 0 -> bullish, < 0 -> bearish
                macd = recent_data['macd'].iloc[-1] if len(recent_data) > 0 else 0
                
                # EMA < Price -> bullish, > Price -> bearish
                ema = recent_data['ema_9'].iloc[-1] if len(recent_data) > 0 else current_price
                
                # SMA < Price -> bullish, > Price -> bearish
                sma = recent_data['sma_20'].iloc[-1] if len(recent_data) > 0 else current_price
                
                # Dự đoán
                prediction = predictions[-1][0]
                predicted_price = self.price_scaler.inverse_transform([[prediction]])[0][0]
                
                # Kiểm tra sự nhất quán
                is_bullish_prediction = predicted_price > current_price
                
                bullish_indicators = 0
                total_indicators = 4
                
                # RSI
                bullish_indicators += 1 if (rsi < 30 and is_bullish_prediction) or (rsi > 70 and not is_bullish_prediction) else 0
                
                # MACD
                bullish_indicators += 1 if (macd > 0 and is_bullish_prediction) or (macd < 0 and not is_bullish_prediction) else 0
                
                # EMA
                bullish_indicators += 1 if (ema < current_price and is_bullish_prediction) or (ema > current_price and not is_bullish_prediction) else 0
                
                # SMA
                bullish_indicators += 1 if (sma < current_price and is_bullish_prediction) or (sma > current_price and not is_bullish_prediction) else 0
                
                # Tính độ nhất quán
                indicator_consistency = bullish_indicators / total_indicators
            
            # Kết hợp các yếu tố để tính độ tin cậy
            base_confidence = (0.5 * volatility_factor + 0.5 * indicator_consistency) * 100
            
            # Điều chỉnh theo timeframe
            timeframe_modifier = {
                'intraday': 0.7,   # Intraday khó dự đoán hơn
                'five_day': 1.0,   # Five_day là cơ sở
                'monthly': 0.8     # Monthly chịu ảnh hưởng nhiều yếu tố
            }.get(self.timeframe, 1.0)
            
            final_confidence = base_confidence * timeframe_modifier
            
            # Giới hạn và làm tròn
            final_confidence = max(5, min(95, final_confidence))
            final_confidence = round(final_confidence, 1)
            
            return final_confidence
        except Exception as e:
            logger.error(f"Lỗi khi tính độ tin cậy: {str(e)}")
            return 50.0  # Default confidence
    
    def _generate_prediction_reason(self, prediction, current_price, data):
        """
        Tạo lý do cho dự đoán
        
        Args:
            prediction (float): Giá dự đoán
            current_price (float): Giá hiện tại
            data (DataFrame): Dữ liệu đầu vào
            
        Returns:
            str: Lý do cho dự đoán
        """
        try:
            # Phần trăm thay đổi
            percent_change = ((prediction - current_price) / current_price) * 100
            
            reasons = []
            
            # Xác định xu hướng
            if percent_change > 0:
                trend = "tăng"
            else:
                trend = "giảm"
            
            # Thêm lý do dựa trên chỉ số kỹ thuật
            # RSI
            if 'rsi_14' in data.columns:
                last_rsi = data['rsi_14'].iloc[-1]
                if last_rsi > 70:
                    reasons.append(f"RSI ({last_rsi:.1f}) đang ở vùng quá mua")
                elif last_rsi < 30:
                    reasons.append(f"RSI ({last_rsi:.1f}) đang ở vùng quá bán")
            
            # MACD
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                last_macd = data['macd'].iloc[-1]
                last_signal = data['macd_signal'].iloc[-1]
                
                if last_macd > last_signal:
                    reasons.append("MACD đang cho tín hiệu tích cực")
                else:
                    reasons.append("MACD đang cho tín hiệu tiêu cực")
            
            # Bollinger Bands
            if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
                last_price = data['close'].iloc[-1]
                last_upper = data['bb_upper'].iloc[-1]
                last_lower = data['bb_lower'].iloc[-1]
                
                if last_price > last_upper:
                    reasons.append("Giá đang vượt ngưỡng Bollinger Band trên")
                elif last_price < last_lower:
                    reasons.append("Giá đang dưới ngưỡng Bollinger Band dưới")
            
            # Moving Averages
            if 'sma_20' in data.columns and 'sma_50' in data.columns:
                last_sma20 = data['sma_20'].iloc[-1]
                last_sma50 = data['sma_50'].iloc[-1]
                
                if last_sma20 > last_sma50:
                    reasons.append("SMA20 đang nằm trên SMA50 (xu hướng tăng)")
                else:
                    reasons.append("SMA20 đang nằm dưới SMA50 (xu hướng giảm)")
            
            # Xu hướng gần đây
            recent_trend = "tăng" if data['close'].tail(5).pct_change().mean() > 0 else "giảm"
            
            # Tạo lý do tổng hợp
            if not reasons:
                return f"Giá có xu hướng {recent_trend} gần đây; Mô hình dự đoán giá {trend} {abs(percent_change):.1f}%"
            else:
                # Lấy tối đa 2 lý do
                selected_reasons = reasons[:2]
                reason_text = "; ".join(selected_reasons)
                return f"{reason_text}; Mô hình dự đoán giá {trend} {abs(percent_change):.1f}%"
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo lý do dự đoán: {str(e)}")
            return f"Mô hình dự đoán giá {trend} {abs(percent_change):.1f}%"