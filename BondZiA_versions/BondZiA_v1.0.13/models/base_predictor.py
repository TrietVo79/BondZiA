import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from datetime import datetime, timedelta
import traceback
from utils.logger_config import logger

# Thêm thư mục gốc vào PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BasePredictor:
    """Lớp cơ sở cho các mô hình dự đoán"""
    
    def __init__(self, symbol, timeframe, config_path="../config/system_config.json"):
        """
        Khởi tạo BasePredictor
        
        Args:
            symbol (str): Mã cổ phiếu
            timeframe (str): Khung thời gian ('intraday', 'five_day', 'monthly')
            config_path (str): Đường dẫn đến file cấu hình
        """
        # Lưu thông tin cơ bản
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Đọc cấu hình
        self.config_path = config_path
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            # Kiểm tra và tạo cấu hình mặc định nếu cần
            if 'prediction' not in self.config:
                logger.warning(f"Không tìm thấy khóa 'prediction' trong file cấu hình. Sử dụng giá trị mặc định.")
                self.config['prediction'] = {}
            
            if timeframe not in self.config['prediction']:
                logger.warning(f"Không tìm thấy khóa '{timeframe}' trong cấu hình prediction. Sử dụng giá trị mặc định.")
                self.config['prediction'][timeframe] = {}
            
            # Lấy cấu hình cho khung thời gian
            self.prediction_config = self.config['prediction'][timeframe]
            
            # Thiết lập các giá trị mặc định nếu không tìm thấy trong cấu hình
            self.model_type = self.prediction_config.get('model_type', 'lstm')
            self.lookback_window = self.prediction_config.get('lookback_window', 30)
            self.confidence_threshold = self.prediction_config.get('confidence_threshold', 70)
            
            # Ghi log cấu hình đang sử dụng để debug
            logger.info(f"Sử dụng cấu hình prediction cho {timeframe}: {self.prediction_config}")
        
        except Exception as e:
            logger.error(f"Lỗi khi truy cập cấu hình prediction: {str(e)}")
            logger.info(f"Thử in ra cấu hình hiện tại: {self.config if 'self.config' in locals() else 'Không có cấu hình'}")
            
            # Thiết lập các giá trị mặc định
            self.model_type = 'lstm'
            self.lookback_window = 30
            self.confidence_threshold = 70
            self.prediction_config = {
                "model_type": self.model_type,
                "lookback_window": self.lookback_window,
                "features": ["close", "volume", "rsi_14", "macd"],
                "confidence_threshold": self.confidence_threshold
            }
            logger.warning(f"Sử dụng cấu hình mặc định cho {timeframe}")
        
        # Khởi tạo scaler
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        
        # Khởi tạo model
        self.model = None
        
        # Đường dẫn đến thư mục lưu mô hình
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), timeframe)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Đường dẫn đến thư mục lưu scaler
        self.scalers_dir = os.path.join(self.models_dir, "scalers")
        os.makedirs(self.scalers_dir, exist_ok=True)
        
        # Các tính năng mặc định
        self.default_features = [
            'open', 'high', 'low', 'close', 'volume', 
            'rsi_14', 'macd', 'sma_20', 'ema_9'
        ]
        
        logger.info(f"Khởi tạo BasePredictor cho {symbol} - {timeframe} với mô hình {self.model_type}")
    
    def _get_model_path(self):
        """
        Lấy đường dẫn đến file mô hình
        
        Returns:
            str: Đường dẫn đến file mô hình
        """
        return os.path.join(self.models_dir, f"{self.symbol}_{self.timeframe}_model.h5")
    
    def _get_price_scaler_path(self):
        """
        Lấy đường dẫn đến file price scaler
        
        Returns:
            str: Đường dẫn đến file price scaler
        """
        return os.path.join(self.scalers_dir, f"{self.symbol}_{self.timeframe}_price_scaler.pkl")
    
    def _get_feature_scaler_path(self):
        """
        Lấy đường dẫn đến file feature scaler
        
        Returns:
            str: Đường dẫn đến file feature scaler
        """
        return os.path.join(self.scalers_dir, f"{self.symbol}_{self.timeframe}_feature_scaler.pkl")
    
    def check_model_exists(self):
        """
        Kiểm tra xem mô hình đã tồn tại hay chưa
        
        Returns:
            bool: True nếu mô hình tồn tại, False nếu không
        """
        model_path = self._get_model_path()
        price_scaler_path = self._get_price_scaler_path()
        feature_scaler_path = self._get_feature_scaler_path()
        
        # Kiểm tra cả mô hình và scalers
        model_exists = os.path.exists(model_path)
        price_scaler_exists = os.path.exists(price_scaler_path)
        feature_scaler_exists = os.path.exists(feature_scaler_path)
        
        # Ghi log kết quả kiểm tra
        logger.debug(f"Kiểm tra mô hình {self.symbol}_{self.timeframe}: model={model_exists}, price_scaler={price_scaler_exists}, feature_scaler={feature_scaler_exists}")
        
        # Trả về True nếu cả ba file đều tồn tại
        return model_exists and price_scaler_exists and feature_scaler_exists
    
    def _build_model(self, input_shape):
        """
        Xây dựng mô hình dự đoán cơ bản (được ghi đè bởi các lớp con)
        
        Args:
            input_shape (tuple): Kích thước đầu vào
            
        Returns:
            Model: Mô hình Keras
        """
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def _prepare_data(self, data, features=None):
        """
        Chuẩn bị dữ liệu cho mô hình
        
        Args:
            data (DataFrame): Dữ liệu đầu vào
            features (list, optional): Danh sách các tính năng sử dụng
            
        Returns:
            tuple: X, y, X_scaler, y_scaler
        """
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
        
        # Tạo các mảng X và y
        X_data = []
        y_data = []
        
        # Chuẩn hóa dữ liệu
        price_data = data[['close']].values
        self.price_scaler.fit(price_data)
        scaled_prices = self.price_scaler.transform(price_data)
        
        feature_data = data[available_features].values
        self.feature_scaler.fit(feature_data)
        scaled_features = self.feature_scaler.transform(feature_data)
        
        # Tạo các mẫu với cửa sổ đánh dấu
        for i in range(self.lookback_window, len(data)):
            X_data.append(scaled_features[i-self.lookback_window:i])
            y_data.append(scaled_prices[i])
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        return X, y
    
    def train(self, data, features=None, epochs=100, batch_size=32, validation_split=0.2):
        """
        Huấn luyện mô hình
        
        Args:
            data (DataFrame): Dữ liệu huấn luyện
            features (list, optional): Danh sách các tính năng sử dụng
            epochs (int): Số epoch huấn luyện
            batch_size (int): Kích thước batch
            validation_split (float): Tỷ lệ dữ liệu dùng cho validation
            
        Returns:
            dict: Lịch sử huấn luyện
        """
        try:
            # Chuẩn bị dữ liệu
            X, y = self._prepare_data(data, features)
            
            if X.shape[0] == 0:
                logger.error(f"Không đủ dữ liệu để huấn luyện mô hình {self.symbol} - {self.timeframe}")
                return None
            
            # Xây dựng mô hình
            self.model = self._build_model(input_shape=(X.shape[1], X.shape[2]))
            
            # Thiết lập callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            model_path = self._get_model_path()
            checkpoint = ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
            
            # Tính validation_split dựa trên số lượng mẫu
            sample_count = X.shape[0]
            if sample_count <= 5:
                adaptive_val_split = 0  # Nếu chỉ có 1 mẫu, không dùng validation
            else:
                adaptive_val_split = min(0.1, max(0.01, 5/sample_count))
            logger.info(f"Áp dụng validation_split={adaptive_val_split} cho {sample_count} mẫu")

            # Huấn luyện mô hình
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=adaptive_val_split,
                callbacks=[early_stopping, checkpoint],
                verbose=1
            )
            
            # Lưu scalers
            price_scaler_path = self._get_price_scaler_path()
            feature_scaler_path = self._get_feature_scaler_path()
            
            joblib.dump(self.price_scaler, price_scaler_path)
            joblib.dump(self.feature_scaler, feature_scaler_path)
            
            logger.info(f"Đã huấn luyện và lưu mô hình {self.symbol} - {self.timeframe}")
            
            return history.history
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình {self.symbol} - {self.timeframe}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def load_model(self):
        """
        Tải mô hình đã lưu
        
        Returns:
            bool: True nếu tải thành công, False nếu thất bại
        """
        try:
            model_path = self._get_model_path()
            price_scaler_path = self._get_price_scaler_path()
            feature_scaler_path = self._get_feature_scaler_path()
            
            if not os.path.exists(model_path):
                logger.warning(f"Không tìm thấy mô hình {model_path}")
                return False
            
            if not os.path.exists(price_scaler_path) or not os.path.exists(feature_scaler_path):
                logger.warning(f"Không tìm thấy scalers cho mô hình {self.symbol} - {self.timeframe}")
                return False
            
            # Tải mô hình
            self.model = load_model(model_path)
            
            # Tải scalers
            self.price_scaler = joblib.load(price_scaler_path)
            self.feature_scaler = joblib.load(feature_scaler_path)
            
            logger.info(f"Đã tải mô hình {self.symbol} - {self.timeframe}")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình {self.symbol} - {self.timeframe}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def evaluate(self, data, features=None):
        """
        Đánh giá mô hình
        
        Args:
            data (DataFrame): Dữ liệu đánh giá
            features (list, optional): Danh sách các tính năng sử dụng
            
        Returns:
            dict: Các chỉ số đánh giá
        """
        try:
            # Chuẩn bị dữ liệu
            X, y = self._prepare_data(data, features)
            
            if X.shape[0] == 0:
                logger.error(f"Không đủ dữ liệu để đánh giá mô hình {self.symbol} - {self.timeframe}")
                return None
            
            # Đảm bảo mô hình đã được tải
            if self.model is None:
                if not self.load_model():
                    logger.error(f"Không thể tải mô hình {self.symbol} - {self.timeframe}")
                    return None
            
            # Dự đoán
            y_pred_scaled = self.model.predict(X)
            
            # Chuyển đổi lại
            y_true = self.price_scaler.inverse_transform(y)
            y_pred = self.price_scaler.inverse_transform(y_pred_scaled)
            
            # Tính các chỉ số
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # Tính định hướng chính xác
            direction_true = np.diff(y_true.flatten())
            direction_pred = np.diff(y_pred.flatten())
            direction_accuracy = np.mean((direction_true > 0) == (direction_pred > 0)) * 100
            
            results = {
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'direction_accuracy': float(direction_accuracy)
            }
            
            logger.info(f"Đánh giá mô hình {self.symbol} - {self.timeframe}: {results}")
            
            return results
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá mô hình {self.symbol} - {self.timeframe}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def predict(self, data, features=None):
        """
        Dự đoán giá cổ phiếu
        
        Args:
            data (DataFrame): Dữ liệu đầu vào
            features (list, optional): Danh sách các tính năng sử dụng
            
        Returns:
            dict: Kết quả dự đoán
        """
        try:
            # Đảm bảo mô hình đã được tải
            if self.model is None:
                if not self.load_model():
                    logger.error(f"Không thể tải mô hình {self.symbol} - {self.timeframe}")
                    return None
            
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
            
            # Lấy dữ liệu gần nhất
            recent_data = data.iloc[-self.lookback_window:].copy()
            
            if len(recent_data) < self.lookback_window:
                logger.error(f"Không đủ dữ liệu cho cửa sổ đánh dấu {self.lookback_window}")
                return None
            
            # Chuẩn hóa dữ liệu
            feature_data = recent_data[available_features].values
            scaled_features = self.feature_scaler.transform(feature_data)
            
            # Chuẩn bị dữ liệu đầu vào
            X = np.array([scaled_features])
            
            # Dự đoán
            y_pred_scaled = self.model.predict(X)
            
            # Chuyển đổi lại
            y_pred = self.price_scaler.inverse_transform(y_pred_scaled)
            
            # Lấy giá hiện tại
            current_price = data['close'].iloc[-1]
            
            # Tính hướng và độ tin cậy
            price_change = (y_pred[0][0] - current_price) / current_price * 100
            
            # Xác định hướng
            if price_change > 0:
                direction = 'up'
            elif price_change < 0:
                direction = 'down'
            else:
                direction = 'neutral'
            
            # Log chi tiết để kiểm tra
            logger.info(f"DEBUG - {self.symbol} prediction details: current={current_price:.2f}, predicted={y_pred[0][0]:.2f}, change={price_change:.2f}%, direction={direction}")

            # Tính độ tin cậy dựa trên hiệu suất quá khứ
            confidence = min(abs(price_change) * 5, 99)  # Scale độ tin cậy theo % thay đổi
            
            # Lý do dự đoán
            reason = self._generate_prediction_reason(data, direction, price_change)
            
            # Kết quả
            result = {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'current_price': float(current_price),
                'predicted_price': float(y_pred[0][0]),
                'price_change_percent': float(price_change),
                'direction': direction,
                'confidence': float(confidence),
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Dự đoán {self.symbol} - {self.timeframe}: {result['direction']} ({result['confidence']:.1f}%) - Giá: {result['predicted_price']:.2f}")
            
            return result
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán {self.symbol} - {self.timeframe}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Thêm cơ chế tự phục hồi
            logger.warning(f"Đang cố gắng khôi phục mô hình {self.symbol} - {self.timeframe}...")
            try:
                # Xóa mô hình cũ nếu có lỗi
                self.model = None
                # Thử tải lại mô hình
                if self.load_model():
                    logger.info(f"Đã tải lại mô hình {self.symbol} - {self.timeframe}")
                    # Thử dự đoán lại (đệ quy với độ sâu = 1)
                    return self.predict(data, features)
            except Exception as recovery_error:
                logger.error(f"Không thể khôi phục mô hình {self.symbol} - {self.timeframe}: {recovery_error}")
    
            return None
    
    def _generate_prediction_reason(self, data, direction, price_change):
        """
        Tạo lý do cho dự đoán
        
        Args:
            data (DataFrame): Dữ liệu
            direction (str): Hướng dự đoán
            price_change (float): Phần trăm thay đổi giá
            
        Returns:
            str: Lý do dự đoán
        """
        reasons = []
        
        # Lý do chính luôn phải phản ánh hướng dự đoán
        if direction == 'up':
            main_reason = f"Giá có xu hướng tăng gần đây"
        elif direction == 'down':
            main_reason = f"Giá có xu hướng giảm gần đây"
        else:
            main_reason = "Giá dự kiến ít thay đổi"
            
        reasons.append(main_reason)
        
        # Kiểm tra khối lượng
        if 'volume' in data.columns:
            avg_volume = data['volume'].iloc[-10:-1].mean()
            last_volume = data['volume'].iloc[-1]
            
            if last_volume > avg_volume * 1.5:
                if direction == 'up':
                    reasons.append("Khối lượng giao dịch tăng đột biến")
            elif last_volume < avg_volume * 0.5:
                if direction == 'down':
                    reasons.append("Khối lượng giao dịch giảm mạnh")
        
        # Kiểm tra RSI nếu có
        if 'rsi_14' in data.columns:
            last_rsi = data['rsi_14'].iloc[-1]
            
            if last_rsi > 70 and direction == 'down':
                reasons.append("RSI cho thấy tình trạng quá mua (>70)")
            elif last_rsi < 30 and direction == 'up':
                reasons.append("RSI cho thấy tình trạng quá bán (<30)")
        
        # Kiểm tra MACD nếu có
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            last_macd = data['macd'].iloc[-1]
            last_signal = data['macd_signal'].iloc[-1]
            prev_macd = data['macd'].iloc[-2]
            prev_signal = data['macd_signal'].iloc[-2]
            
            if prev_macd < prev_signal and last_macd > last_signal and direction == 'up':
                reasons.append("MACD vừa cắt lên đường tín hiệu (tín hiệu mua)")
            elif prev_macd > prev_signal and last_macd < last_signal and direction == 'down':
                reasons.append("MACD vừa cắt xuống đường tín hiệu (tín hiệu bán)")
        
        # Thêm thông tin về % thay đổi
        if direction == 'up':
            reasons.append(f"Mô hình dự đoán giá tăng {abs(price_change):.1f}%")
        elif direction == 'down':
            reasons.append(f"Mô hình dự đoán giá giảm {abs(price_change):.1f}%")
    
        return "; ".join(reasons)

# Các lớp con cụ thể cho từng khung thời gian
class IntradayPredictor(BasePredictor):
    """Lớp dự đoán Intraday"""
    
    def __init__(self, symbol, config_path="../config/system_config.json"):
        super().__init__(symbol, 'intraday', config_path)
    
    def _build_model(self, input_shape):
        """
        Xây dựng mô hình Temporal Fusion Transformer đơn giản cho dự đoán intraday
        
        Args:
            input_shape (tuple): Kích thước đầu vào
            
        Returns:
            Model: Mô hình Keras
        """
        # Đầu vào
        input_layer = Input(shape=input_shape)
        
        # LSTM layers với attention
        x = LSTM(64, return_sequences=True)(input_layer)
        x = Dropout(0.2)(x)
        
        # Lớp LSTM thứ hai với return_sequences=False để có đầu ra 2D
        x = LSTM(32, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        
        # Fully connected layers
        x = Dense(16, activation='relu')(x)
        output = Dense(1)(x)
        
        # Tạo mô hình
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model

class FiveDayPredictor(BasePredictor):
    """Lớp dự đoán 5 ngày"""
    
    def __init__(self, symbol, config_path="../config/system_config.json"):
        super().__init__(symbol, 'five_day', config_path)
    
    def _build_model(self, input_shape):
        """
        Xây dựng mô hình LSTM with Attention cho dự đoán 5 ngày
        
        Args:
            input_shape (tuple): Kích thước đầu vào
            
        Returns:
            Model: Mô hình Keras
        """
        # Mô hình đơn giản nhưng hiệu quả cho dự đoán 5 ngày
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
        return model

class MonthlyPredictor(BasePredictor):
    """Lớp dự đoán 1 tháng"""
    
    def __init__(self, symbol, config_path="../config/system_config.json"):
        super().__init__(symbol, 'monthly', config_path)
    
    def _build_model(self, input_shape):
        """
        Xây dựng mô hình TimeGPT đơn giản cho dự đoán 1 tháng
        
        Args:
            input_shape (tuple): Kích thước đầu vào
            
        Returns:
            Model: Mô hình Keras
        """
        # Input
        input_layer = Input(shape=input_shape)
        
        # LSTM layers
        x = LSTM(256, return_sequences=True)(input_layer)
        x = Dropout(0.4)(x)
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.4)(x)
        x = LSTM(64)(x)
        x = Dropout(0.3)(x)
        
        # Fully connected layers
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        output = Dense(1)(x)
        
        # Tạo mô hình
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
        
        return model

if __name__ == "__main__":
    # Test module
    logger.info("Kiểm tra module BasePredictor")
    
    # Tạo dữ liệu giả
    dates = pd.date_range(start='2023-01-01', periods=100)
    data = {
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(100, 5, 100),
        'volume': np.random.normal(1000000, 200000, 100),
        'rsi_14': np.random.normal(50, 10, 100),
        'macd': np.random.normal(0, 1, 100),
        'macd_signal': np.random.normal(0, 1, 100),
        'sma_20': np.random.normal(100, 5, 100),
        'ema_9': np.random.normal(100, 5, 100)
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Test IntradayPredictor
    intraday = IntradayPredictor('AAPL')
    intraday.train(df, epochs=5, batch_size=16)  # Chỉ chạy 5 epochs để test
    
    # Test dự đoán
    prediction = intraday.predict(df)
    logger.info(f"Dự đoán Intraday: {prediction}")
    
    # Test FiveDayPredictor
    five_day = FiveDayPredictor('AAPL')
    five_day.train(df, epochs=5, batch_size=16)  # Chỉ chạy 5 epochs để test
    
    # Test dự đoán
    prediction = five_day.predict(df)
    logger.info(f"Dự đoán 5 ngày: {prediction}")
    
    # Test MonthlyPredictor
    monthly = MonthlyPredictor('AAPL')
    monthly.train(df, epochs=5, batch_size=16)  # Chỉ chạy 5 epochs để test
    
    # Test dự đoán
    prediction = monthly.predict(df)
    logger.info(f"Dự đoán 1 tháng: {prediction}")