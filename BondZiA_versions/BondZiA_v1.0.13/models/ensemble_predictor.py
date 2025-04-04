import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
import joblib
from datetime import datetime, timedelta
import traceback

# Thêm thư mục gốc vào PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger_config import logger

class EnsemblePredictor:
    """
    Lớp dự đoán sử dụng phương pháp tổng hợp (Ensemble) nhiều mô hình khác nhau
    """
    
    def __init__(self, symbol, timeframe, config_path="../config/system_config.json"):
        """
        Khởi tạo EnsemblePredictor
        
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
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Lấy cấu hình cho khung thời gian
        self.prediction_config = self.config['prediction'][timeframe]
        self.lookback_window = self.prediction_config['lookback_window']
        
        # Tạo đường dẫn cho mô hình
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(root_dir, f"models/{timeframe}/ensemble")
        self.model_path = os.path.join(self.models_dir, f"{symbol}_ensemble_model.h5")
        self.meta_model_path = os.path.join(self.models_dir, f"{symbol}_meta_model.joblib")
        self.scaler_dir = os.path.join(self.models_dir, "scalers")
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.scaler_dir, exist_ok=True)
        
        # Đường dẫn đến scalers
        self.feature_scaler_path = os.path.join(self.scaler_dir, f"{symbol}_feature_scaler.joblib")
        self.price_scaler_path = os.path.join(self.scaler_dir, f"{symbol}_price_scaler.joblib")
        
        # Tham số dự đoán giá và độ tin cậy
        self.uncertainty_quantiles = [0.05, 0.25, 0.75, 0.95]  # Phân vị cho khoảng dự đoán
        
        # Mô hình chính
        self.model = None
        self.meta_model = None
        self.feature_scaler = None
        self.price_scaler = None
        
        # Danh sách base models
        self.base_models = []
        
        logger.info(f"Khởi tạo EnsemblePredictor cho {symbol} - {timeframe}")

    def check_model_exists(self):
        """
        Kiểm tra xem mô hình đã tồn tại chưa
        
        Returns:
            bool: True nếu mô hình đã tồn tại, False nếu chưa
        """
        # Kiểm tra cả mô hình chính và meta model
        return os.path.exists(self.model_path) and os.path.exists(self.meta_model_path)
    
    def _prepare_data(self, data):
        """
        Chuẩn bị dữ liệu cho quá trình huấn luyện và dự đoán
        
        Args:
            data (DataFrame): Dữ liệu cần chuẩn bị
            
        Returns:
            tuple: X, y, feature_scaler, price_scaler
        """
        try:
            # Đảm bảo dữ liệu đã có đủ chỉ báo kỹ thuật
            required_features = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi_14', 'macd', 'sma_20', 'ema_9', 'bb_upper', 'bb_lower',
                'stoch_k', 'stoch_d', 'obv', 'atr_14', 'adx_14'
            ]
            
            # Kiểm tra xem có thiếu feature nào
            missing_features = [f for f in required_features if f not in data.columns]
            
            if missing_features:
                logger.warning(f"Thiếu các tính năng: {missing_features}. Sẽ bỏ qua các tính năng này.")
                required_features = [f for f in required_features if f in data.columns]
            
            # Lọc dữ liệu
            data = data[required_features].copy()
            
            # Xử lý missing values
            data.fillna(method='ffill', inplace=True)
            data.fillna(method='bfill', inplace=True)
            
            # Kiểm tra xem còn missing values không
            if data.isnull().any().any():
                logger.warning("Vẫn còn missing values sau khi xử lý.")
                data.dropna(inplace=True)
            
            # Lấy dữ liệu giá
            y_data = data['close'].values.reshape(-1, 1)
            
            # Lấy dữ liệu tính năng
            X_data = data[required_features].values
            
            # Chuẩn hóa dữ liệu
            if os.path.exists(self.feature_scaler_path) and os.path.exists(self.price_scaler_path):
                # Tải scalers đã lưu
                feature_scaler = joblib.load(self.feature_scaler_path)
                price_scaler = joblib.load(self.price_scaler_path)
            else:
                # Tạo scalers mới
                feature_scaler = StandardScaler()
                feature_scaler.fit(X_data)
                
                price_scaler = MinMaxScaler(feature_range=(0, 1))
                price_scaler.fit(y_data)
            
            # Chuẩn hóa dữ liệu
            X_scaled = feature_scaler.transform(X_data)
            y_scaled = price_scaler.transform(y_data)
            
            # Tạo chuỗi thời gian
            X_sequences = []
            y_sequences = []
            
            for i in range(self.lookback_window, len(X_scaled)):
                X_sequences.append(X_scaled[i-self.lookback_window:i])
                y_sequences.append(y_scaled[i])
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            return X_sequences, y_sequences, feature_scaler, price_scaler
        
        except Exception as e:
            logger.error(f"Lỗi khi chuẩn bị dữ liệu: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None, None, None

    def _create_base_models(self):
        """
        Tạo các mô hình cơ sở cho ensemble
        
        Returns:
            list: Danh sách các mô hình cơ sở
        """
        models = []
        input_shape = (self.lookback_window, len(self.feature_scaler.mean_))
        
        # Mô hình 1: LSTM đơn giản
        model1 = Sequential([
            LSTM(64, input_shape=input_shape),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model1.compile(optimizer='adam', loss='mse')
        models.append(('LSTM_Simple', model1))
        
        # Mô hình 2: LSTM sâu hơn
        model2 = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model2.compile(optimizer='adam', loss='mse')
        models.append(('LSTM_Deep', model2))
        
        # Mô hình 3: GRU
        model3 = Sequential([
            GRU(64, input_shape=input_shape),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model3.compile(optimizer='adam', loss='mse')
        models.append(('GRU', model3))
        
        # Mô hình 4: CNN-LSTM
        model4 = Sequential([
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model4.compile(optimizer='adam', loss='mse')
        models.append(('CNN_LSTM', model4))
        
        # Mô hình 5: Bidirectional LSTM
        model5 = Sequential([
            tf.keras.layers.Bidirectional(LSTM(64), input_shape=input_shape),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model5.compile(optimizer='adam', loss='mse')
        models.append(('Bidirectional_LSTM', model5))
        
        return models
    
    def _create_meta_model(self):
        """
        Tạo meta model để kết hợp các dự đoán từ các mô hình cơ sở
        
        Returns:
            object: Meta model
        """
        # Meta model là GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

    def train(self, data):
        """
        Huấn luyện mô hình ensemble
        
        Args:
            data (DataFrame): Dữ liệu huấn luyện
            
        Returns:
            dict: Kết quả huấn luyện
        """
        try:
            logger.info(f"Bắt đầu huấn luyện mô hình ensemble cho {self.symbol} - {self.timeframe}")
            
            # Chuẩn bị dữ liệu
            X_train, y_train, feature_scaler, price_scaler = self._prepare_data(data)
            
            if X_train is None or y_train is None:
                logger.error("Không thể chuẩn bị dữ liệu huấn luyện")
                return None
            
            # Chia dữ liệu cho huấn luyện và kiểm tra
            split_idx = int(len(X_train) * 0.8)
            X_train_base = X_train[:split_idx]
            X_val_base = X_train[split_idx:]
            y_train_base = y_train[:split_idx]
            y_val_base = y_train[split_idx:]
            
            # Tạo các mô hình cơ sở
            base_models = self._create_base_models()
            
            # Huấn luyện từng mô hình cơ sở
            trained_models = []
            
            for name, model in base_models:
                logger.info(f"Huấn luyện mô hình cơ sở: {name}")
                
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True
                )
                
                model.fit(
                    X_train_base, y_train_base,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val_base, y_val_base),
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                trained_models.append((name, model))
            
            # Tạo dữ liệu cho meta model
            meta_features = np.zeros((len(X_val_base), len(trained_models)))
            
            for i, (name, model) in enumerate(trained_models):
                # Dự đoán trên tập validation
                preds = model.predict(X_val_base)
                meta_features[:, i] = preds.flatten()
            
            # Huấn luyện meta model
            meta_model = self._create_meta_model()
            meta_model.fit(meta_features, y_val_base.flatten())
            
            # Đánh giá trên tập validation
            meta_preds = meta_model.predict(meta_features)
            mse = np.mean((y_val_base.flatten() - meta_preds) ** 2)
            mae = np.mean(np.abs(y_val_base.flatten() - meta_preds))
            
            # Lưu các mô hình
            # Tạo thư mục base_models nếu chưa tồn tại
            base_models_dir = os.path.join(self.models_dir, "base_models")
            os.makedirs(base_models_dir, exist_ok=True)
            
            for name, model in trained_models:
                model_path = os.path.join(base_models_dir, f"{self.symbol}_{name}_model.h5")
                model.save(model_path)
            
            # Lưu meta model
            joblib.dump(meta_model, self.meta_model_path)
            
            # Lưu scalers
            joblib.dump(feature_scaler, self.feature_scaler_path)
            joblib.dump(price_scaler, self.price_scaler_path)
            
            # Lưu thông tin mô hình ensemble
            ensemble_info = {
                'base_models': [name for name, _ in trained_models],
                'meta_model': 'GradientBoostingRegressor',
                'mse': float(mse),
                'mae': float(mae),
                'input_shape': input_shape,
                'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Lưu thông tin ensemble
            ensemble_info_path = os.path.join(self.models_dir, f"{self.symbol}_ensemble_info.json")
            with open(ensemble_info_path, 'w') as f:
                json.dump(ensemble_info, f, indent=4)
            
            # Lưu tham chiếu để dùng sau
            self.base_models = trained_models
            self.meta_model = meta_model
            self.feature_scaler = feature_scaler
            self.price_scaler = price_scaler
            
            logger.info(f"Đã huấn luyện xong mô hình ensemble cho {self.symbol} - {self.timeframe}")
            
            return ensemble_info
            
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình ensemble: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def predict(self, data, calculate_confidence=True):
        """
        Dự đoán giá cổ phiếu và tính toán độ tin cậy
        
        Args:
            data (DataFrame): Dữ liệu đầu vào
            calculate_confidence (bool): Có tính toán độ tin cậy không
            
        Returns:
            dict: Kết quả dự đoán
        """
        try:
            # Chuẩn bị dữ liệu đầu vào
            X, _, feature_scaler, price_scaler = self._prepare_data(data)
            
            if X is None:
                logger.error("Không thể chuẩn bị dữ liệu dự đoán")
                return None
            
            # Nếu chưa tải mô hình và scalers
            if self.meta_model is None or self.feature_scaler is None or self.price_scaler is None:
                self._load_models()
            
            # Dự đoán từ các mô hình cơ sở
            base_predictions = np.zeros((len(X), len(self.base_models)))
            
            # Tải và dự đoán từng mô hình cơ sở
            base_models_dir = os.path.join(self.models_dir, "base_models")
            
            for i, (name, _) in enumerate(self.base_models):
                model_path = os.path.join(base_models_dir, f"{self.symbol}_{name}_model.h5")
                model = load_model(model_path)
                
                preds = model.predict(X)
                base_predictions[:, i] = preds.flatten()
            
            # Dự đoán bằng meta model
            meta_predictions = self.meta_model.predict(base_predictions)
            
            # Chuyển dự đoán về giá thực tế
            price_predictions = self.price_scaler.inverse_transform(meta_predictions.reshape(-1, 1)).flatten()
            
            # Lấy giá hiện tại
            current_price = data['close'].iloc[-1]
            
            # Tính toán các giá trị dự đoán
            if self.timeframe == 'intraday':
                # Intraday: dự đoán giá cuối ngày
                prediction = price_predictions[-1]
                
                # Tính % thay đổi
                percent_change = ((prediction - current_price) / current_price) * 100
                
                result = {
                    'price': float(prediction),
                    'percent_change': float(percent_change),
                    'current_price': float(current_price)
                }
                
            elif self.timeframe == 'five_day':
                # Five_day: dự đoán giá sau 5 ngày
                prediction = price_predictions[-1]
                
                # Tính % thay đổi
                percent_change = ((prediction - current_price) / current_price) * 100
                
                result = {
                    'price': float(prediction),
                    'percent_change': float(percent_change),
                    'current_price': float(current_price)
                }
                
            elif self.timeframe == 'monthly':
                # Monthly: dự đoán giá sau 1 tháng
                prediction = price_predictions[-1]
                
                # Tính % thay đổi
                percent_change = ((prediction - current_price) / current_price) * 100
                
                result = {
                    'price': float(prediction),
                    'percent_change': float(percent_change),
                    'current_price': float(current_price)
                }

            # Tính toán độ tin cậy nếu cần
            if calculate_confidence:
                # Tính toán độ phân tán của dự đoán từ các mô hình cơ sở
                base_price_predictions = self.price_scaler.inverse_transform(base_predictions[-1].reshape(-1, 1)).flatten()
                
                # Độ lệch chuẩn của các dự đoán
                prediction_std = np.std(base_price_predictions)
                
                # Hệ số biến thiên (CV - Coefficient of Variation)
                cv = prediction_std / np.mean(base_price_predictions)
                
                # Tính độ tin cậy dựa trên CV (càng thấp càng tốt)
                confidence = np.exp(-cv * 5) * 100  # Chuyển đổi phi tuyến
                
                # Giới hạn độ tin cậy trong khoảng 0-100%
                confidence = max(0, min(100, confidence))
                
                # Phân tích độ chính xác lịch sử
                historical_accuracy = self._calculate_historical_accuracy()
                
                # Trọng số cho đánh giá cuối cùng
                # 60% dựa trên sự đồng thuận của mô hình, 40% dựa trên độ chính xác lịch sử
                final_confidence = 0.6 * confidence + 0.4 * historical_accuracy
                
                # Làm tròn và giới hạn lại
                final_confidence = round(max(0, min(100, final_confidence)), 1)
                
                # Thêm vào kết quả
                result['confidence'] = float(final_confidence)
                
                # Thêm lý do cho dự đoán
                result['reason'] = self._generate_prediction_reason(prediction, current_price, data)
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _load_models(self):
        """
        Tải các mô hình và scalers đã lưu
        
        Returns:
            bool: True nếu tải thành công, False nếu không
        """
        try:
            # Kiểm tra xem có đủ file không
            if not os.path.exists(self.meta_model_path):
                logger.error(f"Không tìm thấy meta model tại {self.meta_model_path}")
                return False
            
            if not os.path.exists(self.feature_scaler_path):
                logger.error(f"Không tìm thấy feature scaler tại {self.feature_scaler_path}")
                return False
            
            if not os.path.exists(self.price_scaler_path):
                logger.error(f"Không tìm thấy price scaler tại {self.price_scaler_path}")
                return False
            
            # Tải meta model
            self.meta_model = joblib.load(self.meta_model_path)
            
            # Tải scalers
            self.feature_scaler = joblib.load(self.feature_scaler_path)
            self.price_scaler = joblib.load(self.price_scaler_path)
            
            # Tải thông tin mô hình ensemble để biết tên các mô hình cơ sở
            ensemble_info_path = os.path.join(self.models_dir, f"{self.symbol}_ensemble_info.json")
            with open(ensemble_info_path, 'r') as f:
                ensemble_info = json.load(f)
            
            # Thiết lập tham chiếu base_models
            self.base_models = [(name, None) for name in ensemble_info['base_models']]
            
            logger.info(f"Đã tải mô hình ensemble cho {self.symbol} - {self.timeframe}")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _calculate_historical_accuracy(self):
        """
        Tính toán độ chính xác lịch sử
        
        Returns:
            float: Điểm độ chính xác lịch sử (từ 0-100)
        """
        try:
            # Đường dẫn đến file lịch sử dự đoán
            history_dir = os.path.join(os.path.dirname(self.models_dir), "history")
            os.makedirs(history_dir, exist_ok=True)
            
            history_path = os.path.join(history_dir, f"{self.symbol}_{self.timeframe}_history.json")
            
            # Nếu không có file lịch sử, trả về giá trị mặc định
            if not os.path.exists(history_path):
                # Nếu không có lịch sử, chúng ta đặt giá trị trung bình
                return 50.0
            
            # Đọc lịch sử dự đoán
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            # Tính direction accuracy
            correct_directions = 0
            total_predictions = len(history)
            
            if total_predictions == 0:
                return 50.0
            
            for prediction in history:
                actual_change = prediction.get('actual_change', 0)
                predicted_change = prediction.get('predicted_change', 0)
                
                # Kiểm tra hướng đúng
                if (actual_change > 0 and predicted_change > 0) or (actual_change < 0 and predicted_change < 0):
                    correct_directions += 1
            
            # Tính direction accuracy
            direction_accuracy = (correct_directions / total_predictions) * 100
            
            # Nếu chỉ có ít dự đoán, giảm độ tin cậy
            if total_predictions < 10:
                direction_accuracy = direction_accuracy * (total_predictions / 10) + 50 * (1 - total_predictions / 10)
            
            return direction_accuracy
        except Exception as e:
            logger.error(f"Lỗi khi tính độ chính xác lịch sử: {str(e)}")
            return 50.0
    
    def _generate_prediction_reason(self, prediction, current_price, data):
        """
        Tạo lý do cho dự đoán
        
        Args:
            prediction (float): Giá dự đoán
            current_price (float): Giá hiện tại
            data (DataFrame): Dữ liệu
            
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
            
            # Tạo lý do tổng hợp
            if not reasons:
                return f"Giá có xu hướng {trend} gần đây; Mô hình dự đoán giá {trend} {abs(percent_change):.1f}%"
            else:
                # Lấy tối đa 2 lý do
                selected_reasons = reasons[:2]
                reason_text = "; ".join(selected_reasons)
                return f"{reason_text}; Mô hình dự đoán giá {trend} {abs(percent_change):.1f}%"
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo lý do dự đoán: {str(e)}")
            trend = "tăng" if prediction > current_price else "giảm"
            percent_change = ((prediction - current_price) / current_price) * 100
            return f"Mô hình dự đoán giá {trend} {abs(percent_change):.1f}%"