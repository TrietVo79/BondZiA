import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import random
from datetime import datetime
import traceback
from utils.logger_config import logger

# Thêm thư mục gốc vào PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HyperparameterTuner:
    """Lớp tối ưu hóa siêu tham số cho các mô hình dự đoán"""
    
    def __init__(self, symbol, timeframe, config_path="../config/system_config.json"):
        """
        Khởi tạo HyperparameterTuner
        
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
        
        # Lấy cấu hình tiến hóa
        self.evolution_config = self.config['evolution']
        self.max_trials = self.evolution_config['max_trials']
        self.early_stopping_patience = self.evolution_config['early_stopping_patience']
        self.evaluation_metric = self.evolution_config['evaluation_metric']
        
        # Lấy cấu hình cho khung thời gian
        self.prediction_config = self.config['prediction'][timeframe]
        self.lookback_window = self.prediction_config['lookback_window']
        
        # Đường dẫn lưu trữ kết quả
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "evolution/results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Đường dẫn đến thư mục lưu mô hình
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     f"models/{timeframe}")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Đường dẫn đến thư mục lưu scaler
        self.scalers_dir = os.path.join(self.models_dir, "scalers")
        os.makedirs(self.scalers_dir, exist_ok=True)
        
        # Các tính năng mặc định
        self.default_features = [
            'open', 'high', 'low', 'close', 'volume', 
            'rsi_14', 'macd', 'sma_20', 'ema_9'
        ]
        
        logger.info(f"Khởi tạo HyperparameterTuner cho {symbol} - {timeframe}")
    
    def _prepare_data(self, data, features=None):
        """
        Chuẩn bị dữ liệu cho mô hình
        
        Args:
            data (DataFrame): Dữ liệu đầu vào
            features (list, optional): Danh sách các tính năng sử dụng
            
        Returns:
            tuple: X_train, X_test, y_train, y_test, X_scaler, y_scaler
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
        price_scaler = MinMaxScaler(feature_range=(0, 1))
        price_data = data[['close']].values
        price_scaler.fit(price_data)
        scaled_prices = price_scaler.transform(price_data)
        
        feature_scaler = StandardScaler()
        feature_data = data[available_features].values
        feature_scaler.fit(feature_data)
        scaled_features = feature_scaler.transform(feature_data)
        
        # Tạo các mẫu với cửa sổ đánh dấu
        for i in range(self.lookback_window, len(data)):
            X_data.append(scaled_features[i-self.lookback_window:i])
            y_data.append(scaled_prices[i])
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        return X_train, X_test, y_train, y_test, feature_scaler, price_scaler
    
    def _build_model(self, hyperparameters, input_shape):
        """
        Xây dựng mô hình dựa trên hyperparameters
        
        Args:
            hyperparameters (dict): Các siêu tham số
            input_shape (tuple): Kích thước đầu vào
            
        Returns:
            Model: Mô hình Keras
        """
        # Lấy thông tin mô hình từ hyperparameters
        model_type = hyperparameters.get('model_type', 'lstm')
        num_layers = hyperparameters.get('num_layers', 2)
        units = hyperparameters.get('units', [64, 32])
        dropout_rates = hyperparameters.get('dropout_rates', [0.2, 0.2])
        learning_rate = hyperparameters.get('learning_rate', 0.001)
        activation = hyperparameters.get('activation', 'relu')
        
        # Nếu num_layers không khớp với độ dài của units và dropout_rates, điều chỉnh
        if len(units) != num_layers:
            units = units[:num_layers] if len(units) > num_layers else units + [32] * (num_layers - len(units))
        
        if len(dropout_rates) != num_layers:
            dropout_rates = dropout_rates[:num_layers] if len(dropout_rates) > num_layers else dropout_rates + [0.2] * (num_layers - len(dropout_rates))
        
        # Xây dựng mô hình dựa trên loại
        if model_type == 'lstm':
            model = Sequential()
            
            # Lớp LSTM đầu tiên với return_sequences=True nếu có nhiều hơn 1 lớp
            model.add(LSTM(units[0], return_sequences=(num_layers > 1), input_shape=input_shape))
            model.add(Dropout(dropout_rates[0]))
            
            # Thêm các lớp LSTM tiếp theo
            for i in range(1, num_layers - 1):
                model.add(LSTM(units[i], return_sequences=True))
                model.add(Dropout(dropout_rates[i]))
            
            # Lớp LSTM cuối cùng
            if num_layers > 1:
                model.add(LSTM(units[-1]))
                model.add(Dropout(dropout_rates[-1]))
            
            # Thêm các lớp Dense
            dense_layers = hyperparameters.get('dense_layers', 1)
            dense_units = hyperparameters.get('dense_units', [16])
            
            if len(dense_units) != dense_layers:
                dense_units = dense_units[:dense_layers] if len(dense_units) > dense_layers else dense_units + [16] * (dense_layers - len(dense_units))
            
            for i in range(dense_layers):
                model.add(Dense(dense_units[i], activation=activation))
            
            # Lớp đầu ra
            model.add(Dense(1))
        
        elif model_type == 'gru':
            model = Sequential()
            
            # Lớp GRU đầu tiên với return_sequences=True nếu có nhiều hơn 1 lớp
            model.add(tf.keras.layers.GRU(units[0], return_sequences=(num_layers > 1), input_shape=input_shape))
            model.add(Dropout(dropout_rates[0]))
            
            # Thêm các lớp GRU tiếp theo
            for i in range(1, num_layers - 1):
                model.add(tf.keras.layers.GRU(units[i], return_sequences=True))
                model.add(Dropout(dropout_rates[i]))
            
            # Lớp GRU cuối cùng
            if num_layers > 1:
                model.add(tf.keras.layers.GRU(units[-1]))
                model.add(Dropout(dropout_rates[-1]))
            
            # Thêm các lớp Dense
            dense_layers = hyperparameters.get('dense_layers', 1)
            dense_units = hyperparameters.get('dense_units', [16])
            
            if len(dense_units) != dense_layers:
                dense_units = dense_units[:dense_layers] if len(dense_units) > dense_layers else dense_units + [16] * (dense_layers - len(dense_units))
            
            for i in range(dense_layers):
                model.add(Dense(dense_units[i], activation=activation))
            
            # Lớp đầu ra
            model.add(Dense(1))
        
        elif model_type == 'transformer':
            # Đầu vào
            input_layer = Input(shape=input_shape)
            
            # Transformer layers
            x = input_layer
            
            # Số lớp transformer
            transformer_layers = hyperparameters.get('transformer_layers', 1)
            
            for _ in range(transformer_layers):
                # Self-attention
                attention_dim = hyperparameters.get('attention_dim', 64)
                num_heads = hyperparameters.get('num_heads', 4)
                
                # Layer normalization
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
                
                # Multi-head attention
                mha = tf.keras.layers.MultiHeadAttention(
                    key_dim=attention_dim // num_heads,
                    num_heads=num_heads,
                    dropout=hyperparameters.get('attention_dropout', 0.1)
                )(x, x)
                
                # Skip connection
                x = tf.keras.layers.Add()([x, mha])
                
                # Feed-forward network
                ffn_dim = hyperparameters.get('ffn_dim', 128)
                
                # Layer normalization
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
                
                # Dense layers
                ffn = tf.keras.layers.Dense(ffn_dim, activation=activation)(x)
                ffn = tf.keras.layers.Dropout(hyperparameters.get('ffn_dropout', 0.1))(ffn)
                ffn = tf.keras.layers.Dense(input_shape[1])(ffn)
                
                # Skip connection
                x = tf.keras.layers.Add()([x, ffn])
            
            # Global average pooling để có đầu ra 2D
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            
            # Fully connected layers
            dense_layers = hyperparameters.get('dense_layers', 1)
            dense_units = hyperparameters.get('dense_units', [16])
            
            if len(dense_units) != dense_layers:
                dense_units = dense_units[:dense_layers] if len(dense_units) > dense_layers else dense_units + [16] * (dense_layers - len(dense_units))
            
            for i in range(dense_layers):
                x = Dense(dense_units[i], activation=activation)(x)
                x = Dropout(hyperparameters.get('dense_dropout', 0.2))(x)
            
            # Đầu ra
            output = Dense(1)(x)
            
            # Tạo mô hình
            model = Model(inputs=input_layer, outputs=output)
        
        else:
            raise ValueError(f"Không hỗ trợ loại mô hình: {model_type}")
        
        # Biên dịch mô hình
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        return model
    
    def _evaluate_model(self, model, X_test, y_test, price_scaler):
        """
        Đánh giá mô hình
        
        Args:
            model: Mô hình đã huấn luyện
            X_test: Dữ liệu kiểm tra
            y_test: Nhãn kiểm tra
            price_scaler: Bộ scaler giá
            
        Returns:
            dict: Các chỉ số đánh giá
        """
        # Dự đoán
        y_pred_scaled = model.predict(X_test)
        
        # Chuyển đổi lại
        y_true = price_scaler.inverse_transform(y_test)
        y_pred = price_scaler.inverse_transform(y_pred_scaled)
        
        # Tính các chỉ số
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Tính định hướng chính xác
        direction_true = np.diff(y_true.flatten())
        direction_pred = np.diff(y_pred.flatten())
        direction_accuracy = np.mean((direction_true > 0) == (direction_pred > 0)) * 100
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy)
        }
    
    def _generate_hyperparameters(self):
        """
        Tạo ngẫu nhiên các siêu tham số
        
        Returns:
            dict: Các siêu tham số
        """
        # Các loại mô hình hỗ trợ
        model_types = ['lstm', 'gru', 'transformer']
        
        # Lựa chọn loại mô hình
        model_type = random.choice(model_types)
        
        # Cài đặt chung
        num_layers = random.randint(1, 4)
        units = [random.choice([32, 64, 128, 256]) for _ in range(num_layers)]
        dropout_rates = [random.uniform(0.1, 0.5) for _ in range(num_layers)]
        learning_rate = random.choice([0.01, 0.005, 0.001, 0.0005, 0.0001])
        activation = random.choice(['relu', 'elu', 'tanh'])
        batch_size = random.choice([16, 32, 64, 128])
        
        # Cài đặt lớp Dense
        dense_layers = random.randint(1, 3)
        dense_units = [random.choice([16, 32, 64, 128]) for _ in range(dense_layers)]
        
        hyperparameters = {
            'model_type': model_type,
            'num_layers': num_layers,
            'units': units,
            'dropout_rates': dropout_rates,
            'learning_rate': learning_rate,
            'activation': activation,
            'batch_size': batch_size,
            'dense_layers': dense_layers,
            'dense_units': dense_units
        }
        
        # Cài đặt đặc biệt cho transformer
        if model_type == 'transformer':
            hyperparameters.update({
                'transformer_layers': random.randint(1, 3),
                'attention_dim': random.choice([32, 64, 128]),
                'num_heads': random.choice([2, 4, 8]),
                'attention_dropout': random.uniform(0.1, 0.3),
                'ffn_dim': random.choice([64, 128, 256]),
                'ffn_dropout': random.uniform(0.1, 0.3),
                'dense_dropout': random.uniform(0.1, 0.4)
            })
        
        return hyperparameters
    
    def tune(self, data, features=None, patience=None, trials=None):
        """
        Tối ưu hóa siêu tham số
        
        Args:
            data (DataFrame): Dữ liệu huấn luyện
            features (list, optional): Danh sách các tính năng sử dụng
            patience (int, optional): Số lần thử không cải thiện trước khi dừng
            trials (int, optional): Số lần thử tối đa
            
        Returns:
            dict: Siêu tham số tốt nhất và kết quả đánh giá
        """
        if patience is None:
            patience = self.early_stopping_patience
            
        if trials is None:
            trials = self.max_trials
        
        # Chuẩn bị dữ liệu
        X_train, X_test, y_train, y_test, feature_scaler, price_scaler = self._prepare_data(data, features)
        
        if X_train.shape[0] == 0:
            logger.error(f"Không đủ dữ liệu để huấn luyện mô hình {self.symbol} - {self.timeframe}")
            return None
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Tính năng được sử dụng
        if features is None:
            features = self.default_features
        
        used_features = [f for f in features if f in data.columns]
        
        # Theo dõi kết quả tốt nhất
        best_score = float('inf')  # Theo dõi RMSE/MSE thấp nhất
        best_hyperparameters = None
        best_metrics = None
        best_model = None
        no_improvement_count = 0
        
        logger.info(f"Bắt đầu tối ưu hóa siêu tham số cho {self.symbol} - {self.timeframe}")
        
        for trial in range(trials):
            try:
                # Tạo hyperparameters
                hyperparameters = self._generate_hyperparameters()
                
                logger.info(f"Trial {trial+1}/{trials}: {hyperparameters['model_type']}, "
                          f"layers: {hyperparameters['num_layers']}, "
                          f"units: {hyperparameters['units']}, "
                          f"lr: {hyperparameters['learning_rate']}")
                
                # Xây dựng mô hình
                model = self._build_model(hyperparameters, input_shape)
                
                # Huấn luyện mô hình
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=hyperparameters['batch_size'],
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Đánh giá mô hình
                metrics = self._evaluate_model(model, X_test, y_test, price_scaler)
                
                logger.info(f"Trial {trial+1} results: "
                          f"RMSE: {metrics['rmse']:.4f}, "
                          f"Direction Accuracy: {metrics['direction_accuracy']:.2f}%")
                
                # Cập nhật kết quả tốt nhất
                if metrics[self.evaluation_metric] < best_score:
                    best_score = metrics[self.evaluation_metric]
                    best_hyperparameters = hyperparameters
                    best_metrics = metrics
                    best_model = model
                    no_improvement_count = 0
                    
                    logger.info(f"New best model found! {self.evaluation_metric}: {best_score:.4f}")
                else:
                    no_improvement_count += 1
                
                # Kiểm tra early stopping
                if no_improvement_count >= patience:
                    logger.info(f"Early stopping after {trial+1} trials without improvement")
                    break
            
            except Exception as e:
                logger.error(f"Error in trial {trial+1}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        # Lưu kết quả tốt nhất
        if best_hyperparameters is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Lưu hyperparameters
            hyperparams_path = os.path.join(self.results_dir, 
                                         f"{self.symbol}_{self.timeframe}_hyperparameters_{timestamp}.json")
            
            with open(hyperparams_path, 'w') as f:
                # Thêm thông tin bổ sung
                result = {
                    'symbol': self.symbol,
                    'timeframe': self.timeframe,
                    'hyperparameters': best_hyperparameters,
                    'metrics': best_metrics,
                    'features_used': used_features,
                    'timestamp': timestamp
                }
                
                json.dump(result, f, indent=2, default=str)
            
            # Lưu mô hình
            model_path = os.path.join(self.models_dir, f"{self.symbol}_{self.timeframe}_model.h5")
            best_model.save(model_path)
            
            # Lưu scalers
            price_scaler_path = os.path.join(self.scalers_dir, f"{self.symbol}_{self.timeframe}_price_scaler.pkl")
            feature_scaler_path = os.path.join(self.scalers_dir, f"{self.symbol}_{self.timeframe}_feature_scaler.pkl")
            
            joblib.dump(price_scaler, price_scaler_path)
            joblib.dump(feature_scaler, feature_scaler_path)
            
            logger.info(f"Đã lưu mô hình và kết quả tối ưu siêu tham số cho {self.symbol} - {self.timeframe}")
            
            return result
        else:
            logger.error(f"Không tìm thấy mô hình tốt cho {self.symbol} - {self.timeframe}")
            return None

class ModelEvolutionManager:
    """Lớp quản lý tiến hóa của tất cả các mô hình"""
    
    def __init__(self, config_path="../config/system_config.json"):
        """
        Khởi tạo ModelEvolutionManager
        
        Args:
            config_path (str): Đường dẫn đến file cấu hình
        """
        # Đọc cấu hình
        self.config_path = config_path
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Lấy cấu hình tiến hóa
        self.evolution_config = self.config['evolution']
        
        # Lấy danh sách cổ phiếu
        stocks_config_path = os.path.join(os.path.dirname(self.config_path), "stocks.json")
        with open(stocks_config_path, 'r') as f:
            stocks_config = json.load(f)
        
        self.stocks = [stock['symbol'] for stock in stocks_config['stocks'] if stock['enabled']]
        
        # Đường dẫn lưu trữ phiên bản
        self.versions_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "BondZiA_versions")
        os.makedirs(self.versions_dir, exist_ok=True)
        
        # Đường dẫn lưu trữ kết quả tiến hóa
        self.evolution_results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                               "evolution/results")
        os.makedirs(self.evolution_results_dir, exist_ok=True)
        
        # Khung thời gian
        self.timeframes = ['intraday', 'five_day', 'monthly']
        
        # Số lượng mô hình đã tiến hóa
        self.evolved_models = 0
        
        logger.info(f"Khởi tạo ModelEvolutionManager với {len(self.stocks)} cổ phiếu")
    
    def evolve_model(self, symbol, timeframe, data):
        """
        Tiến hóa một mô hình cụ thể
        
        Args:
            symbol (str): Mã cổ phiếu
            timeframe (str): Khung thời gian
            data (DataFrame): Dữ liệu huấn luyện
            
        Returns:
            dict: Kết quả tiến hóa
        """
        try:
            # Tạo tuner
            tuner = HyperparameterTuner(symbol, timeframe, config_path=self.config_path)
            
            # Thực hiện tối ưu hóa
            result = tuner.tune(data)
            
            if result:
                self.evolved_models += 1
                logger.info(f"Thành công: Tiến hóa mô hình {symbol} - {timeframe}")
                return result
            else:
                logger.warning(f"Không thành công: Tiến hóa mô hình {symbol} - {timeframe}")
                return None
        except Exception as e:
            logger.error(f"Lỗi khi tiến hóa mô hình {symbol} - {timeframe}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def evolve_all_models(self, data_dict):
        """
        Tiến hóa tất cả các mô hình
        
        Args:
            data_dict (dict): Dictionary chứa dữ liệu cho mỗi cổ phiếu
            
        Returns:
            dict: Kết quả tiến hóa
        """
        start_time = datetime.now()
        logger.info(f"Bắt đầu tiến hóa tất cả các mô hình tại {start_time.isoformat()}")
        
        results = {
            'total_models': len(self.stocks) * len(self.timeframes),
            'successful_evolutions': 0,
            'failed_evolutions': 0,
            'model_results': {},
            'version': self._get_next_version(),
            'start_time': start_time.isoformat(),
            'end_time': None,
            'total_params_changed': 0
        }
        
        # Tiến hóa từng mô hình
        for symbol in self.stocks:
            results['model_results'][symbol] = {}
            
            for timeframe in self.timeframes:
                # Kiểm tra xem có dữ liệu không
                if symbol not in data_dict:
                    logger.warning(f"Không có dữ liệu cho {symbol}")
                    results['failed_evolutions'] += 1
                    continue
                
                # Lấy dữ liệu phù hợp với timeframe
                if timeframe == 'intraday':
                    if 'intraday_data' not in data_dict[symbol]:
                        logger.warning(f"Không có dữ liệu intraday cho {symbol}")
                        results['failed_evolutions'] += 1
                        continue
                    data = data_dict[symbol]['intraday_data']
                else:
                    if 'daily_data' not in data_dict[symbol]:
                        logger.warning(f"Không có dữ liệu daily cho {symbol}")
                        results['failed_evolutions'] += 1
                        continue
                    data = data_dict[symbol]['daily_data']
                
                # Tiến hóa mô hình
                logger.info(f"Tiến hóa mô hình {symbol} - {timeframe}")
                model_result = self.evolve_model(symbol, timeframe, data)
                
                if model_result:
                    results['model_results'][symbol][timeframe] = model_result
                    results['successful_evolutions'] += 1
                    
                    # Đếm số lượng tham số đã thay đổi
                    if 'hyperparameters' in model_result:
                        results['total_params_changed'] += len(model_result['hyperparameters'])
                else:
                    results['failed_evolutions'] += 1
        
        # Cập nhật thời gian kết thúc
        end_time = datetime.now()
        results['end_time'] = end_time.isoformat()
        results['elapsed_time'] = (end_time - start_time).total_seconds()
        
        # Lưu kết quả
        self._save_evolution_results(results)
        
        logger.info(f"Hoàn thành tiến hóa {results['successful_evolutions']}/{results['total_models']} mô hình")
        logger.info(f"Tổng số tham số đã thay đổi: {results['total_params_changed']}")
        logger.info(f"Thời gian tiến hóa: {results['elapsed_time']} giây")
        
        return results
    
    def _get_next_version(self):
        """
        Lấy phiên bản tiếp theo
        
        Returns:
            str: Phiên bản tiếp theo
        """
        # Tìm phiên bản hiện tại
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            current_version = config['system']['version']
            
            # Phân tích phiên bản
            parts = current_version.split('.')
            major, minor, patch = map(int, parts)
            
            # Tăng số phiên bản patch
            patch += 1
            
            # Trả về phiên bản mới
            return f"{major}.{minor}.{patch}"
        except Exception as e:
            logger.error(f"Lỗi khi lấy phiên bản tiếp theo: {str(e)}")
            # Mặc định 1.0.0
            return "1.0.0"
    
    def _save_evolution_results(self, results):
        """
        Lưu kết quả tiến hóa
        
        Args:
            results (dict): Kết quả tiến hóa
        """
        try:
            # Tạo thư mục phiên bản mới
            version = results['version']
            version_dir = os.path.join(self.versions_dir, f"BondZiA_v{version}")
            os.makedirs(version_dir, exist_ok=True)
            
            # Lưu kết quả
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.evolution_results_dir, f"evolution_results_{timestamp}.json")
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Cập nhật cấu hình với phiên bản mới
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            config['system']['version'] = version
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Đã lưu kết quả tiến hóa vào {results_path}")
            logger.info(f"Đã cập nhật phiên bản hệ thống lên {version}")
            
            # Sao chép các mô hình vào thư mục phiên bản
            for timeframe in self.timeframes:
                source_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        f"models/{timeframe}")
                dest_dir = os.path.join(version_dir, f"models/{timeframe}")
                
                if os.path.exists(source_dir):
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    # Sao chép các file
                    for file in os.listdir(source_dir):
                        source_file = os.path.join(source_dir, file)
                        dest_file = os.path.join(dest_dir, file)
                        
                        if os.path.isfile(source_file):
                            import shutil
                            shutil.copy2(source_file, dest_file)
            
            logger.info(f"Đã sao chép các mô hình vào thư mục phiên bản {version_dir}")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu kết quả tiến hóa: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def get_evolution_improvements(self, results):
        """
        Lấy thông tin cải thiện từ kết quả tiến hóa
        
        Args:
            results (dict): Kết quả tiến hóa
            
        Returns:
            dict: Thông tin cải thiện
        """
        improvements = {}
        
        for symbol in results['model_results']:
            improvements[symbol] = {}
            
            for timeframe in results['model_results'][symbol]:
                model_result = results['model_results'][symbol][timeframe]
                
                if 'metrics' in model_result:
                    # Đọc các hiệu suất trước đó
                    prev_metrics = self._get_previous_metrics(symbol, timeframe)
                    
                    if prev_metrics:
                        # Tính % cải thiện
                        current_rmse = model_result['metrics']['rmse']
                        prev_rmse = prev_metrics['rmse']
                        
                        improvement = ((prev_rmse - current_rmse) / prev_rmse) * 100
                        
                        improvements[symbol][timeframe] = improvement
                    else:
                        # Không có dữ liệu trước đó
                        improvements[symbol][timeframe] = 0.0
        
        return improvements
    
    def _get_previous_metrics(self, symbol, timeframe):
        """
        Lấy các hiệu suất trước đó
        
        Args:
            symbol (str): Mã cổ phiếu
            timeframe (str): Khung thời gian
            
        Returns:
            dict: Các hiệu suất trước đó hoặc None nếu không có
        """
        try:
            # Tìm tất cả file kết quả
            result_files = []
            
            for file in os.listdir(self.evolution_results_dir):
                if file.startswith("evolution_results_") and file.endswith(".json"):
                    result_files.append(os.path.join(self.evolution_results_dir, file))
            
            # Sắp xếp theo thời gian sửa đổi (mới nhất trước)
            result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Tìm file gần nhất chứa thông tin mô hình
            for file_path in result_files:
                with open(file_path, 'r') as f:
                    results = json.load(f)
                
                if 'model_results' in results and symbol in results['model_results']:
                    if timeframe in results['model_results'][symbol]:
                        model_result = results['model_results'][symbol][timeframe]
                        
                        if 'metrics' in model_result:
                            return model_result['metrics']
            
            return None
        except Exception as e:
            logger.error(f"Lỗi khi lấy hiệu suất trước đó: {str(e)}")
            return None

if __name__ == "__main__":
    # Test module
    logger.info("Kiểm tra module HyperparameterTuner")
    
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
    
    # Test HyperparameterTuner
    tuner = HyperparameterTuner('AAPL', 'intraday')
    
    # Thử với số lần thử nhỏ
    result = tuner.tune(df, trials=3)
    
    if result:
        logger.info(f"Kết quả tối ưu siêu tham số: {result}")
    
    # Test ModelEvolutionManager
    # Chuẩn bị dữ liệu giả
    data_dict = {
        'AAPL': {
            'intraday_data': df,
            'daily_data': df
        }
    }
    
    manager = ModelEvolutionManager()
    evolution_results = manager.evolve_model('AAPL', 'intraday', df)
    
    if evolution_results:
        logger.info(f"Kết quả tiến hóa: {evolution_results}")
