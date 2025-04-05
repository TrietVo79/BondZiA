import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input, Attention, Bidirectional
from tensorflow.keras.models import Sequential, Model
import traceback

# Thêm thư mục gốc vào PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from utils.logger_config import logger
from models.base_predictor import BasePredictor

class IntradayPredictor(BasePredictor):
    """Lớp dự đoán giá intraday (trong ngày)"""
    
    def __init__(self, symbol, config_path="../config/system_config.json"):
        """
        Khởi tạo IntradayPredictor
        
        Args:
            symbol (str): Mã cổ phiếu
            config_path (str): Đường dẫn đến file cấu hình
        """
        super().__init__(symbol, "intraday", config_path)
        
        logger.info(f"Khởi tạo IntradayPredictor cho {symbol}")
    
    def _build_model(self, input_shape):
        """
        Xây dựng mô hình dự đoán intraday
        
        Args:
            input_shape (tuple): Kích thước đầu vào
            
        Returns:
            Model: Mô hình Keras
        """
        try:
            # Input layer
            input_layer = Input(shape=input_shape)
            
            # Bidirectional LSTM để nắm bắt cả xu hướng trước và sau
            x = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
            x = Dropout(0.3)(x)
            
            # Thêm LSTM Layer với cơ chế Attention
            # (triển khai đơn giản của Attention)
            lstm_out = LSTM(32, return_sequences=True)(x)
            
            # Tính trọng số attention
            attention = Dense(1, activation='tanh')(lstm_out)
            attention = tf.keras.layers.Reshape((-1,))(attention)
            attention_weights = tf.keras.layers.Softmax()(attention)
            
            # Nhân trọng số với đầu ra LSTM
            context_vector = tf.keras.layers.Dot(axes=[1, 1])([lstm_out, tf.expand_dims(attention_weights, -1)])
            context_vector = tf.keras.layers.Reshape((-1,))(context_vector)
            
            # Kết hợp lớp cuối
            x = Dropout(0.2)(context_vector)
            x = Dense(16, activation='relu')(x)
            output = Dense(1)(x)
            
            # Tạo model
            model = Model(inputs=input_layer, outputs=output)
            
            # Biên dịch model
            model.compile(optimizer='adam', loss='mse')
            
            return model
        
        except Exception as e:
            logger.error(f"Lỗi khi xây dựng mô hình intraday: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fallback to basic model if there's an error
            return super()._build_model(input_shape)
    
    def predict(self, data):
        """
        Dự đoán giá cổ phiếu intraday
        
        Args:
            data (DataFrame): Dữ liệu đầu vào
            
        Returns:
            dict: Kết quả dự đoán
        """
        try:
            prediction = super().predict(data)
            
            if prediction is None:
                return None
            
            # Thông tin bổ sung cho dự đoán intraday
            prediction['timeframe'] = 'intraday'
            prediction['target_time'] = 'end of trading day'
            
            return prediction
        
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán intraday: {str(e)}")
            logger.error(traceback.format_exc())
            return None

class FiveDayPredictor(BasePredictor):
    """Lớp dự đoán giá 5 ngày"""
    
    def __init__(self, symbol, config_path="../config/system_config.json"):
        """
        Khởi tạo FiveDayPredictor
        
        Args:
            symbol (str): Mã cổ phiếu
            config_path (str): Đường dẫn đến file cấu hình
        """
        super().__init__(symbol, "five_day", config_path)
        
        logger.info(f"Khởi tạo FiveDayPredictor cho {symbol}")
    
    def _build_model(self, input_shape):
        """
        Xây dựng mô hình dự đoán 5 ngày
        
        Args:
            input_shape (tuple): Kích thước đầu vào
            
        Returns:
            Model: Mô hình Keras
        """
        try:
            # Model với cơ chế stacked LSTM và Residual connections
            
            # Input layer
            input_layer = Input(shape=input_shape)
            
            # First LSTM layer
            lstm1 = LSTM(64, return_sequences=True)(input_layer)
            lstm1 = Dropout(0.3)(lstm1)
            
            # Second LSTM layer with residual connection
            lstm2 = LSTM(64, return_sequences=True)(lstm1)
            lstm2 = Dropout(0.3)(lstm2)
            
            # Residual connection
            # Trước khi kết nối, cần đảm bảo kích thước phù hợp
            # Sử dụng một projection layer nếu cần
            projection = Dense(64)(lstm1)  # Projection để khớp kích thước
            
            # Residual connection
            residual = tf.keras.layers.Add()([lstm2, projection])
            
            # Third LSTM layer
            lstm3 = LSTM(32)(residual)
            lstm3 = Dropout(0.2)(lstm3)
            
            # Dense layers
            dense1 = Dense(16, activation='relu')(lstm3)
            
            # Output layer
            output = Dense(1)(dense1)
            
            # Create model
            model = Model(inputs=input_layer, outputs=output)
            
            # Compile model
            model.compile(optimizer='adam', loss='mse')
            
            return model
        
        except Exception as e:
            logger.error(f"Lỗi khi xây dựng mô hình 5 ngày: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fallback to basic model if there's an error
            return super()._build_model(input_shape)
    
    def predict(self, data):
        """
        Dự đoán giá cổ phiếu sau 5 ngày
        
        Args:
            data (DataFrame): Dữ liệu đầu vào
            
        Returns:
            dict: Kết quả dự đoán
        """
        try:
            prediction = super().predict(data)
            
            if prediction is None:
                return None
            
            # Thông tin bổ sung cho dự đoán 5 ngày
            prediction['timeframe'] = 'five_day'
            prediction['target_time'] = '5 trading days'
            
            return prediction
        
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán 5 ngày: {str(e)}")
            logger.error(traceback.format_exc())
            return None

class MonthlyPredictor(BasePredictor):
    """Lớp dự đoán giá 1 tháng"""
    
    def __init__(self, symbol, config_path="../config/system_config.json"):
        """
        Khởi tạo MonthlyPredictor
        
        Args:
            symbol (str): Mã cổ phiếu
            config_path (str): Đường dẫn đến file cấu hình
        """
        super().__init__(symbol, "monthly", config_path)
        
        logger.info(f"Khởi tạo MonthlyPredictor cho {symbol}")
    
    def _build_model(self, input_shape):
        """
        Xây dựng mô hình dự đoán 1 tháng
        
        Args:
            input_shape (tuple): Kích thước đầu vào
            
        Returns:
            Model: Mô hình Keras
        """
        try:
            # Model kết hợp GRU và LSTM
            
            # Input layer
            input_layer = Input(shape=input_shape)
            
            # GRU layer
            gru = GRU(64, return_sequences=True)(input_layer)
            gru = Dropout(0.3)(gru)
            
            # LSTM layer
            lstm = LSTM(64, return_sequences=True)(gru)
            lstm = Dropout(0.3)(lstm)
            
            # Tính toán attention scores
            attention_dense = Dense(1)(lstm)
            attention_weights = tf.keras.layers.Softmax(axis=1)(attention_dense)
            
            # Apply attention weights
            context_vector = tf.keras.layers.Multiply()([lstm, attention_weights])
            context_vector = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
            
            # Dense layers
            dense1 = Dense(32, activation='relu')(context_vector)
            dense1 = Dropout(0.2)(dense1)
            dense2 = Dense(16, activation='relu')(dense1)
            
            # Output layer
            output = Dense(1)(dense2)
            
            # Create model
            model = Model(inputs=input_layer, outputs=output)
            
            # Compile model
            model.compile(optimizer='adam', loss='mse')
            
            return model
        
        except Exception as e:
            logger.error(f"Lỗi khi xây dựng mô hình 1 tháng: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fallback to basic model if there's an error
            return super()._build_model(input_shape)
    
    def predict(self, data):
        """
        Dự đoán giá cổ phiếu sau 1 tháng
        
        Args:
            data (DataFrame): Dữ liệu đầu vào
            
        Returns:
            dict: Kết quả dự đoán
        """
        try:
            prediction = super().predict(data)
            
            if prediction is None:
                return None
            
            # Thông tin bổ sung cho dự đoán 1 tháng
            prediction['timeframe'] = 'monthly'
            prediction['target_time'] = '30 calendar days'
            
            return prediction
        
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán 1 tháng: {str(e)}")
            logger.error(traceback.format_exc())
            return None