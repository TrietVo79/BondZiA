import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime, timedelta
from utils.logger_config import logger
import traceback

class StockVisualizer:
    """Lớp tạo biểu đồ trực quan cho dữ liệu cổ phiếu và dự đoán"""
    
    def __init__(self, config_path="../config/system_config.json"):
        """
        Khởi tạo Visualizer với cấu hình
        
        Args:
            config_path (str): Đường dẫn đến file cấu hình
        """
        # Đọc cấu hình
        self.config_path = config_path
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Thiết lập cấu hình visualization
        self.viz_config = self.config['visualization']
        self.theme = self.viz_config['theme']
        self.up_color = self.viz_config['up_color']
        self.down_color = self.viz_config['down_color']
        self.neutral_color = self.viz_config['neutral_color']
        
        # Thiết lập style matplotlib theo theme
        if self.theme == 'dark':
            plt.style.use('dark_background')
            self.bg_color = '#121212'
            self.text_color = 'white'
            self.grid_color = '#333333'
        else:
            plt.style.use('default')
            self.bg_color = 'white'
            self.text_color = 'black'
            self.grid_color = '#dddddd'
        
        # Đảm bảo thư mục charts tồn tại
        self.charts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     "charts")
        os.makedirs(self.charts_dir, exist_ok=True)
        
        logger.info(f"StockVisualizer khởi tạo với theme: {self.theme}")
    
    def _get_color(self, direction):
        """Lấy màu tương ứng với hướng giá"""
        if direction == 'up':
            return self.up_color
        elif direction == 'down':
            return self.down_color
        else:
            return self.neutral_color
    
    def create_price_prediction_chart(self, symbol, historical_data, prediction_data, save=True):
        """
        Tạo biểu đồ dự đoán giá cổ phiếu
        
        Args:
            symbol (str): Mã cổ phiếu
            historical_data (DataFrame): Dữ liệu lịch sử
            prediction_data (dict): Dữ liệu dự đoán
            save (bool): Lưu biểu đồ vào file
            
        Returns:
            str: Đường dẫn đến file biểu đồ (nếu save=True)
        """
        try:
            # Tạo biểu đồ Plotly
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Thiết lập theme
            if self.theme == 'dark':
                template = 'plotly_dark'
                paper_bgcolor = '#121212'
                plot_bgcolor = '#121212'
            else:
                template = 'plotly_white'
                paper_bgcolor = 'white'
                plot_bgcolor = 'white'
            
            # Vẽ dữ liệu lịch sử
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=historical_data['close'],
                    mode='lines',
                    name='Giá đóng cửa',
                    line=dict(color='#2962FF', width=2)
                )
            )
            
            # Thêm các dữ liệu khối lượng
            fig.add_trace(
                go.Bar(
                    x=historical_data.index,
                    y=historical_data['volume'],
                    name='Khối lượng',
                    marker=dict(color='rgba(100, 100, 100, 0.5)'),
                    opacity=0.5
                ),
                secondary_y=True
            )
            
            # Thêm các dự đoán
            last_date = historical_data.index[-1]
            last_price = historical_data['close'].iloc[-1]
            
            prediction_dates = []
            prediction_prices = []
            prediction_texts = []
            prediction_colors = []
            
            # Lấy thời gian hiện tại để đảm bảo dự đoán đúng thời điểm
            current_date = datetime.now().date()
            
            # Dự đoán intraday
            if 'intraday' in prediction_data:
                intraday = prediction_data['intraday']
                # Chỉ thêm dự đoán intraday nếu có hướng và confidence
                if 'direction' in intraday and 'confidence' in intraday:
                    # Sử dụng cùng ngày hiện tại, thêm 4 giờ như một ước lượng hợp lý
                    intraday_date = pd.Timestamp(current_date) + pd.Timedelta(hours=4)
                    prediction_dates.append(intraday_date)
                    
                    # Lấy giá dự đoán, ưu tiên predicted_price, sau đó đến price, cuối cùng là predicted_value
                    price_value = intraday.get('predicted_price', intraday.get('price', intraday.get('predicted_value', 0)))
                    prediction_prices.append(price_value)
                    
                    direction_icon = "🔼" if intraday['direction'] == 'up' else "🔽" if intraday['direction'] == 'down' else "➡️"
                    reason_text = intraday.get('reason', "Dự đoán dựa trên mô hình ML")
                                        
                    prediction_texts.append(f"{direction_icon} Intraday: ${price_value:.2f}<br>Độ tin cậy: {intraday['confidence']:.1f}%<br>Lý do: {reason_text}")
                    prediction_colors.append(self._get_color(intraday['direction']))
            
            # Dự đoán 5 ngày
            if 'five_day' in prediction_data:
                five_day = prediction_data['five_day']
                # Chỉ thêm dự đoán 5 ngày nếu có hướng và confidence
                if 'direction' in five_day and 'confidence' in five_day:
                    # Thêm 5 ngày kể từ ngày hiện tại
                    five_day_date = pd.Timestamp(current_date) + pd.Timedelta(days=5)
                    prediction_dates.append(five_day_date)
                    
                    # Lấy giá dự đoán, ưu tiên predicted_price, sau đó đến price, cuối cùng là predicted_value
                    price_value = five_day.get('predicted_price', five_day.get('price', five_day.get('predicted_value', 0)))
                    prediction_prices.append(price_value)
                    
                    direction_icon = "🔼" if five_day['direction'] == 'up' else "🔽" if five_day['direction'] == 'down' else "➡️"
                    reason_text = five_day.get('reason', "Dự đoán dựa trên mô hình ML")
                                        
                    prediction_texts.append(f"{direction_icon} 5 ngày: ${price_value:.2f}<br>Độ tin cậy: {five_day['confidence']:.1f}%<br>Lý do: {reason_text}")
                    prediction_colors.append(self._get_color(five_day['direction']))
            
            # Dự đoán 1 tháng
            if 'monthly' in prediction_data:
                monthly = prediction_data['monthly']
                # Chỉ thêm dự đoán tháng nếu có hướng và confidence
                if 'direction' in monthly and 'confidence' in monthly:
                    # Thêm 30 ngày kể từ ngày hiện tại
                    monthly_date = pd.Timestamp(current_date) + pd.Timedelta(days=30)
                    prediction_dates.append(monthly_date)
                    
                    # Lấy giá dự đoán, ưu tiên predicted_price, sau đó đến price, cuối cùng là predicted_value
                    price_value = monthly.get('predicted_price', monthly.get('price', monthly.get('predicted_value', 0)))
                    prediction_prices.append(price_value)
                    
                    direction_icon = "🔼" if monthly['direction'] == 'up' else "🔽" if monthly['direction'] == 'down' else "➡️"
                    reason_text = monthly.get('reason', "Dự đoán dựa trên mô hình ML")
                                        
                    prediction_texts.append(f"{direction_icon} 1 tháng: ${price_value:.2f}<br>Độ tin cậy: {monthly['confidence']:.1f}%<br>Lý do: {reason_text}")
                    prediction_colors.append(self._get_color(monthly['direction']))
            
            # Vẽ các đường dự đoán
            for i in range(len(prediction_dates)):
                # Vẽ đường dự đoán
                fig.add_trace(
                    go.Scatter(
                        x=[last_date, prediction_dates[i]],
                        y=[last_price, prediction_prices[i]],
                        mode='lines',
                        line=dict(color=prediction_colors[i], dash='dash', width=2),
                        showlegend=False
                    )
                )
                
                # Vẽ điểm dự đoán
                fig.add_trace(
                    go.Scatter(
                        x=[prediction_dates[i]],
                        y=[prediction_prices[i]],
                        mode='markers',
                        marker=dict(color=prediction_colors[i], size=12, symbol='circle'),
                        name=f"Dự đoán {prediction_dates[i].strftime('%d/%m/%Y')}",
                        text=prediction_texts[i],
                        hoverinfo='text'
                    )
                )
            
            # Cấu hình layout
            current_date_str = datetime.now().strftime("%d/%m/%Y")
            fig.update_layout(
                title=f"Dự đoán giá cổ phiếu {symbol} (Ngày: {current_date_str})",
                template=template,
                paper_bgcolor=paper_bgcolor,
                plot_bgcolor=plot_bgcolor,
                hovermode="closest",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(
                    title="Ngày",
                    showgrid=True,
                    gridcolor=self.grid_color
                ),
                yaxis=dict(
                    title="Giá ($)",
                    showgrid=True,
                    gridcolor=self.grid_color
                ),
                yaxis2=dict(
                    title="Khối lượng",
                    showgrid=False
                ),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Tạo tooltip
            fig.update_traces(
                hovertemplate='%{text}',
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                )
            )
            
            # Lưu biểu đồ nếu cần
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{symbol}_prediction_{timestamp}.html"
                filepath = os.path.join(self.charts_dir, filename)
                
                # Lưu dưới dạng HTML tương tác
                fig.write_html(filepath)
                
                # Lưu dưới dạng ảnh tĩnh
                img_filename = f"{symbol}_prediction_{timestamp}.png"
                img_filepath = os.path.join(self.charts_dir, img_filename)
                fig.write_image(img_filepath, width=1200, height=800)
                
                logger.info(f"Đã lưu biểu đồ dự đoán {symbol} tại {img_filepath}")
                return img_filepath
            else:
                return fig
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ dự đoán cho {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def create_comparison_chart(self, symbol, predictions, actuals, timeframe='intraday', save=True):
        """
        Tạo biểu đồ so sánh giữa dự đoán và thực tế
        
        Args:
            symbol (str): Mã cổ phiếu
            predictions (DataFrame): Dữ liệu dự đoán
            actuals (DataFrame): Dữ liệu thực tế
            timeframe (str): Khung thời gian ('intraday', 'five_day', 'monthly')
            save (bool): Lưu biểu đồ vào file
            
        Returns:
            str: Đường dẫn đến file biểu đồ (nếu save=True)
        """
        try:
            # Tạo biểu đồ Plotly
            fig = go.Figure()
            
            # Thiết lập theme
            if self.theme == 'dark':
                template = 'plotly_dark'
                paper_bgcolor = '#121212'
                plot_bgcolor = '#121212'
            else:
                template = 'plotly_white'
                paper_bgcolor = 'white'
                plot_bgcolor = 'white'
            
            # Vẽ dữ liệu thực tế
            fig.add_trace(
                go.Scatter(
                    x=actuals.index,
                    y=actuals['close'],
                    mode='lines',
                    name='Giá thực tế',
                    line=dict(color='#2962FF', width=2)
                )
            )
            
            # Vẽ dữ liệu dự đoán
            fig.add_trace(
                go.Scatter(
                    x=predictions.index,
                    y=predictions['predicted_price'],
                    mode='lines',
                    name='Giá dự đoán',
                    line=dict(color='#FF6D00', width=2, dash='dash')
                )
            )
            
            # Tính toán sai số
            mape = np.mean(np.abs((actuals['close'] - predictions['predicted_price']) / actuals['close'])) * 100
            rmse = np.sqrt(np.mean((actuals['close'] - predictions['predicted_price']) ** 2))
            
            # Thêm thông tin sai số
            timeframe_names = {
                'intraday': 'Intraday',
                'five_day': '5 ngày',
                'monthly': '1 tháng'
            }
            
            # Cấu hình layout
            fig.update_layout(
                title=f"So sánh dự đoán và thực tế {symbol} - {timeframe_names.get(timeframe, timeframe)}",
                template=template,
                paper_bgcolor=paper_bgcolor,
                plot_bgcolor=plot_bgcolor,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(
                    title="Ngày",
                    showgrid=True,
                    gridcolor=self.grid_color
                ),
                yaxis=dict(
                    title="Giá ($)",
                    showgrid=True,
                    gridcolor=self.grid_color
                ),
                annotations=[
                    dict(
                        x=0.01,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text=f"MAPE: {mape:.2f}%<br>RMSE: ${rmse:.2f}",
                        showarrow=False,
                        font=dict(size=14),
                        bgcolor="rgba(255, 255, 255, 0.7)",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4
                    )
                ],
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Lưu biểu đồ nếu cần
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{symbol}_comparison_{timeframe}_{timestamp}.html"
                filepath = os.path.join(self.charts_dir, filename)
                
                # Lưu dưới dạng HTML tương tác
                fig.write_html(filepath)
                
                # Lưu dưới dạng ảnh tĩnh
                img_filename = f"{symbol}_comparison_{timeframe}_{timestamp}.png"
                img_filepath = os.path.join(self.charts_dir, img_filename)
                fig.write_image(img_filepath, width=1200, height=800)
                
                logger.info(f"Đã lưu biểu đồ so sánh {symbol} tại {img_filepath}")
                return img_filepath
            else:
                return fig
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ so sánh cho {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def create_model_performance_chart(self, performance_data, save=True):
        """
        Tạo biểu đồ hiệu suất của mô hình theo thời gian
        
        Args:
            performance_data (DataFrame): Dữ liệu hiệu suất qua các phiên bản
            save (bool): Lưu biểu đồ vào file
            
        Returns:
            str: Đường dẫn đến file biểu đồ (nếu save=True)
        """
        try:
            # Tạo biểu đồ Plotly
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Thiết lập theme
            if self.theme == 'dark':
                template = 'plotly_dark'
                paper_bgcolor = '#121212'
                plot_bgcolor = '#121212'
            else:
                template = 'plotly_white'
                paper_bgcolor = 'white'
                plot_bgcolor = 'white'
            
            # Vẽ RMSE
            fig.add_trace(
                go.Scatter(
                    x=performance_data.index,
                    y=performance_data['rmse'],
                    mode='lines+markers',
                    name='RMSE',
                    line=dict(color='#FF6D00', width=2)
                )
            )
            
            # Vẽ Accuracy
            fig.add_trace(
                go.Scatter(
                    x=performance_data.index,
                    y=performance_data['accuracy'],
                    mode='lines+markers',
                    name='Độ chính xác (%)',
                    line=dict(color='#2962FF', width=2)
                ),
                secondary_y=True
            )
            
            # Cấu hình layout
            fig.update_layout(
                title="Hiệu suất mô hình BondZiA qua các phiên bản",
                template=template,
                paper_bgcolor=paper_bgcolor,
                plot_bgcolor=plot_bgcolor,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(
                    title="Phiên bản",
                    showgrid=True,
                    gridcolor=self.grid_color
                ),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Cấu hình trục y
            fig.update_yaxes(
                title_text="RMSE ($)",
                showgrid=True,
                gridcolor=self.grid_color,
                secondary_y=False
            )
            
            fig.update_yaxes(
                title_text="Độ chính xác (%)",
                showgrid=False,
                secondary_y=True
            )
            
            # Lưu biểu đồ nếu cần
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"model_performance_{timestamp}.html"
                filepath = os.path.join(self.charts_dir, filename)
                
                # Lưu dưới dạng HTML tương tác
                fig.write_html(filepath)
                
                # Lưu dưới dạng ảnh tĩnh
                img_filename = f"model_performance_{timestamp}.png"
                img_filepath = os.path.join(self.charts_dir, img_filename)
                fig.write_image(img_filepath, width=1200, height=800)
                
                logger.info(f"Đã lưu biểu đồ hiệu suất mô hình tại {img_filepath}")
                return img_filepath
            else:
                return fig
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ hiệu suất mô hình: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def create_stock_dashboard(self, symbol, historical_data, prediction_data, technical_indicators=True, save=True):
        """
        Tạo bảng điều khiển tổng quan cho một cổ phiếu
        
        Args:
            symbol (str): Mã cổ phiếu
            historical_data (DataFrame): Dữ liệu lịch sử
            prediction_data (dict): Dữ liệu dự đoán
            technical_indicators (bool): Hiển thị chỉ báo kỹ thuật
            save (bool): Lưu biểu đồ vào file
            
        Returns:
            str: Đường dẫn đến file biểu đồ (nếu save=True)
        """
        try:
            # Tạo biểu đồ Plotly
            rows = 3 if technical_indicators else 2
            fig = make_subplots(
                rows=rows, 
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=("Giá & Dự đoán", "Khối lượng", "Chỉ báo kỹ thuật") if technical_indicators else ("Giá & Dự đoán", "Khối lượng"),
                row_heights=[0.6, 0.2, 0.2] if technical_indicators else [0.7, 0.3]
            )
            
            # Thiết lập theme
            if self.theme == 'dark':
                template = 'plotly_dark'
                paper_bgcolor = '#121212'
                plot_bgcolor = '#121212'
            else:
                template = 'plotly_white'
                paper_bgcolor = 'white'
                plot_bgcolor = 'white'
            
            # Vẽ candlestick
            fig.add_trace(
                go.Candlestick(
                    x=historical_data.index,
                    open=historical_data['open'],
                    high=historical_data['high'],
                    low=historical_data['low'],
                    close=historical_data['close'],
                    name="OHLC"
                ),
                row=1, col=1
            )
            
            # Thêm các dự đoán
            last_date = historical_data.index[-1]
            last_price = historical_data['close'].iloc[-1]
            
            prediction_dates = []
            prediction_prices = []
            prediction_texts = []
            prediction_colors = []
            
            # Lấy thời gian hiện tại để đảm bảo dự đoán đúng thời điểm
            current_date = datetime.now().date()
            
            # Dự đoán intraday
            if 'intraday' in prediction_data:
                intraday = prediction_data['intraday']
                # Chỉ thêm dự đoán intraday nếu có hướng và confidence
                if 'direction' in intraday and 'confidence' in intraday:
                    # Sử dụng cùng ngày hiện tại, thêm 4 giờ như một ước lượng hợp lý
                    intraday_date = pd.Timestamp(current_date) + pd.Timedelta(hours=4)
                    prediction_dates.append(intraday_date)
                    
                    # Lấy giá dự đoán, ưu tiên predicted_price, sau đó đến price, cuối cùng là predicted_value
                    price_value = intraday.get('predicted_price', intraday.get('price', intraday.get('predicted_value', 0)))
                    prediction_prices.append(price_value)
                    
                    direction_icon = "🔼" if intraday['direction'] == 'up' else "🔽" if intraday['direction'] == 'down' else "➡️"
                    reason_text = intraday.get('reason', "Dự đoán dựa trên mô hình ML")
                    if intraday['direction'] == 'up':
                        reason_text = "Giá có xu hướng tăng trong thời gian gần đây"
                    elif intraday['direction'] == 'down':
                        reason_text = "Giá có xu hướng giảm gần đây"
                    
                    prediction_texts.append(f"{direction_icon} Intraday: ${price_value:.2f}<br>Độ tin cậy: {intraday['confidence']:.1f}%<br>Lý do: {reason_text}")
                    prediction_colors.append(self._get_color(intraday['direction']))
            
            # Dự đoán 5 ngày
            if 'five_day' in prediction_data:
                five_day = prediction_data['five_day']
                # Chỉ thêm dự đoán 5 ngày nếu có hướng và confidence
                if 'direction' in five_day and 'confidence' in five_day:
                    five_day_date = pd.Timestamp(current_date) + pd.Timedelta(days=5)
                    prediction_dates.append(five_day_date)
                    
                    # Lấy giá dự đoán, ưu tiên predicted_price, sau đó đến price, cuối cùng là predicted_value
                    price_value = five_day.get('predicted_price', five_day.get('price', five_day.get('predicted_value', 0)))
                    prediction_prices.append(price_value)
                    
                    direction_icon = "🔼" if five_day['direction'] == 'up' else "🔽" if five_day['direction'] == 'down' else "➡️"
                    reason_text = five_day.get('reason', "Dự đoán dựa trên mô hình ML")
                    if five_day['direction'] == 'up':
                        reason_text = "Giá có xu hướng tăng trong chuỗi ngày gần đây"
                    elif five_day['direction'] == 'down':
                        reason_text = "Giá có xu hướng giảm trong chuỗi ngày gần đây"
                    
                    prediction_texts.append(f"{direction_icon} 5 ngày: ${price_value:.2f}<br>Độ tin cậy: {five_day['confidence']:.1f}%<br>Lý do: {reason_text}")
                    prediction_colors.append(self._get_color(five_day['direction']))
            
            # Dự đoán 1 tháng
            if 'monthly' in prediction_data:
                monthly = prediction_data['monthly']
                # Chỉ thêm dự đoán tháng nếu có hướng và confidence
                if 'direction' in monthly and 'confidence' in monthly:
                    monthly_date = pd.Timestamp(current_date) + pd.Timedelta(days=30)
                    prediction_dates.append(monthly_date)
                    
                    # Lấy giá dự đoán, ưu tiên predicted_price, sau đó đến price, cuối cùng là predicted_value
                    price_value = monthly.get('predicted_price', monthly.get('price', monthly.get('predicted_value', 0)))
                    prediction_prices.append(price_value)
                    
                    direction_icon = "🔼" if monthly['direction'] == 'up' else "🔽" if monthly['direction'] == 'down' else "➡️"
                    reason_text = monthly.get('reason', "Dự đoán dựa trên mô hình ML")
                    if monthly['direction'] == 'up':
                        reason_text = "Phân tích xu hướng dài hạn chỉ báo tăng giá"
                    elif monthly['direction'] == 'down':
                        reason_text = "Phân tích xu hướng dài hạn chỉ báo giảm giá"
                    
                    prediction_texts.append(f"{direction_icon} 1 tháng: ${price_value:.2f}<br>Độ tin cậy: {monthly['confidence']:.1f}%<br>Lý do: {reason_text}")
                    prediction_colors.append(self._get_color(monthly['direction']))
            
            # Vẽ các đường dự đoán
            for i in range(len(prediction_dates)):
                # Vẽ đường dự đoán
                fig.add_trace(
                    go.Scatter(
                        x=[last_date, prediction_dates[i]],
                        y=[last_price, prediction_prices[i]],
                        mode='lines',
                        line=dict(color=prediction_colors[i], dash='dash', width=2),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # Vẽ điểm dự đoán
                fig.add_trace(
                    go.Scatter(
                        x=[prediction_dates[i]],
                        y=[prediction_prices[i]],
                        mode='markers',
                        marker=dict(color=prediction_colors[i], size=12, symbol='circle'),
                        name=f"Dự đoán {prediction_dates[i].strftime('%d/%m/%Y')}",
                        text=prediction_texts[i],
                        hoverinfo='text'
                    ),
                    row=1, col=1
                )
            
            # Vẽ khối lượng
            fig.add_trace(
                go.Bar(
                    x=historical_data.index,
                    y=historical_data['volume'],
                    name='Khối lượng',
                    marker=dict(color='rgba(100, 100, 100, 0.5)')
                ),
                row=2, col=1
            )
            
            # Vẽ chỉ báo kỹ thuật nếu cần
            if technical_indicators:
                # Vẽ RSI
                if 'rsi_14' in historical_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=historical_data.index,
                            y=historical_data['rsi_14'],
                            mode='lines',
                            name='RSI (14)',
                            line=dict(color='#FF6D00', width=1)
                        ),
                        row=3, col=1
                    )
                    
                    # Thêm đường 70 và 30
                    fig.add_trace(
                        go.Scatter(
                            x=[historical_data.index[0], historical_data.index[-1]],
                            y=[70, 70],
                            mode='lines',
                            line=dict(color='red', width=1, dash='dash'),
                            showlegend=False
                        ),
                        row=3, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[historical_data.index[0], historical_data.index[-1]],
                            y=[30, 30],
                            mode='lines',
                            line=dict(color='green', width=1, dash='dash'),
                            showlegend=False
                        ),
                        row=3, col=1
                    )
                
                # Vẽ MACD nếu có
                if 'macd' in historical_data.columns and 'macd_signal' in historical_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=historical_data.index,
                            y=historical_data['macd'],
                            mode='lines',
                            name='MACD',
                            line=dict(color='#2962FF', width=1)
                        ),
                        row=3, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=historical_data.index,
                            y=historical_data['macd_signal'],
                            mode='lines',
                            name='MACD Signal',
                            line=dict(color='#FF6D00', width=1)
                        ),
                        row=3, col=1
                    )
            
            # Cấu hình layout
            current_date_str = datetime.now().strftime("%d/%m/%Y")
            fig.update_layout(
                title=f"Bảng điều khiển cổ phiếu {symbol} (Ngày: {current_date_str})",
                template=template,
                paper_bgcolor=paper_bgcolor,
                plot_bgcolor=plot_bgcolor,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(
                    rangeslider=dict(visible=False),
                    type="date"
                ),
                margin=dict(l=50, r=50, t=80, b=50),
                height=900 if technical_indicators else 700
            )
            
            # Cấu hình tooltip
            fig.update_traces(
                hovertemplate='%{text}',
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                )
            )
            
            # Lưu biểu đồ nếu cần
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{symbol}_dashboard_{timestamp}.html"
                filepath = os.path.join(self.charts_dir, filename)
                
                # Lưu dưới dạng HTML tương tác
                fig.write_html(filepath)
                
                # Lưu dưới dạng ảnh tĩnh
                img_filename = f"{symbol}_dashboard_{timestamp}.png"
                img_filepath = os.path.join(self.charts_dir, img_filename)
                fig.write_image(img_filepath, width=1200, height=800)
                
                logger.info(f"Đã lưu bảng điều khiển {symbol} tại {img_filepath}")
                return img_filepath
            else:
                return fig
        except Exception as e:
            logger.error(f"Lỗi khi tạo bảng điều khiển cho {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

if __name__ == "__main__":
    # Test module
    logger.info("Kiểm tra module StockVisualizer")
    
    # Tạo dữ liệu giả
    dates = pd.date_range(start='2023-01-01', periods=30)
    data = {
        'open': np.random.normal(100, 5, 30),
        'high': np.random.normal(105, 5, 30),
        'low': np.random.normal(95, 5, 30),
        'close': np.random.normal(100, 5, 30),
        'volume': np.random.normal(1000000, 200000, 30)
    }
    historical_data = pd.DataFrame(data, index=dates)
    
    # Tính chỉ báo kỹ thuật
    historical_data['rsi_14'] = np.random.normal(50, 10, 30)
    historical_data['macd'] = np.random.normal(0, 1, 30)
    historical_data['macd_signal'] = np.random.normal(0, 1, 30)
    
    # Dữ liệu dự đoán
    prediction_data = {
        'intraday': {
            'predicted_price': 102.5,
            'direction': 'up',
            'confidence': 75.3,
            'reason': 'Tin tức tích cực và khối lượng tăng'
        },
        'five_day': {
            'predicted_price': 105.8,
            'direction': 'up',
            'confidence': 68.2,
            'reason': 'Đà tăng giá trong ngành công nghệ'
        },
        'monthly': {
            'predicted_price': 98.2,
            'direction': 'down',
            'confidence': 60.5,
            'reason': 'Dự báo lợi nhuận quý sau thấp hơn'
        }
    }
    
    # Khởi tạo visualizer
    visualizer = StockVisualizer()
    
    # Test tạo biểu đồ dự đoán
    visualizer.create_price_prediction_chart('AAPL', historical_data, prediction_data)
    
    # Test tạo bảng điều khiển
    visualizer.create_stock_dashboard('AAPL', historical_data, prediction_data)       