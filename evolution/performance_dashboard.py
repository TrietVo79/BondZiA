import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import glob

# Lấy đường dẫn tuyệt đối
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # Đi lên một cấp từ thư mục evolution

# Xây dựng đường dẫn tuyệt đối
prediction_history_dir = os.path.join(root_dir, "prediction_history")
data_dir = os.path.join(root_dir, "data/raw")

class PerformanceDashboard:
    def __init__(self, prediction_history_dir=prediction_history_dir, 
                data_dir=data_dir):
        self.prediction_history_dir = prediction_history_dir
        self.data_dir = data_dir
        
        # Kiểm tra thư mục
        if not os.path.exists(prediction_history_dir):
            st.error(f"Thư mục {prediction_history_dir} không tồn tại.")
            st.info(f"Đường dẫn hiện tại: {os.getcwd()}")
        
        if not os.path.exists(data_dir):
            st.error(f"Thư mục {data_dir} không tồn tại.")
    
    def get_symbols(self):
        """Lấy danh sách các cổ phiếu có dự đoán"""
        symbols = set()
        
        for filename in os.listdir(self.prediction_history_dir):
            if '_' in filename:
                symbol = filename.split('_')[0]
                symbols.add(symbol)
        
        return list(symbols)
    
    def load_predictions(self, symbol, days=30):
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
                    st.error(f"Lỗi khi đọc file {file_path}: {str(e)}")
        
        return predictions
    
    def load_actual_prices(self, symbol, days=30):
        """
        Hàm tải giá thực tế từ nhiều nguồn dữ liệu khác nhau.
        Hỗ trợ cả file CSV và JSON.
        """
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
                        print(f"Không thể đọc timestamp từ file {json_file}: {str(e)}")
                        continue
                
                # Format 1: Dạng danh sách symbols
                if isinstance(market_data, list):
                    for item in market_data:
                        if item.get('symbol') == symbol:
                            actual_prices.append({
                                'date': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                'price': item.get('price', item.get('current_price', 0)),
                                'source': 'market_data_json_list'
                            })
                            break
                
                # Format 2: Dạng dictionary với key là symbol
                elif isinstance(market_data, dict):
                    if symbol in market_data:
                        actual_prices.append({
                            'date': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            'price': market_data[symbol].get('price', market_data[symbol].get('current_price', 0)),
                            'source': 'market_data_json_dict'
                        })
                    # Format 3: Dạng dictionary với symbols là một trường
                    elif 'symbols' in market_data and symbol in market_data['symbols']:
                        actual_prices.append({
                            'date': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            'price': market_data['symbols'][symbol].get('price', market_data['symbols'][symbol].get('current_price', 0)),
                            'source': 'market_data_json_symbols'
                        })
                    # Format 4: Dạng dictionary với trường data
                    elif 'data' in market_data:
                        if isinstance(market_data['data'], list):
                            for item in market_data['data']:
                                if item.get('symbol') == symbol:
                                    actual_prices.append({
                                        'date': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                        'price': item.get('price', item.get('current_price', 0)),
                                        'source': 'market_data_json_data_list'
                                    })
                                    break
                        elif isinstance(market_data['data'], dict) and symbol in market_data['data']:
                            actual_prices.append({
                                'date': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                'price': market_data['data'][symbol].get('price', market_data['data'][symbol].get('current_price', 0)),
                                'source': 'market_data_json_data_dict'
                            })
            except Exception as e:
                print(f"Lỗi khi đọc file {json_file}: {str(e)}")
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
                            'date': row[date_column],
                            'price': row[price_column],
                            'source': 'csv'
                        })
            except Exception as e:
                print(f"Lỗi khi đọc file CSV {csv_files[0]}: {str(e)}")
        
        # Chuyển sang DataFrame và sắp xếp theo ngày
        if len(actual_prices) > 0:
            actual_df = pd.DataFrame(actual_prices)
            actual_df['date'] = pd.to_datetime(actual_df['date'])
            actual_df = actual_df.sort_values('date')
            
            # Loại bỏ các giá trị trùng lặp, giữ lại bản ghi mới nhất
            actual_df = actual_df.drop_duplicates(subset='date', keep='last')
            
            return actual_df
        else:
            print(f"Không tìm thấy dữ liệu giá thực tế cho {symbol} trong khoảng thời gian từ {start_date} đến {end_date}")
            return pd.DataFrame(columns=['date', 'price', 'source'])
    
    def prepare_data(self, symbol, timeframe, days=30):
        """Chuẩn bị dữ liệu cho biểu đồ"""
        predictions = self.load_predictions(symbol, days)
        actual_data = self.load_actual_prices(symbol, days)
        
        # Kiểm tra dữ liệu
        if len(predictions) == 0 or actual_data.empty:
            st.warning(f"Không đủ dữ liệu để hiển thị cho {symbol}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Tạo dataframe cho dự đoán
        pred_records = []
        for pred in predictions:
            if timeframe in pred:
                pred_time = datetime.fromisoformat(pred['timestamp'])
                pred_price = pred[timeframe].get('price', None)
                confidence = pred[timeframe].get('confidence', None)
                
                if pred_price is not None:
                    pred_records.append({
                        'date': pred_time,
                        'value': pred_price,
                        'type': 'prediction',
                        'confidence': confidence
                    })
        
        # Tạo dataframe cho giá thực tế
        actual_records = []
        for _, actual in actual_data.iterrows():
            actual_records.append({
                'date': actual['date'],
                'value': actual['price'],
                'type': 'actual'
            })
        
        # Kết hợp dữ liệu
        df_pred = pd.DataFrame(pred_records)
        df_actual = pd.DataFrame(actual_records)
        
        return df_pred, df_actual
    
    def run_dashboard(self):
        """Hiển thị dashboard"""
        st.title("BondZiA AI Dashboard Hiệu suất")
        
        # Debug mode
        debug_mode = st.sidebar.checkbox("Chế độ gỡ lỗi", value=False)

        # Sidebar
        st.sidebar.header("Cấu hình")
        symbols = self.get_symbols()
        
        if debug_mode:
            st.sidebar.write(f"Tìm thấy {len(symbols)} cổ phiếu: {symbols}")

        if not symbols:
            st.error("Không tìm thấy dữ liệu dự đoán cho bất kỳ cổ phiếu nào.")
            st.info(f"Thư mục dự đoán: {self.prediction_history_dir}")
            st.info(f"Các file trong thư mục dự đoán: {os.listdir(self.prediction_history_dir)}")
            return
        
        symbol = st.sidebar.selectbox("Chọn cổ phiếu", symbols)
        timeframe = st.sidebar.selectbox("Chọn timeframe", ["intraday", "five_day", "monthly"])
        days = st.sidebar.slider("Số ngày phân tích", 5, 90, 30)
        
        # Tải dữ liệu
        if debug_mode:
            st.write(f"Đang tải dữ liệu cho {symbol} - {timeframe} trong {days} ngày qua")

        # Tải dự đoán
        predictions = self.load_predictions(symbol, days)
        if debug_mode:
            st.write(f"Tìm thấy {len(predictions)} dự đoán")
            if len(predictions) > 0:
                st.json(predictions[0])

        # Tải giá thực tế
        actual_data = self.load_actual_prices(symbol, days)
        if debug_mode:
            if not actual_data.empty:
                st.write(f"Tìm thấy {len(actual_data)} giá thực tế")
                st.dataframe(actual_data.head())
            else:
                st.write("Không tìm thấy giá thực tế")

        # Chuẩn bị dữ liệu
        df_pred, df_actual = self.prepare_data(symbol, timeframe, days)
        
        if df_pred.empty or df_actual.empty:
            st.warning(f"Không đủ dữ liệu để hiển thị cho {symbol} - {timeframe}")
            
            if debug_mode:
                st.write("Thông tin chi tiết:")
                if df_pred.empty:
                    st.write("- df_pred trống")
                else:
                    st.write(f"- df_pred có dữ liệu: {len(df_pred)} hàng")
                    st.dataframe(df_pred)
                
                if df_actual.empty:
                    st.write("- df_actual trống")
                else:
                    st.write(f"- df_actual có dữ liệu: {len(df_actual)} hàng")
                    st.dataframe(df_actual)
            
            return
        
        # Hiển thị thông tin tổng quan
        st.header(f"Biểu đồ dự đoán {symbol} - {timeframe}")
        
        # Biểu đồ so sánh dự đoán và thực tế
        fig = go.Figure()
        
        # Thêm giá thực tế
        fig.add_trace(go.Scatter(
            x=df_actual['date'],
            y=df_actual['value'],
            mode='lines+markers',
            name='Giá thực tế',
            line=dict(color='blue', width=2)
        ))
        
        # Thêm dự đoán
        if 'confidence' in df_pred.columns:
            # Sử dụng màu sắc dựa trên độ tin cậy
            fig.add_trace(go.Scatter(
                x=df_pred['date'],
                y=df_pred['value'],
                mode='markers',
                name='Dự đoán',
                marker=dict(
                    size=10,
                    color=df_pred['confidence'],
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title="Độ tin cậy (%)")
                ),
                text=df_pred['confidence'].apply(lambda x: f"Độ tin cậy: {x:.1f}%")
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df_pred['date'],
                y=df_pred['value'],
                mode='markers',
                name='Dự đoán',
                marker=dict(size=10, color='red')
            ))
        
        fig.update_layout(
            title=f"Giá thực tế vs. Dự đoán - {symbol} ({timeframe})",
            xaxis_title="Thời gian",
            yaxis_title="Giá",
            hovermode="closest",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tính các metrics
        if not df_pred.empty and not df_actual.empty:
            st.header("Metrics Hiệu suất")
            
            # Kết hợp dữ liệu
            df_pred_for_eval = df_pred.copy()
            
            # Tính toán metrics
            # Đối với mỗi dự đoán, tìm giá thực tế gần nhất
            accuracy_records = []
            
            for _, pred_row in df_pred.iterrows():
                pred_date = pred_row['date']
                pred_value = pred_row['value']
                
                # Tìm giá thực tế tương ứng
                target_date = None
                if timeframe == 'intraday':
                    # Tìm giá đóng cửa cùng ngày
                    target_date = pred_date.replace(hour=16, minute=0, second=0)
                elif timeframe == 'five_day':
                    # Tìm giá sau 5 ngày giao dịch
                    target_date = pred_date + timedelta(days=7)
                elif timeframe == 'monthly':
                    # Tìm giá sau 30 ngày
                    target_date = pred_date + timedelta(days=30)
                
                # Tìm giá thực tế gần nhất với target_date
                closest_actual = df_actual.iloc[(df_actual['date'] - target_date).abs().argsort()[:1]]
                
                if not closest_actual.empty:
                    actual_date = closest_actual['date'].iloc[0]
                    actual_value = closest_actual['value'].iloc[0]
                    
                    # Tính sai số
                    error = actual_value - pred_value
                    pct_error = (error / actual_value) * 100
                    
                    # Xác định hướng
                    prev_actual = df_actual[df_actual['date'] < pred_date]
                    
                    if not prev_actual.empty:
                        prev_price = prev_actual.iloc[-1]['value']
                        pred_direction = 'up' if pred_value > prev_price else 'down'
                        actual_direction = 'up' if actual_value > prev_price else 'down'
                        direction_correct = pred_direction == actual_direction
                    else:
                        direction_correct = None
                    
                    accuracy_records.append({
                        'pred_date': pred_date,
                        'actual_date': actual_date,
                        'pred_value': pred_value,
                        'actual_value': actual_value,
                        'error': error,
                        'pct_error': pct_error,
                        'direction_correct': direction_correct,
                        'confidence': pred_row.get('confidence', None)
                    })
            
            if len(accuracy_records) > 0:
                df_accuracy = pd.DataFrame(accuracy_records)
                
                # Hiển thị metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    direction_accuracy = df_accuracy['direction_correct'].mean() * 100
                    st.metric("Độ chính xác theo hướng", f"{direction_accuracy:.1f}%")
                
                with col2:
                    mae = df_accuracy['error'].abs().mean()
                    st.metric("MAE (Sai số tuyệt đối trung bình)", f"${mae:.2f}")
                
                with col3:
                    mape = df_accuracy['pct_error'].abs().mean()
                    st.metric("MAPE (Sai số phần trăm tuyệt đối trung bình)", f"{mape:.2f}%")
                
                # Biểu đồ sai số theo thời gian
                st.subheader("Sai số theo thời gian")
                
                fig_error = px.scatter(
                    df_accuracy,
                    x='pred_date',
                    y='pct_error',
                    color='confidence' if 'confidence' in df_accuracy.columns else None,
                    color_continuous_scale='viridis',
                    title=f"Sai số dự đoán theo thời gian - {symbol} ({timeframe})"
                )
                
                fig_error.update_layout(
                    xaxis_title="Thời gian dự đoán",
                    yaxis_title="Sai số (%)",
                    height=400
                )
                
                st.plotly_chart(fig_error, use_container_width=True)
                
                # Biểu đồ độ tin cậy vs. độ chính xác
                if 'confidence' in df_accuracy.columns:
                    st.subheader("Mối quan hệ giữa độ tin cậy và độ chính xác")
                    
                    fig_conf = px.scatter(
                        df_accuracy,
                        x='confidence',
                        y='pct_error',
                        title=f"Độ tin cậy vs. Sai số - {symbol} ({timeframe})"
                    )
                    
                    fig_conf.update_layout(
                        xaxis_title="Độ tin cậy (%)",
                        yaxis_title="Sai số (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_conf, use_container_width=True)
                    
                    # Binning theo độ tin cậy
                    st.subheader("Độ chính xác theo nhóm độ tin cậy")
                    
                    # Tạo nhóm độ tin cậy
                    df_accuracy['confidence_bin'] = pd.cut(
                        df_accuracy['confidence'], 
                        bins=[0, 25, 50, 75, 100], 
                        labels=['0-25%', '25-50%', '50-75%', '75-100%']
                    )
                    
                    # Tính độ chính xác theo hướng cho mỗi nhóm
                    direction_by_conf = df_accuracy.groupby('confidence_bin')['direction_correct'].mean() * 100
                    
                    fig_bins = px.bar(
                        direction_by_conf,
                        title="Độ chính xác theo nhóm độ tin cậy"
                    )
                    
                    fig_bins.update_layout(
                        xaxis_title="Nhóm độ tin cậy",
                        yaxis_title="Độ chính xác (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_bins, use_container_width=True)
            
            # Hiển thị dữ liệu thô
            if st.checkbox("Hiển thị dữ liệu thô"):
                st.subheader("Dữ liệu dự đoán")
                st.dataframe(df_pred)
                
                st.subheader("Dữ liệu thực tế")
                st.dataframe(df_actual)

def main():
    dashboard = PerformanceDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()