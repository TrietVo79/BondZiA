# BondZiA AI - Hệ thống Dự đoán Giá Cổ phiếu

BondZiA AI là hệ thống dự đoán giá cổ phiếu sử dụng các mô hình Deep Learning tiên tiến. Hệ thống theo dõi và dự đoán giá cổ phiếu trên sàn NASDAQ ở 3 mốc thời gian: intraday (trong ngày), 5 ngày và 1 tháng.

## Tính năng chính

- **Dự đoán giá cổ phiếu** ở 3 mốc thời gian: intraday, 5 ngày và 1 tháng
- **Tự động theo dõi giờ giao dịch** của NASDAQ, tính toán DST (Daylight Saving Time)
- **Tự động tiến hóa** để cải thiện mô hình dự đoán
- **Tự sửa lỗi** và khôi phục khi gặp sự cố
- **Gửi thông báo qua Discord và Telegram** về dự đoán giá và trạng thái hệ thống
- **Tạo biểu đồ trực quan** về dự đoán giá
- **Lưu trữ và quản lý phiên bản** của mô hình
- **Mô hình tổng hợp (Ensemble)** kết hợp nhiều mô hình để cho kết quả chính xác hơn
- **Tự đánh giá độ tin cậy** của dự đoán dựa trên nhiều yếu tố
- **Đa nguồn dữ liệu** tích hợp từ Polygon, Yahoo Finance, và các nguồn miễn phí khác
- **Phân tích hiệu suất dự đoán** qua báo cáo và dashboard trực quan

## Cài đặt

### Yêu cầu hệ thống

- Python 3.9
- NVIDIA GPU (đề xuất)
- Tài khoản Polygon.io (gói Stocks Advanced)
- Webhook Discord hoặc Telegram

Bước 2: Cài đặt các thư viện cần thiết
pip install -r requirements.txt
Nếu cần cài đặt thêm các thư viện cho module đánh giá hiệu suất:
pip install streamlit plotly pandas numpy matplotlib

Bước 3: Cấu hình API và webhook
	1	Mở file config/api_keys.json
	2	Thêm API key Polygon.io và webhook Discord/Telegram:
{
  "polygon": {
    "api_key": "YOUR_POLYGON_API_KEY",
    "plan": "Stocks Advanced"
  },
  "discord": {
    "prediction_webhook": "YOUR_DISCORD_PREDICTION_WEBHOOK_URL",
    "update_webhook": "YOUR_DISCORD_UPDATE_WEBHOOK_URL"
  },
  "telegram": {
    "bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
    "chat_id": "YOUR_TELEGRAM_CHAT_ID"
  }
}

Bước 4: Cấu hình danh sách cổ phiếu theo dõi
	1	Mở file config/stocks.json
	2	Cập nhật danh sách cổ phiếu theo nhu cầu (mặc định đã có 9 cổ phiếu)
Bước 5: Kiểm tra cấu hình hệ thống
Kiểm tra lại cấu hình chính trong file config/system_config.json. Tùy chỉnh theo nhu cầu:
	•	Cấu hình giờ giao dịch
	•	Cấu hình dự đoán
	•	Cấu hình tiến hóa AI
	•	Cấu hình thông báo và hiển thị
	•	Cấu hình độ tin cậy
	•	Cấu hình nguồn dữ liệu

Sử dụng

Khởi động hệ thống
python main.py

Khởi động với các tùy chọn
	•	Chạy dự đoán ngay lập tức:
python main.py --predict
	•	Chạy tiến hóa mô hình ngay lập tức:
python main.py --evolve
	•	Huấn luyện tất cả mô hình ngay lập tức:
python main.py --train
	•	Tạo báo cáo hiệu suất cho một cổ phiếu:
python evaluation/performance_reporter.py --symbol TSLA --days 30
	•	Khởi động dashboard hiệu suất:
cd evaluation
streamlit run performance_dashboard.py
	•	Đánh giá độ chính xác tự động:
python evaluation/accuracy_evaluator.py --symbol TSLA
	•	Chỉ định đường dẫn file cấu hình khác:
python main.py --config path/to/config.json

Chạy BondZiA trong background
Để duy trì hệ thống chạy liên tục trong background trên server Linux:
nohup python main.py > bondzia.log 2>&1 &


```
BondZiA
├─ BondZiA_versions
│  ├─ BondZiA_v1.0.14
│  │  └─ models
│  │     ├─ five_day
│  │     │  ├─ AGX_five_day_model.h5
│  │     │  ├─ AMZN_five_day_model.h5
│  │     │  ├─ MSFT_five_day_model.h5
│  │     │  ├─ NVDA_five_day_model.h5
│  │     │  ├─ PLTR_five_day_model.h5
│  │     │  └─ TSLA_five_day_model.h5
│  │     ├─ intraday
│  │     │  ├─ AGX_intraday_model.h5
│  │     │  ├─ AMZN_intraday_model.h5
│  │     │  ├─ MSFT_intraday_model.h5
│  │     │  ├─ NVDA_intraday_model.h5
│  │     │  ├─ PLTR_intraday_model.h5
│  │     │  └─ TSLA_intraday_model.h5
│  │     └─ monthly
│  │        ├─ AMZN_monthly_model.h5
│  │        ├─ MSFT_monthly_model.h5
│  │        ├─ NVDA_monthly_model.h5
│  │        ├─ PLTR_monthly_model.h5
│  │        └─ TSLA_monthly_model.h5
├─ charts
│  ├─ NVDA_prediction_20250403_122914.html
│  ├─ TSLA_prediction_20250404_121052.png
│  ├─ TSLA_prediction_20250404_125120.html
│  └─ TSLA_prediction_20250404_125120.png
├─ config
│  ├─ api_keys.json
│  ├─ stocks.json
│  └─ system_config.json
├─ data
│  ├─ backtest
│  ├─ processed
│  │  ├─ AAPL_daily_processed_2015-04-05_2025-04-02.csv
│  │  ├─ AAPL_daily_processed_2020-04-02_2025-04-01.csv
│  │  ├─ AAPL_monthly_processed_2015-04-05_2025-04-02.csv
│  │  ├─ AGX_daily_processed_2015-04-05_2025-04-02.csv
│  │  ├─ AGX_daily_processed_2020-04-02_2025-04-01.csv
│  │  ├─ AGX_monthly_processed_2015-04-05_2025-04-02.csv
│  │  ├─ AMZN_daily_processed_2015-04-05_2025-04-02.csv
│  │  ├─ AMZN_daily_processed_2020-04-02_2025-04-01.csv
│  │  ├─ AMZN_monthly_processed_2015-04-05_2025-04-02.csv
│  │  ├─ MSFT_daily_processed_2015-04-05_2025-04-02.csv
│  │  ├─ MSFT_daily_processed_2020-04-02_2025-04-01.csv
│  │  ├─ MSFT_monthly_processed_2015-04-05_2025-04-02.csv
│  │  ├─ NVDA_daily_processed_2015-04-05_2025-04-02.csv
│  │  ├─ NVDA_daily_processed_2020-04-02_2025-04-01.csv
│  │  ├─ NVDA_monthly_processed_2015-04-05_2025-04-02.csv
│  │  ├─ PLTR_daily_processed_2015-04-05_2025-04-02.csv
│  │  ├─ PLTR_daily_processed_2020-04-02_2025-04-01.csv
│  │  ├─ PLTR_monthly_processed_2015-04-05_2025-04-02.csv
│  │  ├─ TSLA_daily_processed_2015-04-05_2025-04-02.csv
│  │  ├─ TSLA_daily_processed_2020-04-02_2025-04-01.csv
│  │  └─ TSLA_monthly_processed_2015-04-05_2025-04-02.csv
│  └─ raw
│     ├─ AAPL_daily_2015-04-05_2025-04-02.csv
│     ├─ AAPL_daily_2020-04-02_2025-04-01.csv
│     ├─ AMZN_daily_2015-04-05_2025-04-02.csv
│     ├─ AMZN_daily_2020-04-02_2025-04-01.csv
│     ├─ AMZN_daily_2025-01-06_2025-04-04.csv
│     ├─ META_daily_2015-04-05_2025-04-02.csv
│     ├─ META_daily_2020-04-02_2025-04-01.csv
│     ├─ MSFT_daily_2015-04-05_2025-04-02.csv
│     ├─ MSFT_daily_2020-04-02_2025-04-01.csv
│     ├─ MSFT_daily_2025-01-06_2025-04-04.csv
│     ├─ NVDA_daily_2015-04-05_2025-04-02.csv
│     ├─ NVDA_daily_2020-04-02_2025-04-01.csv
│     ├─ NVDA_daily_2025-01-06_2025-04-04.csv
│     ├─ PLTR_daily_2015-04-05_2025-04-02.csv
│     ├─ PLTR_daily_2020-04-02_2025-04-01.csv
│     ├─ PLTR_daily_2025-01-06_2025-04-04.csv
│     ├─ TSLA_daily_2015-04-05_2025-04-02.csv
│     ├─ TSLA_daily_2020-04-02_2025-04-01.csv
│     ├─ TSLA_daily_2025-01-06_2025-04-04.csv
│     ├─ market_data_20250404_110320.json
│     ├─ market_data_20250404_111010.json
│     ├─ market_data_20250404_113017.json
│     ├─ market_data_20250404_120046.json
│     ├─ market_data_20250404_122541.json
│     ├─ market_data_20250404_125116.json
│     └─ market_data_20250404_125614.json
├─ evolution
│  ├─ accuracy_evaluator.py
│  ├─ hyperparameter_tuner.py
│  ├─ performance_dashboard.py
│  ├─ performance_reporter.py
│  ├─ results
│  │  ├─ AGX_five_day_hyperparameters_20250401_000240.json
│  │  ├─ AGX_five_day_hyperparameters_20250401_051911.json
│  │  
│  └─ version_manager.py
├─ home
│  └─ trietvo
│     └─ BondZiA
│        └─ config
│           └─ system_config.json
├─ integration
│  ├─ integration_script.py
│  └─ new_modules
│     ├─ base_predictor.py
│     ├─ base_predictor_integration.py
│     ├─ confidence_evaluator.py
│     ├─ data_sources_integration.py
│     ├─ enhanced_data_fetcher.py
│     ├─ ensemble_model.py
│     ├─ ensemble_predictor.py
│     └─ specific_predictors.py
├─ logs
│  ├─ bondzia_2025-04-02.log
│  ├─ bondzia_2025-04-03.log
│  ├─ bondzia_2025-04-04.log
│  ├─ errors
│  │  ├─ error_20250331_010804.log
│  │  ├─ error_20250331_010805.log
│  │  ├─ error_20250331_010813.log
│  │  ├─ error_20250331_010821.log
│  │  ├─ error_20250331_010822.log
│  │  ├─ errors_2025-03-31.log
│  │  ├─ errors_2025-04-01.log
│  │  ├─ errors_2025-04-02.log
│  │  ├─ errors_2025-04-03.log
│  │  └─ errors_2025-04-04.log
│  ├─ evolution
│  │  ├─ evolution_2025-03-31.log
│  │  ├─ evolution_2025-04-01.log
│  │  ├─ evolution_2025-04-02.log
│  │  ├─ evolution_2025-04-03.log
│  │  └─ evolution_2025-04-04.log
│  ├─ predictions
│  │  ├─ predictions_2025-03-31.log
│  │  ├─ predictions_2025-04-01.log
│  │  ├─ predictions_2025-04-02.log
│  │  ├─ predictions_2025-04-03.log
│  │  └─ predictions_2025-04-04.log
│  ├─ startup.log
│  └─ system
│     ├─ bondzia_2025-03-31.log.zip
│     ├─ bondzia_2025-04-01.log.zip
│     ├─ bondzia_2025-04-02.log.zip
│     ├─ bondzia_2025-04-03.log.zip
│     └─ bondzia_2025-04-04.log
├─ main.py
├─ models
│  ├─ base_predictor.py
│  ├─ ensemble
│  ├─ ensemble_predictor.py
│  ├─ five_day
│  │  ├─ AGX_five_day_model.h5
│  │  ├─ AMZN_five_day_model.h5
│  │  ├─ MSFT_five_day_model.h5
│  │  ├─ NVDA_five_day_model.h5
│  │  ├─ PLTR_five_day_model.h5
│  │  ├─ TSLA_five_day_model.h5
│  │  └─ scalers
│  │     ├─ AGX_five_day_feature_scaler.pkl
│  │     ├─ AGX_five_day_price_scaler.pkl
│  │     ├─ AMZN_five_day_feature_scaler.pkl
│  │     ├─ AMZN_five_day_price_scaler.pkl
│  │     ├─ MSFT_five_day_feature_scaler.pkl
│  │     ├─ MSFT_five_day_price_scaler.pkl
│  │     ├─ NVDA_five_day_feature_scaler.pkl
│  │     ├─ NVDA_five_day_price_scaler.pkl
│  │     ├─ PLTR_five_day_feature_scaler.pkl
│  │     ├─ PLTR_five_day_price_scaler.pkl
│  │     ├─ TSLA_five_day_feature_scaler.pkl
│  │     └─ TSLA_five_day_price_scaler.pkl
│  ├─ history
│  ├─ intraday
│  │  ├─ AGX_intraday_model.h5
│  │  ├─ AMZN_intraday_model.h5
│  │  ├─ MSFT_intraday_model.h5
│  │  ├─ NVDA_intraday_model.h5
│  │  ├─ PLTR_intraday_model.h5
│  │  ├─ TSLA_intraday_model.h5
│  │  └─ scalers
│  │     ├─ AGX_intraday_feature_scaler.pkl
│  │     ├─ AGX_intraday_price_scaler.pkl
│  │     ├─ AMZN_intraday_feature_scaler.pkl
│  │     ├─ AMZN_intraday_price_scaler.pkl
│  │     ├─ MSFT_intraday_feature_scaler.pkl
│  │     ├─ MSFT_intraday_price_scaler.pkl
│  │     ├─ NVDA_intraday_feature_scaler.pkl
│  │     ├─ NVDA_intraday_price_scaler.pkl
│  │     ├─ PLTR_intraday_feature_scaler.pkl
│  │     ├─ PLTR_intraday_price_scaler.pkl
│  │     ├─ TSLA_intraday_feature_scaler.pkl
│  │     └─ TSLA_intraday_price_scaler.pkl
│  ├─ monthly
│  │  ├─ AMZN_monthly_model.h5
│  │  ├─ MSFT_monthly_model.h5
│  │  ├─ NVDA_monthly_model.h5
│  │  ├─ PLTR_monthly_model.h5
│  │  ├─ TSLA_monthly_model.h5
│  │  └─ scalers
│  │     ├─ AMZN_monthly_feature_scaler.pkl
│  │     ├─ AMZN_monthly_price_scaler.pkl
│  │     ├─ MSFT_monthly_feature_scaler.pkl
│  │     ├─ MSFT_monthly_price_scaler.pkl
│  │     ├─ NVDA_monthly_feature_scaler.pkl
│  │     ├─ NVDA_monthly_price_scaler.pkl
│  │     ├─ PLTR_monthly_feature_scaler.pkl
│  │     ├─ PLTR_monthly_price_scaler.pkl
│  │     ├─ TSLA_monthly_feature_scaler.pkl
│  │     └─ TSLA_monthly_price_scaler.pkl
│  └─ specific_predictors.py
├─ prediction_history
│  ├─ AMZN_prediction_20250404_113017.json
│  ├─ AMZN_prediction_20250404_125116.json
│  ├─ MSFT_prediction_20250404_113017.json
│  ├─ MSFT_prediction_20250404_125116.json
│  ├─ NVDA_prediction_20250404_113017.json
│  ├─ NVDA_prediction_20250404_125116.json
│  ├─ PLTR_prediction_20250404_113017.json
│  ├─ PLTR_prediction_20250404_125116.json
│  ├─ TSLA_prediction_20250404_113017.json
│  └─ TSLA_prediction_20250404_125116.json
├─ prediction_stats
│  ├─ NVDA_accuracy_20250404.json
│  └─ NVDA_accuracy_summary.json
├─ requirements.txt
├─ status
│  └─ evolution_status.json
├─ utils
│  ├─ confidence_evaluator.py
│  ├─ convert_csv_to_json.py
│  ├─ csv_to_json_converter.log
│  ├─ data_fetcher.py
│  ├─ data_loader.log
│  ├─ data_loader.py
│  ├─ discord_notifier.py
│  ├─ enhanced_data_fetcher.py
│  ├─ error_handler.py
│  ├─ logger_config.py
│  ├─ market_hours.py
│  ├─ telegram_integration.py
│  ├─ telegram_notifier.py
│  └─ visualization.py

Mô hình Dự đoán
BondZiA sử dụng các mô hình Deep Learning tiên tiến cho từng mốc thời gian:
1. Dự đoán Intraday (Trong ngày)
	•	Mô hình chính: Temporal Fusion Transformer (TFT)
	•	Mô hình phụ: LSTM với Attention mechanism, Bidirectional LSTM
	•	Đặc điểm: Kết hợp Transformer với attention mechanism
	•	Dữ liệu vào: Dữ liệu tick-by-tick tần suất cao
	•	Phương pháp: Sử dụng WebSocket stream realtime
2. Dự đoán 5 ngày
	•	Mô hình chính: LSTM với Attention mechanism
	•	Mô hình phụ: GRU, CNN-LSTM, Stacked LSTM
	•	Đặc điểm: Kết hợp LSTM để xử lý chuỗi thời gian và attention
	•	Dữ liệu vào: Dữ liệu OHLC, volume, và các chỉ báo kỹ thuật
	•	Phương pháp: Multi-input, Transfer Learning
3. Dự đoán 1 tháng
	•	Mô hình chính: TimeGPT
	•	Mô hình phụ: GRU-LSTM Hybrid, Attention-based GRU
	•	Đặc điểm: Mô hình dựa trên GPT tối ưu cho chuỗi thời gian
	•	Dữ liệu vào: Dữ liệu daily bars kết hợp với chỉ số kinh tế
	•	Phương pháp: Feature augmentation, Bayesian Optimization
4. Mô hình Tổng hợp (Ensemble)
	•	Phương pháp: Kết hợp 5 mô hình khác nhau cho mỗi khung thời gian
	•	Meta-model: Gradient Boosting Regressor
	•	Đánh giá: Tự động so sánh hiệu suất và chọn mô hình tốt nhất
	•	Kết quả: Dự đoán giá cuối cùng với khoảng tin cậy

Hệ thống Đánh giá Độ tin cậy
BondZiA bao gồm hệ thống đánh giá độ tin cậy cho mỗi dự đoán:
1. Các yếu tố ảnh hưởng đến độ tin cậy
	•	Sự đồng thuận của mô hình: Mức độ thống nhất giữa các mô hình
	•	Điều kiện thị trường: Biến động (volatility) và tâm lý (sentiment)
	•	Lịch sử dự đoán: Độ chính xác của các dự đoán trước đó
	•	Chỉ báo kỹ thuật: Mức độ nhất quán giữa các chỉ báo
	•	Khung thời gian: Mức độ khó dự đoán của mỗi khung thời gian
2. Cách tính độ tin cậy
	•	Cơ sở: Tính độ phân tán dự đoán giữa các mô hình (CV - Coefficient of Variation)
	•	Điều chỉnh thị trường: Giảm độ tin cậy khi thị trường biến động cao (VIX)
	•	Điều chỉnh theo lịch sử: Tăng/giảm dựa trên độ chính xác trong quá khứ
	•	Điều chỉnh theo chỉ báo: Tăng độ tin cậy khi nhiều chỉ báo đồng thuận
3. Cải thiện độ tin cậy
	•	Tự học: Hệ thống theo dõi độ chính xác dự đoán và tự điều chỉnh
	•	Lọc dữ liệu: Phát hiện và loại bỏ outliers, điểm dữ liệu bất thường
	•	Tự đánh giá: Giảm độ tin cậy khi phát hiện điều kiện bất thường
	•	Cảnh báo: Thông báo khi độ tin cậy dưới ngưỡng cho phép

Nguồn dữ liệu đa dạng

BondZiA tích hợp dữ liệu từ nhiều nguồn khác nhau:
1. Nguồn dữ liệu chính
	•	Polygon.io: Dữ liệu OHLCV chính xác theo thời gian thực
	•	Yahoo Finance: Dữ liệu bổ sung, thông tin công ty, chỉ số tài chính
	•	Dữ liệu kinh tế vĩ mô: Chỉ số thị trường, lãi suất, chỉ số kinh tế
2. Phân tích tâm lý
	•	Tin tức cổ phiếu: Phân tích tâm lý từ tin tức
	•	Google Trends: Xu hướng tìm kiếm liên quan
	•	Bảng điều khiển thị trường: VIX, S&P 500, Dow Jones
3. Xử lý dữ liệu nâng cao
	•	Phát hiện outliers: Tự động phát hiện và xử lý dữ liệu bất thường
	•	Điền missing values: Sử dụng phương pháp điền nâng cao
	•	Feature engineering: Tạo các chỉ báo kỹ thuật, chỉ số tâm lý

Tiến hóa Mô hình
BondZiA có khả năng tự động cải thiện hiệu suất dự đoán qua cơ chế tiến hóa:
	1	Tối ưu hóa siêu tham số: Tự động tìm kiếm bộ tham số tối ưu
	2	Tiến hóa kiến trúc: Thử nghiệm các cấu trúc mô hình khác nhau
	3	Quản lý phiên bản: Lưu trữ và theo dõi tiến trình cải thiện
	4	Tự động kiểm tra hiệu suất: So sánh các phiên bản mô hình
Lịch trình tiến hóa
	•	Ngày trong tuần: 16:01 - 06:59 (ngoài giờ giao dịch)
	•	Cuối tuần: Chạy mỗi 3 giờ để tối ưu hóa liên tục
	•	Tiến hóa ưu tiên: Cho các cổ phiếu có độ chính xác thấp
Thông báo và Biểu đồ
BondZiA tạo các thông báo chi tiết qua Discord và Telegram:
1. Thông báo dự đoán
	•	Dự đoán giá: Giá dự đoán cho 3 khung thời gian
	•	Độ tin cậy: Mức độ tin cậy của từng dự đoán (%)
	•	Lý do: Phân tích nguyên nhân của dự đoán
	•	Biểu đồ: Biểu đồ trực quan hiển thị giá hiện tại và dự đoán
2. Biểu đồ nâng cao
	•	Khoảng tin cậy: Hiển thị khoảng dự đoán với xác suất
	•	Chỉ báo kỹ thuật: Hiển thị các chỉ báo kỹ thuật chính
	•	So sánh lịch sử: Độ chính xác của các dự đoán trước

Đánh giá hiệu suất
BondZiA cung cấp nhiều công cụ để đánh giá hiệu suất dự đoán:

1. Báo cáo hiệu suất (Performance Reporter)
Module báo cáo hiệu suất phân tích lịch sử dự đoán và so sánh với giá thực tế:
	•	Tính năng chính:
	◦	Phân tích lịch sử dự đoán từ thư mục prediction_history/
	◦	Tải giá thực tế từ dữ liệu thị trường trong thư mục data/raw/
	◦	Tính toán các metrics đánh giá: MAE, MAPE, direction accuracy
	◦	Tạo biểu đồ trực quan của hiệu suất dự đoán
	◦	Phân tích mối quan hệ giữa độ tin cậy và độ chính xác
	•	Sử dụng: python evaluation/performance_reporter.py --symbol TSLA --days 30
	•	
	•	Báo cáo: Tạo các file báo cáo JSON và biểu đồ trong thư mục reports/

2. Dashboard hiệu suất (Performance Dashboard)
Dashboard trực quan theo thời gian thực để theo dõi hiệu suất dự đoán:
	•	Tính năng chính:
	◦	Giao diện web đơn giản với Streamlit
	◦	Biểu đồ so sánh dự đoán và giá thực tế
	◦	Phân tích độ chính xác theo khung thời gian
	◦	Phân tích độ tin cậy và mối quan hệ với độ chính xác
	◦	Bảng dữ liệu chi tiết về các dự đoán
	•	Sử dụng: cd evaluation
	•	streamlit run performance_dashboard.py
	•	
	•	Truy cập: Mở trình duyệt và truy cập http://localhost:8501

3. Đánh giá độ chính xác tự động (Accuracy Evaluator)
Module tự động đánh giá độ chính xác và đề xuất điều chỉnh mô hình:
	•	Tính năng chính:
	◦	Đánh giá tự động lịch sử dự đoán
	◦	Tính toán metrics đánh giá chi tiết
	◦	Theo dõi hiệu suất qua thời gian
	◦	Phát hiện và đánh dấu mô hình cần điều chỉnh
	◦	Lưu trữ thống kê trong thư mục prediction_stats/
	•	Sử dụng: python evaluation/accuracy_evaluator.py --symbol TSLA
	•	
	•	Tích hợp: Tự động chạy hàng ngày lúc 20:00 (có thể tùy chỉnh)

4. Backtesting (Kiểm thử lịch sử)
Đánh giá hiệu suất mô hình trên dữ liệu lịch sử:
python main.py --backtest --days 30 --symbol TSLA
Khắc phục sự cố
Lỗi kết nối Polygon API
	•	Kiểm tra API key trong config/api_keys.json
	•	Xác nhận tình trạng dịch vụ Polygon.io
	•	Kiểm tra giới hạn truy vấn API
	•	Hệ thống sẽ tự động chuyển sang Yahoo Finance nếu Polygon gặp sự cố
Lỗi dự đoán
	•	Kiểm tra logs trong logs/errors/
	•	Đảm bảo có dữ liệu cho các cổ phiếu cần dự đoán
	•	Kiểm tra các mô hình trong thư mục models/
	•	Sử dụng tùy chọn --train để huấn luyện lại mô hình
Lỗi webhook Discord/Telegram
	•	Xác minh các URL webhook và tokens trong config/api_keys.json
	•	Kiểm tra quyền của bot trên kênh
	•	Xem logs trong logs/errors/
	•	Với lỗi Telegram, tăng thời gian chờ giữa các tin nhắn
Cải thiện độ tin cậy thấp
	•	Kiểm tra lịch sử dự đoán trong prediction_history/
	•	Chạy lệnh --evolve để tiến hóa mô hình
	•	Bổ sung nguồn dữ liệu trong config/system_config.json
	•	Tăng số lượng dữ liệu huấn luyện

Hướng dẫn nâng cao

1. Tùy chỉnh độ tin cậy

Điều chỉnh cơ chế đánh giá độ tin cậy trong config/system_config.json:
"confidence": {
  "verify_interval_hours": 24, 
  "min_predictions_for_stats": 10,
  "adjust_by_market_conditions": true,
  "min_threshold_for_alerts": 40
}

2. Kiểm soát mô hình Ensemble
Tùy chỉnh cách sử dụng mô hình tổng hợp:
"ensemble": {
  "use_ensemble": true,
  "models_count": 5,
  "weighting_strategy": "adaptive",
  "meta_model": "gradient_boosting"
}

3. Theo dõi phiên bản mô hình
Xem lịch sử và hiệu suất các phiên bản mô hình:
python main.py --list-versions --symbol TSLA

4. Kiểm soát cảnh báo
Cấu hình cảnh báo khi phát hiện xu hướng bất thường:
"alerts": {
  "deviation_threshold": 10,
  "confidence_threshold": 30,
  "volatility_threshold": 5
}

Tính năng mới (v1.0.11)

1. Mô hình Ensemble nâng cao
Phiên bản mới tích hợp mô hình tổng hợp (Ensemble) kết hợp 5 kiến trúc khác nhau:
	•	LSTM đơn giản
	•	LSTM sâu
	•	GRU
	•	CNN-LSTM
	•	Bidirectional LSTM
Mô hình tổng hợp sử dụng GradientBoostingRegressor làm meta-model để kết hợp kết quả từ các mô hình cơ sở, cải thiện độ chính xác lên đến 17% so với mô hình đơn.

2. Hệ thống đánh giá độ tin cậy cải tiến
	•	Phân tích đồng thuận mô hình: Đánh giá mức độ thống nhất giữa 5 mô hình cơ sở
	•	Phân tích điều kiện thị trường: Tự động điều chỉnh độ tin cậy dựa trên điều kiện thị trường
	•	Phân tích lịch sử dự đoán: Học từ độ chính xác các dự đoán trước đó
	•	Phân tích chỉ báo kỹ thuật: Kiểm tra sự đồng thuận của nhiều chỉ báo kỹ thuật

3. Đa nguồn dữ liệu
	•	Tích hợp Yahoo Finance: Tự động chuyển sang Yahoo Finance khi Polygon API gặp vấn đề
	•	Phân tích tâm lý thị trường: Thu thập dữ liệu từ Google Trends và tin tức
	•	Dữ liệu kinh tế vĩ mô: Tích hợp các chỉ số kinh tế quan trọng

4. Lưu trữ và phân tích lịch sử
	•	Theo dõi dự đoán: Lưu và phân tích tất cả dự đoán lịch sử
	•	Quản lý phiên bản: Lưu trữ và theo dõi hiệu suất của các phiên bản mô hình
	•	Tự động tiến hóa: Dựa trên lịch sử hiệu suất để cải thiện mô hình

5. Modules đánh giá hiệu suất
	•	Báo cáo hiệu suất: Phân tích toàn diện lịch sử dự đoán
	•	Dashboard trực quan: Theo dõi hiệu suất theo thời gian thực
	•	Đánh giá độ chính xác tự động: Tự động phát hiện và đề xuất điều chỉnh mô hình
Hiểu về cách đọc kết quả dự đoán
Khi BondZiA đưa ra dự đoán, kết quả sẽ bao gồm các thông tin sau:
	1	Giá dự đoán: Giá cổ phiếu dự kiến tại thời điểm mục tiêu
	2	Phần trăm thay đổi: Phần trăm thay đổi so với giá hiện tại
	3	Hướng: "up" (tăng) hoặc "down" (giảm)
	4	Độ tin cậy: Mức độ tin cậy của dự đoán (0-100%)
	5	Lý do: Giải thích ngắn gọn về các yếu tố ảnh hưởng đến dự đoán
Để đọc kết quả dự đoán hiệu quả:
	•	Độ tin cậy > 70%: Dự đoán có độ tin cậy cao, có thể tham khảo
	•	Độ tin cậy 40-70%: Dự đoán có độ tin cậy trung bình, nên thận trọng
	•	Độ tin cậy < 40%: Dự đoán có độ tin cậy thấp, cần kết hợp với các phân tích khác
Lưu ý rằng vào những thời điểm biến động cao (tin tức quan trọng, báo cáo tài chính, v.v.), độ tin cậy thường giảm do khó dự đoán phản ứng thị trường.

Nâng cấp hệ thống

Nâng cấp phần cứng

BondZiA được thiết kế để chạy trên cấu hình:
	•	Hiện tại: GPU Nvidia Tesla T4, CPU x86_64 (24 cores), RAM: 31 GB
	•	Dự kiến: CPU AMD Ryzen 9 7950X, NVIDIA GeForce RTX 4090, RAM 64GB
Chuyển đổi hệ thống

Để chuyển BondZiA sang hệ thống mới:

	1	Sao lưu tất cả dữ liệu và mô hình:
		tar -czvf bondzia_backup.tar.gz BondZiA/
	2	Di chuyển file sao lưu sang máy mới
	4	Giải nén và cài đặt các thư viện:
	6	tar -xzvf bondzia_backup.tar.gz
	7	cd BondZiA
	8	pip install -r requirements.txt
	9	Tích hợp cải tiến mới
	10	BondZiA có thể được cập nhật với các cải tiến mới thông qua tập lệnh tích hợp:
	11	cd integration
	12	python integration_script.py
	13	Tập lệnh này sẽ:
	14	Sao lưu các tệp hiện tại
	15	Cài đặt các phụ thuộc cần thiết
	16	Cập nhật cấu hình hệ thống
	17	Tích hợp các module mới
	18	Hỗ trợ
	19	Nếu gặp vấn đề trong quá trình cài đặt hoặc sử dụng, vui lòng tham khảo:
	20	File logs trong thư mục logs/
	21	Kiểm tra cấu hình trong thư mục config/
	22	Đảm bảo API key Polygon.io và webhook Discord/Telegram hoạt động chính xác
	23	Lưu ý về độ tin cậy dự đoán
	24	Kết quả dự đoán từ BondZiA luôn đi kèm với độ tin cậy, giúp người dùng đánh giá được mức 		độ tin cậy của dự đoán. Độ tin cậy thấp không có nghĩa là mô hình không tốt, mà là thị 		trường đang trong điều kiện khó dự đoán.
	25	Các yếu tố ảnh hưởng đến độ tin cậy:
	26	Độ biến động thị trường cao
	27	Xu hướng không rõ ràng
	28	Dữ liệu mâu thuẫn từ các nguồn khác nhau
	29	Thiếu dữ liệu lịch sử
	30	BondZiA liên tục cải thiện độ tin cậy dự đoán thông qua:
	31	Tự động học từ các dự đoán trước đó
	32	Tiến hóa mô hình để thích ứng với điều kiện thị trường mới
	33	Tự đánh giá và điều chỉnh dựa trên kết quả thực tế
	34	Đánh giá độ chính xác tự động và đề xuất điều chỉnh
	35	Đóng góp và phát triển
	36	BondZiA là một dự án mở rộng liên tục. Các ý tưởng đóng góp luôn được hoan nghênh:
	37	Tối ưu hiệu suất: Cải thiện tốc độ dự đoán, giảm tài nguyên sử dụng
	38	Thêm chỉ báo kỹ thuật: Tích hợp các chỉ báo kỹ thuật mới
	39	Cải thiện trực quan hóa: Nâng cao chất lượng biểu đồ và dashboard
	40	Cải thiện độ chính xác: Đề xuất cải tiến mô hình hiện có
	41	Hỏi đáp thường gặp
	42	Q: BondZiA có thể dự đoán giá cổ phiếu chính xác 100% không?
	43	A: Không. Thị trường tài chính chịu ảnh hưởng của nhiều yếu tố phức tạp, một số không thể 		dự đoán được. BondZiA cung cấp dự đoán dựa trên dữ liệu lịch sử và các mô hình machine 		learning, kèm theo độ tin cậy để giúp đánh giá mức độ chắc chắn của dự đoán.
	44	Q: Tại sao độ tin cậy lại khác nhau giữa các cổ phiếu?
	45	A: Mỗi cổ phiếu có đặc điểm riêng về tính biến động, tính thanh khoản và các yếu tố ảnh 		hưởng. Một số cổ phiếu có thể dễ dự đoán hơn do chúng tuân theo xu hướng rõ ràng.
	46	Q: Làm thế nào để cải thiện độ chính xác của dự đoán?
	47	A: Chạy lệnh --evolve để tối ưu hóa mô hình, bổ sung thêm dữ liệu, và điều chỉnh tham số 		trong tệp cấu hình. Sử dụng module đánh giá hiệu suất để theo dõi và cải thiện hiệu suất 		dự đoán.
	48	Q: BondZiA có thể theo dõi bao nhiêu cổ phiếu cùng lúc?
	49	A: Về mặt kỹ thuật, BondZiA có thể theo dõi hàng trăm cổ phiếu, nhưng giới hạn API của 		Polygon.io và tài nguyên hệ thống có thể ảnh hưởng đến hiệu suất. Đề xuất theo dõi 10-15 		cổ phiếu để có hiệu suất tối ưu.
	50	Q: Làm thế nào để thêm module mới vào BondZiA?
	51	A: Thêm module mới vào thư mục tương ứng (ví dụ: utils/ hoặc evaluation/), sau đó cập 		nhật file main.py để tích hợp module mới. Tham khảo mẫu trong integration/ để biết cách 		tích hợp module mới.
	
	# requirements.txt
	
	54	Data processing
	55	numpy==1.23.5 pandas==2.0.3 pytz==2023.3 python-dateutil==2.8.2
	
	56	API interaction
	57	requests==2.31.0 websocket-client==1.6.1 polygon-api-client==1.12.3 yfinance==0.2.30 # Thêm cho EnhancedDataFetcher
	
	58	Data visualization
	59	matplotlib==3.7.2 seaborn==0.12.2 plotly==5.15.0 kaleido==0.2.1 streamlit==1.28.0 # Thêm cho Dashboard
	
	60	Machine Learning & Deep Learning
	61	scikit-learn==1.3.0 tensorflow==2.12.0 torch==2.0.1 transformers==4.30.2 pytorch-forecasting==1.0.0 statsmodels==0.14.0
	
	62	Reinforcement Learning
	63	gym==0.26.0 stable-baselines3==2.0.0
	6
	4	Time series specific
	65	darts==0.24.0 pmdarima==2.0.3 prophet==1.1.4 neuralforecast==1.6.3
	
	66	Tâm lý thị trường và tin tức
	67	beautifulsoup4==4.12.2 # Thêm cho phân tích tin tức pytrends==4.9.2 # Thêm cho Google Trends
	
	68	Utilities
	69	tqdm==4.65.0 joblib==1.3.2 pyyaml==6.0.1 schedule==1.2.0 discord-webhook==1.1.0 loguru==0.7.0
	
	70	Self-healing & monitoring
	71	psutil==5.9.5 watchdog==3.0.0
	
	72	Đánh giá hiệu suất
	
	73	streamlit==1.28.0 plotly==5.15.0
	
	74	Với hai file này, bạn có một README hoàn chỉnh mô tả chi tiết hệ thống BondZiA AI bao gồm tất cả các tính năng và cải tiến mới, cùng với file requirements.txt cập nhật chứa tất cả các thư viện cần thiết. README này cung cấp đầy đủ thông tin để người đọc hiểu toàn bộ hệ thống, cách sử dụng, và đặc biệt là các module mới thêm vào đánh giá hiệu suất. File requirements.txt cập nhật có thêm các thư viện mới cần thiết cho các tính năng đã thêm.
