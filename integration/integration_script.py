#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tập lệnh tích hợp các cải tiến vào hệ thống BondZiA
"""

import os
import sys
import json
import shutil
import argparse
import traceback
from datetime import datetime

# Đảm bảo chạy tập lệnh từ thư mục gốc của dự án
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# Import các thư viện cần thiết
from utils.logger_config import logger

def backup_original_files(backup_dir=None):
    """
    Sao lưu các file gốc trước khi tích hợp
    
    Args:
        backup_dir (str, optional): Thư mục sao lưu. Nếu None, tạo thư mục mới.
    
    Returns:
        str: Đường dẫn đến thư mục sao lưu
    """
    try:
        # Nếu không có thư mục sao lưu, tạo một thư mục mới
        if backup_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(root_dir, f"backup_{timestamp}")
        
        # Tạo thư mục sao lưu nếu chưa tồn tại
        os.makedirs(backup_dir, exist_ok=True)
        
        # Danh sách các file cần sao lưu
        files_to_backup = [
            os.path.join(root_dir, "models", "base_predictor.py"),
            os.path.join(root_dir, "utils", "data_fetcher.py"),
            os.path.join(root_dir, "utils", "visualization.py"),
            os.path.join(root_dir, "config", "system_config.json")
        ]
        
        # Sao lưu từng file
        for file_path in files_to_backup:
            if os.path.exists(file_path):
                # Tạo thư mục đích tương ứng
                rel_path = os.path.relpath(file_path, root_dir)
                backup_path = os.path.join(backup_dir, rel_path)
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                
                # Sao chép file
                shutil.copy2(file_path, backup_path)
                logger.info(f"Đã sao lưu {file_path} vào {backup_path}")
            else:
                logger.warning(f"Không tìm thấy file {file_path}")
        
        logger.info(f"Đã sao lưu các file gốc vào {backup_dir}")
        
        return backup_dir
    
    except Exception as e:
        logger.error(f"Lỗi khi sao lưu file gốc: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def copy_new_modules():
    """
    Sao chép các module mới vào thư mục thích hợp
    
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    try:
        # Danh sách các module mới và đường dẫn đích
        new_modules = {
            "enhanced_data_fetcher.py": os.path.join(root_dir, "utils", "enhanced_data_fetcher.py"),
            "ensemble_predictor.py": os.path.join(root_dir, "models", "ensemble_predictor.py"),
            "confidence_evaluator.py": os.path.join(root_dir, "utils", "confidence_evaluator.py")
        }
        
        # Danh sách các thư mục cần tạo
        required_dirs = [
            os.path.join(root_dir, "models", "ensemble"),
            os.path.join(root_dir, "models", "history"),
            os.path.join(root_dir, "prediction_history"),
            os.path.join(root_dir, "prediction_stats")
        ]
        
        # Tạo các thư mục cần thiết
        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Đã tạo thư mục {dir_path}")
        
        # Đường dẫn đến thư mục chứa các file mới
        source_dir = os.path.join(current_dir, "new_modules")
        
        # Kiểm tra xem thư mục source có tồn tại không
        if not os.path.exists(source_dir):
            # Tạo thư mục source nếu chưa tồn tại
            os.makedirs(source_dir, exist_ok=True)
            logger.warning(f"Thư mục {source_dir} không tồn tại, đã tạo mới")
        
        # Sao chép các module mới
        success_count = 0
        
        for source_file, dest_path in new_modules.items():
            source_path = os.path.join(source_dir, source_file)
            
            # Nếu file nguồn không tồn tại, tạo file từ string content
            if not os.path.exists(source_path):
                logger.warning(f"File nguồn {source_file} không tồn tại trong {source_dir}")
                continue
            
            # Sao chép file
            shutil.copy2(source_path, dest_path)
            logger.info(f"Đã sao chép {source_file} vào {dest_path}")
            success_count += 1
        
        if success_count == len(new_modules):
            logger.info("Đã sao chép tất cả các module mới thành công")
            return True
        else:
            logger.warning(f"Đã sao chép {success_count}/{len(new_modules)} module mới")
            return success_count > 0
    
    except Exception as e:
        logger.error(f"Lỗi khi sao chép module mới: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def update_config_file():
    """
    Cập nhật file cấu hình để hỗ trợ các tính năng mới
    
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    try:
        # Đường dẫn đến file cấu hình
        config_path = os.path.join(root_dir, "config", "system_config.json")
        
        # Đọc cấu hình hiện tại
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Tạo bản sao
        backup_config = config.copy()
        
        # Thêm cấu hình mới
        
        # 1. Bật/tắt sử dụng mô hình ensemble
        if 'use_ensemble' not in config:
            config['use_ensemble'] = True
        
        # 2. Cấu hình cho độ tin cậy
        if 'confidence' not in config:
            config['confidence'] = {
                'verify_interval_hours': 24,  # Kiểm tra độ chính xác mỗi 24 giờ
                'min_predictions_for_stats': 10,  # Số lượng dự đoán tối thiểu để tính thống kê
                'adjust_by_market_conditions': True,  # Điều chỉnh độ tin cậy theo điều kiện thị trường
                'store_history': True  # Lưu lịch sử dự đoán
            }
        
        # 3. Cấu hình cho nguồn dữ liệu nâng cao
        if 'data_sources' not in config:
            config['data_sources'] = {
                'use_enhanced_data': True,  # Sử dụng dữ liệu đa nguồn
                'use_yahoo_finance': True,  # Sử dụng Yahoo Finance
                'use_google_trends': False,  # Tắt Google Trends mặc định
                'use_news_sentiment': True,  # Sử dụng phân tích tâm lý tin tức
                'cache_timeout': {
                    'news': 6,  # Giờ
                    'trends': 24,  # Giờ
                    'economic_data': 24  # Giờ
                }
            }
        
        # 4. Thêm mô hình TimeGPT cho dự đoán 1 tháng
        if 'prediction' in config:
            if 'monthly' in config['prediction']:
                config['prediction']['monthly']['model_type'] = 'gpt_enhanced'
                config['prediction']['monthly']['use_economic_indicators'] = True
                
            # Kiểm tra nếu model_types không được định nghĩa
            for timeframe in ['intraday', 'five_day', 'monthly']:
                if timeframe in config['prediction'] and 'model_type' not in config['prediction'][timeframe]:
                    if timeframe == 'intraday':
                        config['prediction'][timeframe]['model_type'] = 'tft'  # Temporal Fusion Transformer
                    elif timeframe == 'five_day':
                        config['prediction'][timeframe]['model_type'] = 'lstm_attention'  # LSTM with Attention
                    elif timeframe == 'monthly':
                        config['prediction'][timeframe]['model_type'] = 'gpt_enhanced'  # TimeGPT enhanced
        
        # 5. Cấu hình đánh giá thị trường
        if 'market_evaluation' not in config:
            config['market_evaluation'] = {
                'enable': True,
                'adjust_confidence': True,
                'volatility_threshold': {
                    'low': 15,    # VIX < 15
                    'medium': 25, # VIX < 25
                    'high': 35    # VIX < 35
                }
            }
        
        # 6. Bật logging chi tiết hơn
        if 'logging' not in config:
            config['logging'] = {
                'prediction_detail_level': 'high',
                'confidence_detail_level': 'high',
                'save_prediction_charts': True
            }
        
        # Ghi lại cấu hình đã cập nhật
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # So sánh sự khác biệt
        diff_keys = set()
        
        def find_diff_keys(d1, d2, path=""):
            for k in d1:
                if k not in d2:
                    diff_keys.add(path + k)
                elif isinstance(d1[k], dict) and isinstance(d2[k], dict):
                    find_diff_keys(d1[k], d2[k], path + k + ".")
                elif d1[k] != d2[k]:
                    diff_keys.add(path + k)
            
            for k in d2:
                if k not in d1:
                    diff_keys.add(path + k)
        
        find_diff_keys(config, backup_config)
        
        logger.info(f"Đã cập nhật cấu hình thành công, với {len(diff_keys)} thay đổi")
        for key in diff_keys:
            logger.info(f"  - Thay đổi: {key}")
        
        return True
    
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật file cấu hình: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def replace_base_predictor():
    """
    Thay thế BasePredictor với phiên bản cải tiến
    
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    try:
        # Đường dẫn đến file BasePredictor
        base_predictor_path = os.path.join(root_dir, "models", "base_predictor.py")
        
        # Đường dẫn đến file BasePredictor mới
        new_base_predictor_path = os.path.join(current_dir, "new_modules", "base_predictor.py")
        
        # Kiểm tra xem file mới có tồn tại không
        if not os.path.exists(new_base_predictor_path):
            logger.error(f"Không tìm thấy file {new_base_predictor_path}")
            return False
        
        # Sao chép file mới đè lên file cũ
        shutil.copy2(new_base_predictor_path, base_predictor_path)
        logger.info(f"Đã thay thế {base_predictor_path} thành công")
        
        return True
    
    except Exception as e:
        logger.error(f"Lỗi khi thay thế BasePredictor: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def copy_specific_predictors():
    """
    Sao chép các lớp dự đoán cụ thể
    
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    try:
        # Đường dẫn đến thư mục models
        models_dir = os.path.join(root_dir, "models")
        
        # Đường dẫn đến file các lớp dự đoán cụ thể mới
        specific_predictors_path = os.path.join(current_dir, "new_modules", "specific_predictors.py")
        
        # Kiểm tra xem file mới có tồn tại không
        if not os.path.exists(specific_predictors_path):
            logger.error(f"Không tìm thấy file {specific_predictors_path}")
            return False
        
        # Sao chép file mới vào thư mục models
        shutil.copy2(specific_predictors_path, os.path.join(models_dir, "specific_predictors.py"))
        logger.info(f"Đã sao chép các lớp dự đoán cụ thể thành công")
        
        return True
    
    except Exception as e:
        logger.error(f"Lỗi khi sao chép các lớp dự đoán cụ thể: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def check_dependencies():
    """
    Kiểm tra và cài đặt các gói phụ thuộc cần thiết
    
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    try:
        # Danh sách các gói cần thiết
        required_packages = [
            "yfinance",
            "pytrends",
            "tensorflow>=2.0.0",
            "scikit-learn>=0.24.0",
            "pandas>=1.0.0",
            "numpy>=1.18.0",
            "matplotlib>=3.0.0",
            "seaborn>=0.11.0",
            "beautifulsoup4>=4.9.0",
            "requests>=2.0.0"
        ]
        
        # Kiểm tra và cài đặt các gói
        missing_packages = []
        
        for package in required_packages:
            try:
                package_name = package.split(">=")[0]
                __import__(package_name)
                logger.info(f"Gói {package_name} đã được cài đặt")
            except ImportError:
                missing_packages.append(package)
        
        # Nếu có gói bị thiếu, cài đặt
        if missing_packages:
            logger.warning(f"Các gói sau chưa được cài đặt: {', '.join(missing_packages)}")
            
            # Hỏi người dùng có muốn cài đặt không
            response = input("Bạn có muốn cài đặt các gói còn thiếu không? (y/n): ").lower()
            
            if response == 'y':
                import subprocess
                
                for package in missing_packages:
                    logger.info(f"Đang cài đặt {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    logger.info(f"Đã cài đặt {package} thành công")
            else:
                logger.warning("Bỏ qua cài đặt gói, một số tính năng có thể không hoạt động")
        
        return True
    
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra và cài đặt gói phụ thuộc: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def create_module_files():
    """
    Tạo các file module mới nếu không tồn tại
    
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    try:
        # Tạo thư mục chứa file mới nếu chưa tồn tại
        new_modules_dir = os.path.join(current_dir, "new_modules")
        os.makedirs(new_modules_dir, exist_ok=True)
        
        # Tạo các file module từ string
        module_contents = {
            "enhanced_data_fetcher.py": enhanced_data_fetcher_content,
            "ensemble_predictor.py": ensemble_predictor_content,
            "confidence_evaluator.py": confidence_evaluator_content,
            "base_predictor.py": base_predictor_content,
            "specific_predictors.py": specific_predictors_content
        }
        
        for filename, content in module_contents.items():
            file_path = os.path.join(new_modules_dir, filename)
            
            # Chỉ tạo file nếu chưa tồn tại
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    f.write(content)
                logger.info(f"Đã tạo file {filename}")
        
        return True
    
    except Exception as e:
        logger.error(f"Lỗi khi tạo file module: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """
    Hàm chính để tích hợp các cải tiến
    """
    parser = argparse.ArgumentParser(description="Tích hợp các cải tiến vào hệ thống BondZiA")
    parser.add_argument('--skip-backup', action='store_true', help='Bỏ qua sao lưu các file gốc')
    parser.add_argument('--skip-config', action='store_true', help='Bỏ qua cập nhật cấu hình')
    parser.add_argument('--skip-dependencies', action='store_true', help='Bỏ qua kiểm tra và cài đặt gói phụ thuộc')
    
    args = parser.parse_args()
    
    # Hiển thị thông tin
    print("=== BondZiA AI - Tích hợp cải tiến ===")
    print("Thời gian: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Thư mục gốc: " + root_dir)
    print("=" * 40)
    
    # Kiểm tra và cài đặt gói phụ thuộc
    if not args.skip_dependencies:
        print("\n[1/6] Kiểm tra và cài đặt gói phụ thuộc...")
        if check_dependencies():
            print("  ✓ Đã kiểm tra và cài đặt gói phụ thuộc thành công")
        else:
            print("  ✗ Lỗi khi kiểm tra và cài đặt gói phụ thuộc")
            return
    
    # Tạo các file module mới
    print("\n[2/6] Tạo các file module mới...")
    if create_module_files():
        print("  ✓ Đã tạo các file module mới thành công")
    else:
        print("  ✗ Lỗi khi tạo các file module mới")
        return
    
    # Sao lưu các file gốc
    if not args.skip_backup:
        print("\n[3/6] Sao lưu các file gốc...")
        backup_dir = backup_original_files()
        if backup_dir:
            print(f"  ✓ Đã sao lưu các file gốc vào {backup_dir}")
        else:
            print("  ✗ Lỗi khi sao lưu các file gốc")
            return
    
    # Cập nhật cấu hình
    if not args.skip_config:
        print("\n[4/6] Cập nhật cấu hình...")
        if update_config_file():
            print("  ✓ Đã cập nhật cấu hình thành công")
        else:
            print("  ✗ Lỗi khi cập nhật cấu hình")
            return
    
    # Thay thế BasePredictor
    print("\n[5/6] Thay thế BasePredictor...")
    if replace_base_predictor():
        print("  ✓ Đã thay thế BasePredictor thành công")
    else:
        print("  ✗ Lỗi khi thay thế BasePredictor")
        return
    
    # Sao chép các module mới
    print("\n[6/6] Sao chép các module mới...")
    if copy_new_modules() and copy_specific_predictors():
        print("  ✓ Đã sao chép các module mới thành công")
    else:
        print("  ✗ Lỗi khi sao chép các module mới")
        return
    
    # Hoàn tất
    print("\n" + "=" * 40)
    print("✅ Đã tích hợp cải tiến thành công!")
    print("Để áp dụng thay đổi, vui lòng khởi động lại BondZiA AI.")
    print("=" * 40)

if __name__ == "__main__":
    # Định nghĩa nội dung cho các module
    # Đây là nơi lưu trữ nội dung string của các file module mới
    
    # Nội dung file enhanced_data_fetcher.py
    enhanced_data_fetcher_content = """
# Nội dung file enhanced_data_fetcher.py
# Đây là placeholder, nội dung thực sẽ được thay thế khi triển khai
"""
    
    # Nội dung file ensemble_predictor.py
    ensemble_predictor_content = """
# Nội dung file ensemble_predictor.py
# Đây là placeholder, nội dung thực sẽ được thay thế khi triển khai
"""
    
    # Nội dung file confidence_evaluator.py
    confidence_evaluator_content = """
# Nội dung file confidence_evaluator.py
# Đây là placeholder, nội dung thực sẽ được thay thế khi triển khai
"""
    
    # Nội dung file base_predictor.py
    base_predictor_content = """
# Nội dung file base_predictor.py
# Đây là placeholder, nội dung thực sẽ được thay thế khi triển khai
"""
    
    # Nội dung file specific_predictors.py
    specific_predictors_content = """
# Nội dung file specific_predictors.py
# Đây là placeholder, nội dung thực sẽ được thay thế khi triển khai
"""
    
    main()