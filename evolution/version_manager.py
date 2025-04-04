import os
import sys
import json
import shutil
from datetime import datetime
import traceback
from utils.logger_config import logger

# Thêm thư mục gốc vào PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class VersionManager:
    """Lớp quản lý phiên bản của BondZiA AI"""
    
    def __init__(self, config_path="../config/system_config.json"):
        """
        Khởi tạo VersionManager
        
        Args:
            config_path (str): Đường dẫn đến file cấu hình
        """
        # Đọc cấu hình
        self.config_path = config_path
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Đường dẫn đến thư mục phiên bản
        self.versions_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "BondZiA_versions")
        os.makedirs(self.versions_dir, exist_ok=True)
        
        # Số lượng phiên bản tối đa cần giữ
        # Sử dụng giá trị mặc định nếu không tìm thấy khóa
        self.max_versions = self.config.get('system', {}).get('max_versions_to_keep', 5)
        
        logger.info(f"Khởi tạo VersionManager với max_versions={self.max_versions}")
    
    def get_current_version(self):
        """
        Lấy phiên bản hiện tại
        
        Returns:
            str: Phiên bản hiện tại
        """
        return self.config['system']['version']
    
    def get_all_versions(self):
        """
        Lấy danh sách tất cả các phiên bản
        
        Returns:
            list: Danh sách các phiên bản
        """
        versions = []
        
        # Tìm tất cả thư mục BondZiA_v*
        for item in os.listdir(self.versions_dir):
            if os.path.isdir(os.path.join(self.versions_dir, item)) and item.startswith("BondZiA_v"):
                # Trích xuất phiên bản từ tên thư mục
                version = item[len("BondZiA_v"):]
                versions.append(version)
        
        # Sắp xếp theo phiên bản
        versions.sort(key=lambda v: [int(x) for x in v.split('.')])
        
        return versions
    
    def create_new_version(self, version=None):
        """
        Tạo phiên bản mới
        
        Args:
            version (str, optional): Phiên bản mới. Nếu None, tự động tăng số phiên bản.
            
        Returns:
            str: Đường dẫn đến thư mục phiên bản mới
        """
        try:
            # Nếu không chỉ định phiên bản, tự động tăng số phiên bản
            if version is None:
                current_version = self.get_current_version()
                
                # Phân tích phiên bản
                parts = current_version.split('.')
                major, minor, patch = map(int, parts)
                
                # Tăng số phiên bản patch
                patch += 1
                
                # Phiên bản mới
                version = f"{major}.{minor}.{patch}"
            
            # Tạo thư mục phiên bản mới
            version_dir = os.path.join(self.versions_dir, f"BondZiA_v{version}")
            os.makedirs(version_dir, exist_ok=True)
            
            # Cập nhật cấu hình
            self.config['system']['version'] = version
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Đã tạo phiên bản mới {version} tại {version_dir}")
            
            # Dọn dẹp các phiên bản cũ
            self._cleanup_old_versions()
            
            return version_dir
        except Exception as e:
            logger.error(f"Lỗi khi tạo phiên bản mới: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _cleanup_old_versions(self):
        """Dọn dẹp các phiên bản cũ"""
        try:
            versions = self.get_all_versions()
            
            # Nếu số lượng phiên bản vượt quá giới hạn
            if len(versions) > self.max_versions:
                # Xóa các phiên bản cũ
                versions_to_delete = versions[:len(versions) - self.max_versions]
                
                for version in versions_to_delete:
                    version_dir = os.path.join(self.versions_dir, f"BondZiA_v{version}")
                    
                    if os.path.exists(version_dir):
                        shutil.rmtree(version_dir)
                        logger.info(f"Đã xóa phiên bản cũ {version}")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi dọn dẹp phiên bản cũ: {str(e)}")
            return False
    
    def backup_current_state(self):
        """
        Sao lưu trạng thái hiện tại vào phiên bản hiện tại
        
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            current_version = self.get_current_version()
            version_dir = os.path.join(self.versions_dir, f"BondZiA_v{current_version}")
            
            # Đảm bảo thư mục tồn tại
            os.makedirs(version_dir, exist_ok=True)
            
            # Sao lưu các file cấu hình
            config_dir = os.path.dirname(self.config_path)
            backup_config_dir = os.path.join(version_dir, "config")
            os.makedirs(backup_config_dir, exist_ok=True)
            
            for file in os.listdir(config_dir):
                if file.endswith(".json"):
                    source_file = os.path.join(config_dir, file)
                    dest_file = os.path.join(backup_config_dir, file)
                    shutil.copy2(source_file, dest_file)
            
            # Sao lưu các mô hình
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            backup_models_dir = os.path.join(version_dir, "models")
            
            if os.path.exists(models_dir):
                # Sao chép tất cả các mô hình
                shutil.copytree(models_dir, backup_models_dir, dirs_exist_ok=True)
            
            # Tạo file timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(os.path.join(version_dir, "backup_timestamp.txt"), 'w') as f:
                f.write(f"Backup created at: {timestamp}\n")
                f.write(f"Version: {current_version}\n")
            
            logger.info(f"Đã sao lưu trạng thái hiện tại vào phiên bản {current_version}")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi sao lưu trạng thái hiện tại: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def restore_version(self, version=None):
        """
        Khôi phục từ một phiên bản
        
        Args:
            version (str, optional): Phiên bản cần khôi phục. Nếu None, khôi phục phiên bản mới nhất.
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            # Nếu không chỉ định phiên bản, sử dụng phiên bản mới nhất
            if version is None:
                versions = self.get_all_versions()
                
                if not versions:
                    logger.error("Không có phiên bản nào để khôi phục")
                    return False
                
                version = versions[-1]
            
            # Kiểm tra xem phiên bản tồn tại không
            version_dir = os.path.join(self.versions_dir, f"BondZiA_v{version}")
            
            if not os.path.exists(version_dir):
                logger.error(f"Phiên bản {version} không tồn tại")
                return False
            
            # Sao lưu phiên bản hiện tại trước khi khôi phục
            self.backup_current_state()
            
            # Khôi phục cấu hình
            backup_config_dir = os.path.join(version_dir, "config")
            config_dir = os.path.dirname(self.config_path)
            
            if os.path.exists(backup_config_dir):
                for file in os.listdir(backup_config_dir):
                    if file.endswith(".json"):
                        source_file = os.path.join(backup_config_dir, file)
                        dest_file = os.path.join(config_dir, file)
                        shutil.copy2(source_file, dest_file)
            
            # Khôi phục các mô hình
            backup_models_dir = os.path.join(version_dir, "models")
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            
            if os.path.exists(backup_models_dir):
                # Xóa thư mục models hiện tại nếu tồn tại
                if os.path.exists(models_dir):
                    shutil.rmtree(models_dir)
                
                # Sao chép từ bản sao lưu
                shutil.copytree(backup_models_dir, models_dir, dirs_exist_ok=True)
            
            # Cập nhật cấu hình với phiên bản đã khôi phục
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            self.config['system']['version'] = version
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Đã khôi phục thành công phiên bản {version}")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi khôi phục phiên bản {version}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def compare_versions(self, version1, version2):
        """
        So sánh hai phiên bản
        
        Args:
            version1 (str): Phiên bản 1
            version2 (str): Phiên bản 2
            
        Returns:
            dict: Thông tin so sánh
        """
        try:
            # Kiểm tra tồn tại
            version1_dir = os.path.join(self.versions_dir, f"BondZiA_v{version1}")
            version2_dir = os.path.join(self.versions_dir, f"BondZiA_v{version2}")
            
            if not os.path.exists(version1_dir) or not os.path.exists(version2_dir):
                logger.error(f"Một trong các phiên bản không tồn tại")
                return None
            
            comparison = {
                'version1': version1,
                'version2': version2,
                'config_differences': {},
                'model_differences': {
                    'added': [],
                    'removed': [],
                    'modified': []
                }
            }
            
            # So sánh cấu hình
            config1_dir = os.path.join(version1_dir, "config")
            config2_dir = os.path.join(version2_dir, "config")
            
            if os.path.exists(config1_dir) and os.path.exists(config2_dir):
                # Lấy danh sách file cấu hình
                config_files1 = {f for f in os.listdir(config1_dir) if f.endswith(".json")}
                config_files2 = {f for f in os.listdir(config2_dir) if f.endswith(".json")}
                
                # Các file chung
                common_files = config_files1.intersection(config_files2)
                
                for file in common_files:
                    file1_path = os.path.join(config1_dir, file)
                    file2_path = os.path.join(config2_dir, file)
                    
                    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
                        config1 = json.load(f1)
                        config2 = json.load(f2)
                    
                    # So sánh cấu hình
                    comparison['config_differences'][file] = self._compare_configs(config1, config2)
            
            # So sánh mô hình
            models1_dir = os.path.join(version1_dir, "models")
            models2_dir = os.path.join(version2_dir, "models")
            
            if os.path.exists(models1_dir) and os.path.exists(models2_dir):
                # Lấy danh sách mô hình trong các phiên bản
                models1 = set()
                models2 = set()
                
                for root, _, files in os.walk(models1_dir):
                    for file in files:
                        if file.endswith(".h5"):
                            rel_path = os.path.relpath(os.path.join(root, file), models1_dir)
                            models1.add(rel_path)
                
                for root, _, files in os.walk(models2_dir):
                    for file in files:
                        if file.endswith(".h5"):
                            rel_path = os.path.relpath(os.path.join(root, file), models2_dir)
                            models2.add(rel_path)
                
                # Tìm các mô hình đã thêm, xóa và sửa đổi
                comparison['model_differences']['added'] = list(models2 - models1)
                comparison['model_differences']['removed'] = list(models1 - models2)
                
                # Các mô hình chung, kiểm tra sửa đổi
                common_models = models1.intersection(models2)
                
                for model in common_models:
                    file1_path = os.path.join(models1_dir, model)
                    file2_path = os.path.join(models2_dir, model)
                    
                    # Kiểm tra kích thước file hoặc thời gian sửa đổi
                    if os.path.getsize(file1_path) != os.path.getsize(file2_path) or \
                       os.path.getmtime(file1_path) != os.path.getmtime(file2_path):
                        comparison['model_differences']['modified'].append(model)
            
            return comparison
        except Exception as e:
            logger.error(f"Lỗi khi so sánh phiên bản: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _compare_configs(self, config1, config2, path=""):
        """
        So sánh hai cấu hình
        
        Args:
            config1 (dict): Cấu hình 1
            config2 (dict): Cấu hình 2
            path (str): Đường dẫn hiện tại trong cấu hình
            
        Returns:
            list: Danh sách các khác biệt
        """
        differences = []
        
        # Kiểm tra các khóa trong config1
        for key in config1:
            current_path = f"{path}.{key}" if path else key
            
            if key not in config2:
                differences.append({
                    'path': current_path,
                    'type': 'removed',
                    'value1': config1[key],
                    'value2': None
                })
            elif isinstance(config1[key], dict) and isinstance(config2[key], dict):
                # Đệ quy cho các dict
                nested_diffs = self._compare_configs(config1[key], config2[key], current_path)
                differences.extend(nested_diffs)
            elif config1[key] != config2[key]:
                differences.append({
                    'path': current_path,
                    'type': 'modified',
                    'value1': config1[key],
                    'value2': config2[key]
                })
        
        # Kiểm tra các khóa mới trong config2
        for key in config2:
            current_path = f"{path}.{key}" if path else key
            
            if key not in config1:
                differences.append({
                    'path': current_path,
                    'type': 'added',
                    'value1': None,
                    'value2': config2[key]
                })
        
        return differences

if __name__ == "__main__":
    # Test module
    logger.info("Kiểm tra module VersionManager")
    
    version_manager = VersionManager()
    
    # Lấy phiên bản hiện tại
    current_version = version_manager.get_current_version()
    logger.info(f"Phiên bản hiện tại: {current_version}")
    
    # Lấy tất cả các phiên bản
    all_versions = version_manager.get_all_versions()
    logger.info(f"Tất cả các phiên bản: {all_versions}")
    
    # Sao lưu trạng thái hiện tại
    backup_result = version_manager.backup_current_state()
    logger.info(f"Sao lưu trạng thái hiện tại: {'Thành công' if backup_result else 'Thất bại'}")
    
    # Tạo phiên bản mới
    new_version_dir = version_manager.create_new_version()
    logger.info(f"Tạo phiên bản mới tại: {new_version_dir}")
    
    # Lấy tất cả các phiên bản sau khi tạo mới
    all_versions = version_manager.get_all_versions()
    logger.info(f"Tất cả các phiên bản sau khi tạo mới: {all_versions}")
