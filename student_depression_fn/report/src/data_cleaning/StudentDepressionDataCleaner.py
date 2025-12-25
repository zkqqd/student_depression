import pandas as pd
import numpy as np
import re
class StudentDepressionDataCleaner:
  
    def __init__(self, data):
        """
        Khởi tạo class với dữ liệu
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Dữ liệu gốc cần làm sạch
        """
        self.original_data = data.copy()
        self.cleaned_data = data.copy()
        self.categorical_columns = [
            'Gender', 'City', 'Profession', 'Degree',
            'Have you ever had suicidal thoughts ?', 
            'Family History of Mental Illness'
        ]
        
    def convert_depression_to_int(self):
        print("1. Chuyển đổi cột 'Depression' sang integer...")
        self.cleaned_data['Depression'] = self.cleaned_data['Depression'].astype(int)
        print("   ✓ Hoàn thành")
        return self
    
    def convert_to_categorical(self):
        #Chuyển đổi các cột phân loại sang kiểu category
        
        print("\n2. Chuyển đổi các cột phân loại sang category...")
        for col in self.categorical_columns:
            if col in self.cleaned_data.columns:
                self.cleaned_data[col] = self.cleaned_data[col].astype('category')
                print(f"   ✓ {col}: {self.cleaned_data[col].dtype}")
        return self
    
    def check_unique_values(self):
        #Kiểm tra giá trị duy nhất trong các cột quan trọng

        print("\n3. Kiểm tra giá trị duy nhất:")
        print("-" * 40)
        
        print("   'Sleep Duration':", self.original_data['Sleep Duration'].unique()[:10], 
              "..." if len(self.original_data['Sleep Duration'].unique()) > 10 else "")
        print("   'Financial Stress':", self.original_data['Financial Stress'].unique()[:10], 
              "..." if len(self.original_data['Financial Stress'].unique()) > 10 else "")
        
        return self
    
    @staticmethod
    def extract_hours(s):
        """
        Trích xuất số giờ từ chuỗi Sleep Duration
        
        Parameters:
        -----------
        s : str
            Chuỗi chứa thông tin thời gian ngủ
            
        Returns:
        --------
        float or np.nan
            Số giờ đã được trích xuất
        """
        # Tìm một số (bao gồm các số thập phân)
        match = re.search(r"(\d+(\.\d+)?)", str(s))
        return float(match.group(1)) if match else np.nan
    
    def clean_sleep_duration(self):
        """
        Làm sạch cột Sleep Duration
        """
        print("\n4. Làm sạch cột 'Sleep Duration'...")
        
        # Áp dụng hàm extract_hours
        self.cleaned_data['Sleep Duration'] = self.cleaned_data['Sleep Duration'].apply(
            self.extract_hours
        )
        
        # Hiển thị kết quả
        print(f"   ✓ Đã chuyển đổi sang numeric")
        print(f"   ✓ Giá trị min: {self.cleaned_data['Sleep Duration'].min():.1f} giờ")
        print(f"   ✓ Giá trị max: {self.cleaned_data['Sleep Duration'].max():.1f} giờ")
        print(f"   ✓ Giá trị trung bình: {self.cleaned_data['Sleep Duration'].mean():.1f} giờ")
        
        return self
    
    def clean_financial_stress(self):
        """
        Chuyển đổi Financial Stress sang categorical
        """
        print("\n5. Chuyển đổi 'Financial Stress' sang category...")
        
        if 'Financial Stress' in self.cleaned_data.columns:
            self.cleaned_data['Financial Stress'] = self.cleaned_data['Financial Stress'].astype('category')
            print(f"   ✓ Hoàn thành")
            print(f"   ✓ Số lượng category: {len(self.cleaned_data['Financial Stress'].cat.categories)}")
        else:
            print("   ✗ Không tìm thấy cột 'Financial Stress'")
            
        return self
    
    def check_missing_values(self):
        #Kiểm tra và hiển thị giá trị thiếu
        
        print("\n6. Kiểm tra giá trị thiếu:")
        print("-" * 40)
        
        missing_values = self.cleaned_data.isnull().sum()
        missing_columns = missing_values[missing_values > 0]
        
        if len(missing_columns) > 0:
            print("   Các cột có giá trị thiếu:")
            for col, count in missing_columns.items():
                percentage = (count / len(self.cleaned_data)) * 100
                print(f"   • {col}: {count} giá trị ({percentage:.1f}%)")
        else:
            print("   ✓ Không có giá trị thiếu")
            
        return self
    
    def impute_missing_values(self):
        """
        Thay thế giá trị thiếu bằng median cho các cột numeric
        """
        print("\n7. Xử lý giá trị thiếu...")
        
        numeric_columns = self.cleaned_data.select_dtypes(include=[np.number]).columns
        imputed_columns = []
        
        for col in numeric_columns:
            missing_count = self.cleaned_data[col].isnull().sum()
            if missing_count > 0:
                median_value = self.cleaned_data[col].median()
                self.cleaned_data[col].fillna(median_value, inplace=True)
                imputed_columns.append((col, missing_count, median_value))
                
        if imputed_columns:
            print("   Đã thay thế giá trị thiếu:")
            for col, count, median in imputed_columns:
                print(f"   • {col}: {count} giá trị → median = {median:.2f}")
        else:
            print("   ✓ Không có giá trị thiếu cần xử lý")
            
        return self
    
    def verify_changes(self):
        #Kiểm tra kết quả sau khi làm sạch
        
        print("\n8. Kiểm tra kết quả làm sạch:")
        print("-" * 40)
        
        # Hiển thị 5 dòng đầu tiên của các cột quan trọng
        print("   Sleep Duration và Financial Stress:")
        print(self.cleaned_data[['Sleep Duration', 'Financial Stress']].head().to_string())
        
        # Thông tin tổng quan
        print(f"\n   Tổng số dòng: {self.cleaned_data.shape[0]}")
        print(f"   Tổng số cột: {self.cleaned_data.shape[1]}")
        
        return self
    
    def get_summary(self):
        """
        Tạo báo cáo tổng quan về dữ liệu đã làm sạch
        """
        summary = {
            'original_shape': self.original_data.shape,
            'cleaned_shape': self.cleaned_data.shape,
            'dtypes': self.cleaned_data.dtypes.to_dict(),
            'categorical_columns': list(self.cleaned_data.select_dtypes(include=['category']).columns),
            'numeric_columns': list(self.cleaned_data.select_dtypes(include=[np.number]).columns),
            'missing_values': self.cleaned_data.isnull().sum().sum()
        }
        return summary
    
    def run_full_cleaning(self):
        """
        Chạy toàn bộ quy trình làm sạch
        """
        print("=" * 60)
        print("BẮT ĐẦU QUY TRÌNH LÀM SẠCH DỮ LIỆU")
        print("=" * 60)
        
        # Thực hiện tuần tự các bước làm sạch
        (self.convert_depression_to_int()
         .convert_to_categorical()
         .check_unique_values()
         .clean_sleep_duration()
         .clean_financial_stress()
         .check_missing_values()
         .impute_missing_values()
         .verify_changes())
        
        print("\n" + "=" * 60)
        print("HOÀN THÀNH QUY TRÌNH LÀM SẠCH")
        print("=" * 60)
        
        return self.cleaned_data
    
    def save_cleaned_data(self, filename='student_depression_cleaned.csv'):
        """
        Lưu dữ liệu đã làm sạch vào file CSV
        
        Parameters:
        -----------
        filename : str
            Tên file để lưu dữ liệu đã làm sạch
        """
        self.cleaned_data.to_csv(filename, index=False)
        print(f"\n✓ Dữ liệu đã làm sạch được lưu vào: {filename}")


# Class để quản lý việc đọc và xử lý file
class DepressionDataProcessor:
    """
    Class để xử lý toàn bộ pipeline đọc và làm sạch dữ liệu
    """
    
    def __init__(self, filepath='student_depression_.csv'):
        """
        Khởi tạo processor với đường dẫn file
        
        Parameters:
        -----------
        filepath : str
            Đường dẫn đến file CSV
        """
        self.filepath = filepath
        self.data = None
        self.cleaner = None
        
    def load_data(self):
        """
        Đọc dữ liệu từ file CSV
        """
        print(f"Đang đọc dữ liệu từ: {self.filepath}")
        
        try:
            self.data = pd.read_csv(self.filepath)
            print(f"✓ Đã đọc dữ liệu: {self.data.shape[0]} dòng, {self.data.shape[1]} cột")
            return True
        except FileNotFoundError:
            print(f"✗ Lỗi: Không tìm thấy file {self.filepath}")
            return False
        except Exception as e:
            print(f"✗ Lỗi khi đọc file: {str(e)}")
            return False
    
    def process_data(self):
        """
        Xử lý và làm sạch dữ liệu
        """
        if self.data is not None:
            # Tạo instance của cleaner
            self.cleaner = StudentDepressionDataCleaner(self.data)
            
            # Chạy quy trình làm sạch
            cleaned_data = self.cleaner.run_full_cleaning()
            
            # Lưu dữ liệu đã làm sạch
            self.cleaner.save_cleaned_data()
            
            return cleaned_data
        else:
            print("✗ Chưa có dữ liệu để xử lý. Vui lòng load_data() trước.")
            return None
    
    def get_cleaner_summary(self):
        """
        Lấy báo cáo tổng quan từ cleaner
        """
        if self.cleaner:
            return self.cleaner.get_summary()
        else:
            print("✗ Chưa có cleaner. Vui lòng chạy process_data() trước.")
            return None


