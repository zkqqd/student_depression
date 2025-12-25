import pandas as pd
import re
import numpy as np

#Hàm lọc độ tuổi từ 18-25 sinh viên:
def filter_age_18_25(df, age_column='Age'):
    original_count = len(df)
    df = df[(df[age_column] >= 18) & (df[age_column] <= 25)].copy()
    filtered_count = len(df)
    removed_count = original_count - filtered_count
    print(f"✅ Đã lọc độ tuổi (18-25):")
    print(f"   - Số bản ghi gốc: {original_count}")
    print(f"   - Số bản ghi sau lọc: {filtered_count}")
    print(f"   - Đã loại bỏ: {removed_count} bản ghi")
    print(f"   - Tỷ lệ giữ lại: {(filtered_count/original_count)*100:.1f}%")
    return df

#Hàm xóa các ô dữ liệu chứa dấu ' '
def clean_apostrophe(df):
    """
    Xử lý các ô dữ liệu chứa dấu nháy đơn (')
    - Thay thế bằng ký tự an toàn hoặc xóa
    - Đặc biệt quan trọng với cột 'Degree' và các cột text khác
    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame cần làm sạch
    
    Returns:
    --------
    DataFrame đã được xử lý dấu '
    """
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace("'", "", regex=False)
    return df

#Hàm chuyển đổi Depression sang integer
def convert_depression_to_int(df):
    """Chuyển cột Depression sang kiểu integer"""
    df['Depression'] = df['Depression'].astype(int)
    return df

#Hàm chuyển các cột sang categorical
def convert_to_categorical(df):
    """Chuyển các cột phân loại sang category"""
    cat_cols = ['Gender', 'City', 'Profession', 'Degree',
                'Have you ever had suicidal thoughts ?', 
                'Family History of Mental Illness']
    for col in cat_cols:
        df[col] = df[col].astype('category')
    return df

#Hàm kiểm tra giá trị duy nhất
def check_unique_values(df):
    """Hiển thị giá trị duy nhất trong các cột quan trọng"""
    print("Unique values in 'Sleep Duration':", df['Sleep Duration'].unique())
    print("Unique values in 'Financial Stress':", df['Financial Stress'].unique())
    return df

#Hàm trích xuất giờ từ Sleep Duration
def extract_hours(s):
    """Trích xuất số giờ từ chuỗi"""
    match = re.search(r"(\d+(\.\d+)?)", str(s))
    return float(match.group(1)) if match else np.nan

def clean_sleep_duration(df):
    """Làm sạch cột Sleep Duration"""
    df['Sleep Duration'] = df['Sleep Duration'].apply(extract_hours)
    return df

#Hàm làm sạch Financial Stress
def clean_financial_stress(df):
    """Chuyển Financial Stress sang category"""
    df['Financial Stress'] = df['Financial Stress'].astype('category')
    return df

#Hàm kiểm tra giá trị thiếu
def check_missing_values(df):
    """Hiển thị số lượng giá trị thiếu"""
    missing = df.isnull().sum()
    print("Missing values:\n", missing[missing > 0])
    return df

#Hàm thay thế giá trị thiếu
def impute_missing_values(df):
    """Thay thế giá trị thiếu bằng median"""
    for col in ['Sleep Duration']:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    return df

#Hàm kiểm tra kết quả
def verify_cleaning(df):
    """Kiểm tra kết quả sau khi làm sạch"""
    print(df[['Sleep Duration', 'Financial Stress']].head())
    return df

#Hàm chạy toàn bộ quy trình làm sạch
def run_full_cleaning(file_path):
    """Chạy tất cả các bước làm sạch"""
    print("Bắt đầu quy trình làm sạch dữ liệu...")
    df = pd.read_csv(file_path) 
    df = filter_age_18_25(df, age_column='Age')
    df = clean_apostrophe(df)
    df = convert_depression_to_int(df)
    df = convert_to_categorical(df)
    df = check_unique_values(df)
    df = clean_sleep_duration(df)
    df = clean_financial_stress(df)
    df = check_missing_values(df)
    df = impute_missing_values(df)
    df = verify_cleaning(df)
    
    print("\n✅ Hoàn thành quy trình làm sạch!")
    return df

