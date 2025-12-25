import pandas as pd

def normalize_minmax(df, col):
    """Chuẩn hóa cột về [0, 1]"""
    df = df.copy()
    min_val = df[col].min()
    max_val = df[col].max()
    if max_val > min_val:
        df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
    return df

def standardize_zscore(df, col):
    """Chuẩn hóa cột về mean=0, std=1"""
    df = df.copy()
    mean_val = df[col].mean()
    std_val = df[col].std()
    if std_val > 0:
        df[f'{col}_std'] = (df[col] - mean_val) / std_val
    return df

def encode_yesno_to_binary(df, col):
    """Chuyển Yes/No thành 1/0"""
    df = df.copy()
    df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df

def encode_sleep_hours(df, col='Sleep Duration'):
    """Chuyển giờ ngủ thành số"""
    df = df.copy()
    
    df[col] = df[col].astype(str).str.strip()
    
    sleep_map = {
        'Less than 5 hours': 4,
        '5-6 hours': 5.5,
        '7-8 hours': 7.5,
        'More than 8 hours': 9,
        'Nan': 7.5,  # Giá trị mặc định
        'None': 7.5
    }
    
    # Áp dụng mapping
    df['sleep_hours'] = df[col].map(sleep_map)
    
    # Điền các giá trị thiếu bằng trung bình
    if df['sleep_hours'].isnull().any():
        avg_sleep = df['sleep_hours'].mean()
        df['sleep_hours'] = df['sleep_hours'].fillna(avg_sleep)
        print(f"     Đã điền {df['sleep_hours'].isnull().sum()} giá trị bằng trung bình: {avg_sleep:.2f}")
        
    return df

def encode_diet_score(df, col='Dietary Habits'):
    """Chuyển thói quen ăn uống thành điểm"""
    df = df.copy()
    diet_map = {'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3}
    df['diet_score'] = df[col].map(diet_map)
    return df

def encode_pressure_level(df, col='Academic Pressure'):
    """Chuyển mức áp lực thành số"""
    df = df.copy()
    pressure_map = {'Low': 1, 'Medium': 2, 'High': 3}
    df[col] = df[col].astype(str).str.strip().map(pressure_map)
    
    # Điền các giá trị không hợp lệ bằng trung bình
    if df[col].isnull().any():
        df[col] = df[col].fillna(2)  # Mặc định Medium = 2
    
    return df

def encode_financial_stress(df, col='Financial Stress'):
    """Chuyển áp lực tài chính Yes/No thành 0/1"""
    df = df.copy()
    stress_map = {'No': 0, 'Yes': 1}
    df[col] = df[col].astype(str).str.strip().map(stress_map)
    
    # Điền các giá trị không hợp lệ bằng 0
    if df[col].isnull().any():
        df[col] = df[col].fillna(0)
    
    return df

def normalize_dataset(df):
    """Chuẩn hóa toàn bộ dataset với các bước cần thiết"""
    df = df.copy()
    
    print(f"Bắt đầu chuẩn hóa dữ liệu...")
    print(f"   Shape ban đầu: {df.shape}")
    
    # Kiểm tra missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"⚠️  Có missing values:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"   - {col}: {count} missing")
    
    # 0. Mã hóa cột Academic Pressure từ text thành số
    if 'Academic Pressure' in df.columns:
        print(f"   Mã hóa cột: Academic Pressure")
        df = encode_pressure_level(df, 'Academic Pressure')
    
    # 0. Mã hóa cột Financial Stress từ Yes/No thành số
    if 'Financial Stress' in df.columns:
        print(f"   Mã hóa cột: Financial Stress")
        df = encode_financial_stress(df, 'Financial Stress')
    
    # 1. Chuẩn hóa cột số
    numeric_cols = ['CGPA', 'Academic Pressure', 'Work/Study Hours', 'Financial Stress']
    for col in numeric_cols:
        if col in df.columns:
            print(f"   Chuẩn hóa cột: {col}")
            
            # Xử lý missing values trước khi chuẩn hóa
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"     Đã điền {df[col].isnull().sum()} missing values bằng median: {median_val:.2f}")
            
            df = standardize_zscore(df, col)
    
    # 2. Mã hóa Yes/No với xử lý lỗi
    yesno_cols = ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
    for col in yesno_cols:
        if col in df.columns:
            print(f"   Mã hóa cột: {col}")
            
            # Chuyển về string, strip whitespace, xử lý missing
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col] = df[col].replace({'Nan': 'No', 'None': 'No'})
            
            # Map giá trị
            mapping = {'Yes': 1, 'No': 0}
            df[col] = df[col].map(mapping)
            
            # Điền các giá trị không phải Yes/No bằng 0
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)
                print(f"     Đã điền {df[col].isnull().sum()} giá trị không hợp lệ bằng 0")
    
    # 3. Mã hóa cột phân loại đặc biệt
    if 'Sleep Duration' in df.columns:
        print(f"   Mã hóa cột: Sleep Duration")
        df = encode_sleep_hours(df)
    
    if 'Dietary Habits' in df.columns:
        print(f"   Mã hóa cột: Dietary Habits")
        df = encode_diet_score(df)
    
    print(f"✅ Chuẩn hóa dữ liệu hoàn tất")
    print(f"   Shape cuối cùng: {df.shape}")
    
    # Hiển thị các cột mới được tạo
    new_cols = [col for col in df.columns if '_std' in col or '_norm' in col 
                or col in ['sleep_hours', 'diet_score']]
    if new_cols:
        print(f"   Các cột mới tạo: {new_cols}")
    
    return df