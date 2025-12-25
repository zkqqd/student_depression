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
    sleep_map = {
        'Less than 5 hours': 4,
        '5-6 hours': 5.5,
        '7-8 hours': 7.5,
        'More than 8 hours': 9
    }
    df['sleep_hours'] = df[col].map(sleep_map)
    return df

def encode_diet_score(df, col='Dietary Habits'):
    """Chuyển thói quen ăn uống thành điểm"""
    df = df.copy()
    diet_map = {'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3}
    df['diet_score'] = df[col].map(diet_map)
    return df

def normalize_dataset(df):
    """Chuẩn hóa toàn bộ dataset với các bước cần thiết"""
    df = df.copy()
    
    # 1. Chuẩn hóa cột số
    numeric_cols = ['CGPA', 'Academic Pressure', 'Work/Study Hours', 'Financial Stress']
    for col in numeric_cols:
        if col in df.columns:
            df = standardize_zscore(df, col)
    
    # 2. Mã hóa Yes/No
    yesno_cols = ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
    for col in yesno_cols:
        if col in df.columns:
            df = encode_yesno_to_binary(df, col)
    
    # 3. Mã hóa cột phân loại đặc biệt
    if 'Sleep Duration' in df.columns:
        df = encode_sleep_hours(df)
    
    if 'Dietary Habits' in df.columns:
        df = encode_diet_score(df)
    
    print("✅ Chuẩn hóa dữ liệu hoàn tất")
    return df