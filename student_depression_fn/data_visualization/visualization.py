import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import os
import matplotlib.pyplot as plt
# Lưu biểu đồ hiện tại thành file ảnh
def save_current_figure(name):
    os.makedirs("chart", exist_ok=True)
    plt.savefig(f"chart/{name}.png", dpi=300, bbox_inches="tight")
def show_save_and_wait(name):
    save_current_figure(name)
    plt.close()  
# Hàm thống kê mô tả cơ bản
def describe_data(df):
    description = df.describe(include = 'all')
    missing_values = df.isnull().sum()
    data_types = df.dtypes
    return {'description': description, 'missing_values': missing_values, 'data_types': data_types}
# Tính bảng chéo giữa 'Depression' và 'Academic Pressure'
def crosstab_depression_academic_pressure(df):
    crosstab_result = pd.crosstab(df['Depression'], df['Academic Pressure'])
    return crosstab_result
# Nhóm dữ liệu theo cột 'Gender' và tính tỷ lệ trầm cảm trung bình
def group_by_gender_depression(df):
    grouped_data = df.groupby('Gender', observed=False)['Depression'].mean()
    return grouped_data
# Số lượng trầm cảm theo nghề
def count_depression_by_profession(df):
    count_data = df.groupby('Profession', observed=False)['Depression'].count()
    return count_data
# Vẽ biểu đồ countplot cho 'Depression' theo 'Gender'
def plot_depression_by_gender(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Depression', data=df, hue='Gender')
    plt.title('Depression levels by Gender')
    show_save_and_wait("depression_by_gender")
# Biểu đồ tính lượng trầm cảm theo 'Gender'
def plot_gender_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Gender', data=df)
    plt.title('Distribution of Gender')
    plt.xlabel('Gender')
    plt.ylabel('Frequency')
    show_save_and_wait("gender_distribution")
# Biểu đồ tần suất của các giá trị trong cột 'Age' và lấy top 
def plot_top_age_distribution(df):
    top_ages = df['Age'].value_counts().head(8)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Age', data=df, order=top_ages.index)
    plt.title('Distribution of Age (Top): ')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    show_save_and_wait("top_age_distribution")
# Biểu đồ phân phối CGPA
def plot_cgpa_distribution(df, bins=20):
    x = pd.to_numeric(df['CGPA'], errors='coerce')
    plt.figure(figsize=(10, 6))
    plt.hist(x.dropna(), bins=bins, edgecolor='black')
    plt.title('Distribution of CGPA')
    plt.xlabel('CGPA')
    plt.ylabel('Frequency')
    show_save_and_wait("cgpa_distribution")
# Biểu đồ xác định thời gian ngủ phổ biến của sinh viên
def plot_sleep_duration_distribution(df, bins=20):
    x = pd.to_numeric(df['Sleep Duration'], errors='coerce')
    plt.figure(figsize=(10, 6))
    plt.hist(x.dropna(), bins=bins, edgecolor='black')
    plt.title('Distribution of Sleep Duration')
    plt.xlabel('Sleep Duration')
    plt.ylabel('Frequency')
    show_save_and_wait("sleep_duration_distribution")
# Ma trận tương quan
def plot_correlation_matrix(df):
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    exclude_columns = ['Work Pressure', 'Job Satisfaction']
    numeric_features = [c for c in numeric_features if c not in exclude_columns]

    if len(numeric_features) < 2:
        print("Không đủ cột numeric để vẽ correlation.")
        return

    corr = df[numeric_features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    show_save_and_wait("correlation_matrix")
# Biểu đồ ảnh hưởng của CGPA và Despression
def plot_cgpa_vs_depression(df):
    x = pd.to_numeric(df['CGPA'], errors='coerce')
    d = pd.to_numeric(df['Depression'], errors='coerce')

    data = df.copy()
    data['CGPA'] = x
    data['Depression'] = d

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Depression', y='CGPA', data=data)
    plt.title("CGPA Distribution by Depression Status")
    plt.xlabel("Depression (0 = No, 1 = Yes)")
    plt.ylabel("CGPA")
    show_save_and_wait("cgpa_vs_depression")
# Biểu đồ phân tán đánh giá trầm cảm
def plot_depression_assessment_scatter(df):
    x = pd.to_numeric(df['CGPA_std'], errors='coerce')
    y = pd.to_numeric(df['Work/Study Hours_std'], errors='coerce')
    d = pd.to_numeric(df['Depression'], errors='coerce')

    mask = x.notna() & y.notna() & d.notna()
    x, y, d = x[mask], y[mask], d[mask]

    plt.figure(figsize=(9, 7))
    plt.scatter(x, y, c=d)

    x0 = x.mean()
    y0 = y.mean()
    plt.axvline(x0)
    plt.axhline(y0)

    plt.title("Depression Assessment: Standardized CGPA vs Work/Study Hours")
    plt.xlabel("CGPA_std")
    plt.ylabel("Work/Study Hours_std")
    plt.colorbar(label="Depression (0 = No, 1 = Yes)")

    plt.tight_layout()
    show_save_and_wait("depression_assessment_scatter")
# Biểu đồ phân tán tương tác giữa giấc ngủ - thời gian học/ làm và trạng thái trầm cảm
def plot_sleep_work_depression(df):
    plt.figure(figsize=(9, 7))
    plt.scatter(
        df['Sleep Duration'],
        df['Work/Study Hours'],
        c=df['Depression']
    )
    plt.xlabel("Sleep Duration")
    plt.ylabel("Work/Study Hours")
    plt.title("Sleep - Work Interaction and Depression")
    plt.colorbar(label="Depression (0 = No, 1 = Yes)")
    show_save_and_wait("sleep_work_depression") 
    
# Hàm tổng hợp
def run_all_analysis(df):
    print("===== THỐNG KÊ MÔ TẢ DỮ LIỆU =====")
    desc = describe_data(df)
    print(desc['description'])
    print("\nGiá trị thiếu:")
    print(desc['missing_values'])
    print("\nKiểu dữ liệu:")
    print(desc['data_types'])

    print("\n===== BẢNG CHÉO: DEPRESSION vs ACADEMIC PRESSURE =====")
    print(crosstab_depression_academic_pressure(df))

    print("\n===== TỶ LỆ TRẦM CẢM THEO GIỚI TÍNH =====")
    print(group_by_gender_depression(df))

    print("\n===== SỐ LƯỢNG TRẦM CẢM THEO NGHỀ NGHIỆP =====")
    print(count_depression_by_profession(df))

    print("\n===== VẼ BIỂU ĐỒ =====")

    print("1. Depression theo Gender")
    plot_depression_by_gender(df)

    print("2. Phân phối Gender")
    plot_gender_distribution(df)

    print("3. Phân phối Age (Top)")
    plot_top_age_distribution(df)

    print("4. Phân phối CGPA")
    plot_cgpa_distribution(df)

    print("5. Phân phối Sleep Duration")
    plot_sleep_duration_distribution(df)

    print("6. Ma trận tương quan")
    plot_correlation_matrix(df)

    print("7. Ảnh hưởng CGPA đến Depression")
    plot_cgpa_vs_depression(df)

    print("8. Đánh giá trầm cảm (CGPA_std vs Work/Study Hours_std)")
    plot_depression_assessment_scatter(df)

    print("9. Tương tác Giấc ngủ - Học/Làm - Trầm cảm")
    plot_sleep_work_depression(df)

    print("\n✅ Hoàn thành quy trình phân tích!")
