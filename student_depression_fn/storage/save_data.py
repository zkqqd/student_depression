import os

def save_data(df, output_path):
    if df is None:
        raise ValueError("❌ Không thể lưu file: df = None")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Đã lưu dữ liệu tại: {output_path}")
