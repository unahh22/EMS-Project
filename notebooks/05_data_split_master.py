import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. Cập nhật đường dẫn đọc file: trỏ vào thư mục data/
# Lưu ý: Chạy script này từ thư mục gốc D:\ADY201m
file_name = 'data/04_Normalized_Data.csv'
print(f"Đang xử lý file: {file_name}")

# Kiểm tra xem file có tồn tại không để tránh lỗi
if not os.path.exists(file_name):
    print(f"LỖI: Không tìm thấy file tại {file_name}. Hãy đảm bảo bạn đang đứng ở thư mục gốc của dự án để chạy.")
else:
    df = pd.read_csv(file_name)

    # Tách X (Features) và y (Target)
    X = df.drop(columns=['exam_score'])
    y = df['exam_score']

    # Chia dữ liệu tỷ lệ 9/1 (test_size = 0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Gộp X và y lại để lưu thành file csv hoàn chỉnh
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # 2. Cập nhật đường dẫn lưu file: Lưu vào trong thư mục data/
    train_path = 'data/train_91_norm.csv'
    test_path = 'data/test_91_norm.csv'

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"HOÀN TẤT! Đã lưu thành công vào thư mục data/:")
    print(f"- {train_path} ({len(train_data)} dòng)")
    print(f"- {test_path} ({len(test_data)} dòng)")