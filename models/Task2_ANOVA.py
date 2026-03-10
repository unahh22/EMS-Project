import pandas as pd
import os

# ==========================================
# GIAI ĐOẠN 1: LOAD VÀ LÀM SẠCH DATA GỐC
# ==========================================
# Trỏ chính xác vào file data gốc trong thư mục data
file_path = r'D:\ADY201m\notebooks\dataADY201m_cleaned_normalized1.csv'
df = pd.read_csv(file_path)

# Làm sạch dữ liệu (bỏ số 0) để đảm bảo chất lượng
numeric_cols = ['study_hours', 'age', 'class_attendance', 'sleep_hours']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[df[col] > 0]
if 'course' in df.columns:
    df = df[~df['course'].astype(str).isin(['0', '0.0', 'nan', 'None'])]

# Cài đặt thư mục đích là chính folder 'data'
output_folder = r"D:\ADY201m\data"

# ==========================================
# GIAI ĐOẠN 2: CẮT FILE VÀ DROP-GROUP
# ==========================================
features_to_group = ['study_hours', 'class_attendance', 'sleep_hours']
groups = ['Low', 'Medium', 'High']

print(f"--- ĐANG TIẾN HÀNH CẮT VÀ LƯU VÀO FOLDER: {output_folder} ---")

for feature in features_to_group:
    # Cắt thành 3 nhóm dựa trên tứ phân vị (để cân bằng số lượng)
    df['temp_group'] = pd.qcut(df[feature], q=3, labels=groups, duplicates='drop')

    for group_val in groups:
        # 1. Lọc lấy nhóm sinh viên
        group_df = df[df['temp_group'] == group_val].copy()

        # 2. DROP-GROUP (Xóa cột đang phân tích)
        group_df = group_df.drop(columns=['temp_group', feature])

        # 3. Tạo đường dẫn tuyệt đối và lưu file
        # Ví dụ: D:\ADY201m\data\study_hours_Low.csv
        filename = os.path.join(output_folder, f"{feature}_{group_val}.csv")
        group_df.to_csv(filename, index=False)
        print(f"✅ Đã lưu file: {feature}_{group_val}.csv (Gồm {len(group_df)} dòng)")

# Dọn dẹp DataFrame gốc trong bộ nhớ
df = df.drop(columns=['temp_group'], errors='ignore')

print("\n🚀 HOÀN TẤT! Hãy mở thư mục D:\\ADY201m\\data để kiểm tra 9 file con mới.")