import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
file_path = r'D:\ADY201m\data\data.csv'
df = pd.read_csv(file_path)
print(f"Original count: {len(df)}") # Should be 20,000

# ==========================================
# 1. LỌC BỎ ZEROS VÀ DỮ LIỆU LỖI
# ==========================================
numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[df[col] > 0]

df = df[~df['course'].astype(str).isin(['0', '0.0', 'nan', 'None'])]
print(f"Count after removing Zeros & Invalid: {len(df)}")

# ==========================================
# [FIXED 1]: XỬ LÝ DỮ LIỆU THỨ BẬC (ORDINAL ENCODING)
# ==========================================
# Không dùng LabelEncoder nữa. Tự map bằng tay để máy hiểu đúng thứ tự:
# Kém = 0, Trung bình = 1, Tốt/Khó = 2
sleep_map = {'poor': 0, 'average': 1, 'good': 2}
facility_map = {'low': 0, 'medium': 1, 'high': 2}
exam_map = {'easy': 0, 'moderate': 1, 'hard': 2}

df['sleep_quality'] = df['sleep_quality'].map(sleep_map)
df['facility_rating'] = df['facility_rating'].map(facility_map)
df['exam_difficulty'] = df['exam_difficulty'].map(exam_map)

# ==========================================
# [FIXED 2]: XỬ LÝ DỮ LIỆU DANH NGHĨA (ONE-HOT ENCODING)
# ==========================================
# Dùng get_dummies để các khóa học (course), giới tính... bình đẳng với nhau
nominal_cols = ['gender', 'course', 'internet_access', 'study_method']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

# ==========================================
# [FIXED 3]: CHUẨN HÓA MIN-MAX SCALING
# ==========================================
# Chỉ scale các biến số, không đụng vào các biến chữ đã mã hóa ở trên
scaler = MinMaxScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# ==========================================
# 4. XUẤT FILE CHUẨN MỰC
# ==========================================
# Đã loại bỏ lệnh drop_duplicates() để giữ lại mật độ sinh viên thực tế
if 'student_id' in df.columns:
    df = df.drop(columns=['student_id'])

output_path = r'D:\ADY201m\notebooks\dataADY201m_cleaned_normalized1.csv'
df.to_csv(output_path, index=False)

print(f"Final exported rows: {len(df)}")
print(f"SUCCESS: Data mathematically corrected and saved to {output_path}!")