import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load data
df = pd.read_csv('D:\ADY201m\data\data.csv')
print(f"Original count: {len(df)}") # Should be 20,000

# ==========================================
# CỰC KỲ QUAN TRỌNG: LỌC BỎ ZEROS (THE "REAL" CLEANING)
# ==========================================
# 1. Remove Zeros from numeric columns
numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
for col in numeric_features:
    # Convert to numeric just in case there are strings
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # This is the line that actually REMOVES the rows:
    df = df[df[col] > 0]

# 2. Remove '0' from categorical 'course'
df = df[~df['course'].astype(str).isin(['0', '0.0', 'nan', 'None'])]

print(f"Count after removing Zeros: {len(df)}") # This must be < 20,000!

# ==========================================
# TIẾP TỤC CÁC BƯỚC CŨ
# ==========================================
# 3. Standard cleanup (Optional since profiling shows 0 missing/duplicates)
df = df.drop_duplicates().dropna()

# 4. Label Encoding
categorical_cols = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# 5. Min-Max Scaling (Chỉ làm trên data ĐÃ SẠCH)
scaler = MinMaxScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# 6. Final Export
df_final = df.drop(columns=['student_id']) if 'student_id' in df.columns else df
df_final.to_csv('dataADY201m_cleaned_normalized1.csv', index=False)

print(f"Final exported rows: {len(df_final)}")
print("SUCCESS: Data cleaned and saved!")