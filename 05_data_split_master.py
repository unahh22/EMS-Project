import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc dữ liệu đã chuẩn hóa
file_name = 'dataADY201m_cleaned_normalized.csv'
print(f"Đang xử lý file: {file_name}")
df = pd.read_csv(file_name)

# Tách X (Features) và y (Target)
X = df.drop(columns=['exam_score'])
y = df['exam_score']

# Chia dữ liệu tỷ lệ 9/1 (test_size = 0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Gộp X và y lại để lưu thành file csv hoàn chỉnh
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Lưu ra máy tính
train_data.to_csv('train_91_norm.csv', index=False)
test_data.to_csv('test_91_norm.csv', index=False)

print(f"HOÀN TẤT! Đã lưu thành công:")
print(f"- train_91_norm.csv ({len(train_data)} dòng)")
print(f"- test_91_norm.csv ({len(test_data)} dòng)")