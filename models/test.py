import pandas as pd
df = pd.read_csv('D:\ADY201m\notebooks\dataADY201m_cleaned_normalized1.csv')

# Kiểm tra xem có giá trị nào "giống" số 0 không
print("--- FINAL CHECK FOR ZERO-LIKE VALUES ---")
check_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours', 'course']

for col in check_cols:
    # Tìm cả số 0, chữ '0', và khoảng trắng
    zeros_count = len(df[df[col].astype(str).isin(['0', '0.0', ' '])])
    print(f"Column '{col}': found {zeros_count} hidden zeros.")

print("-" * 40)