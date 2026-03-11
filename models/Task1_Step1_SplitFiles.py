# Tên file gợi ý: Task1_Step1_SplitData.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# 1. Load data
file_path = r'D:\ADY201m\notebooks\dataADY201m_cleaned_normalized1.csv'
df = pd.read_csv(file_path)

# 2. Setup thư mục đích
train_folder = r"D:\ADY201m\data\Train_split\Task_1"
test_folder = r"D:\ADY201m\data\Test_split\Task_1"

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

print("=" * 70)
print("BẮT ĐẦU CHIA DỮ LIỆU (90/10) VÀO THƯ MỤC TRAIN/TEST RIÊNG BIỆT")
print("=" * 70)

for i in range(1, 4):
    # Trộn ngẫu nhiên toàn bộ dữ liệu (mỗi lần trộn sẽ ra thứ tự khác nhau)
    df_shuffled = shuffle(df, random_state=42 + i)

    # Cắt 90% Train - 10% Test
    df_train, df_test = train_test_split(df_shuffled, test_size=0.1, random_state=42)

    # Lưu vào đúng thư mục
    train_filename = os.path.join(train_folder, f'train_run_{i}.csv')
    test_filename = os.path.join(test_folder, f'test_run_{i}.csv')

    df_train.to_csv(train_filename, index=False)
    df_test.to_csv(test_filename, index=False)

    print(f"Lần chạy {i}:")
    print(f"  -> Đã đẩy File Train vào: {train_filename}")
    print(f"  -> Đã đẩy File Test  vào: {test_filename}")

print("\nHOÀN TẤT! Dữ liệu đã được phân lô sạch sẽ.")