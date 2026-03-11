import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# =========================================================
# 1. NẠP DỮ LIỆU & THIẾT LẬP THƯ MỤC GỐC
# =========================================================
file_path = r'D:\ADY201m\notebooks\dataADY201m_cleaned_normalized1.csv'
df = pd.read_csv(file_path)

train_base_folder = r"D:\ADY201m\data\Train_split\Task_2"
test_base_folder = r"D:\ADY201m\data\Test_split\Task_2"

print("=" * 80)
print("TẠO CẤU TRÚC THƯ MỤC LỒNG NHAU (NESTED DIRECTORIES) & CHIA DATA")
print("=" * 80)

# =========================================================
# 2. XỬ LÝ 5 FEATURE THEO CẤU TRÚC NHÁNH
# =========================================================
continuous_features = ['study_hours', 'class_attendance', 'sleep_hours']
ordinal_features = ['sleep_quality', 'facility_rating']
group_names = ['Low', 'Medium', 'High']


def process_nested_feature(feature_name, is_continuous):
    print(f"\n📁 Đang tạo nhánh cho Feature: {feature_name.upper()}")

    # Phân loại dữ liệu
    if is_continuous:
        df['temp_group'] = pd.qcut(df[feature_name], q=3, labels=group_names, duplicates='drop')
    else:
        map_dict = {0: 'Low', 1: 'Medium', 2: 'High'}
        df['temp_group'] = df[feature_name].map(map_dict)

    for group_val in group_names:
        # 1. Trích xuất nhóm & Drop Feature
        group_df = df[df['temp_group'] == group_val].copy()
        if len(group_df) == 0: continue
        group_df = group_df.drop(columns=['temp_group', feature_name])

        # 2. TẠO THƯ MỤC CON LỒNG NHAU (Ví dụ: Task_2/study_hours/Low/)
        train_nested_dir = os.path.join(train_base_folder, feature_name, group_val)
        test_nested_dir = os.path.join(test_base_folder, feature_name, group_val)

        os.makedirs(train_nested_dir, exist_ok=True)
        os.makedirs(test_nested_dir, exist_ok=True)

        print(f"  └── 📂 {group_val}: Chứa {len(group_df)} dòng")

        # 3. Trộn, Cắt (9/1) và Lưu file
        for i in range(1, 4):
            df_shuffled = shuffle(group_df, random_state=42 + i)
            df_train, df_test = train_test_split(df_shuffled, test_size=0.1, random_state=42)

            # Tên file giờ chỉ cần đơn giản là train_run_i.csv vì thư mục đã nói lên tất cả
            train_filename = os.path.join(train_nested_dir, f"train_run_{i}.csv")
            test_filename = os.path.join(test_nested_dir, f"test_run_{i}.csv")

            df_train.to_csv(train_filename, index=False)
            df_test.to_csv(test_filename, index=False)


# Thực thi
for feat in continuous_features: process_nested_feature(feat, True)
for feat in ordinal_features: process_nested_feature(feat, False)

if 'temp_group' in df.columns: df.drop(columns=['temp_group'], inplace=True)
print("\n[HOÀN TẤT] Hãy mở Windows Explorer để chiêm ngưỡng hệ thống thư mục mới!")