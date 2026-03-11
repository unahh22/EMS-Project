import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Trỏ đến thư mục GỐC
train_base_folder = r"D:\ADY201m\data\Train_split\Task_2"
test_base_folder = r"D:\ADY201m\data\Test_split\Task_2"

features = ['study_hours', 'class_attendance', 'sleep_hours', 'sleep_quality', 'facility_rating']
groups = ['Low', 'Medium', 'High']

print("=" * 80)
print(f"{'BÁO CÁO KẾT QUẢ ANOVA BẰNG LINEAR REGRESSION - TASK 2':^80}")
print("=" * 80)
print(f"{'Feature Đã Drop':<20} | {'Nhóm (Group)':<10} | {'R2 Trung Bình (3 Lần)':<25} | {'Độ Lệch (Std)':<15}")
print("-" * 80)

results_summary = []

for feature in features:
    for group in groups:
        r2_list = []

        # Lặn xuống thư mục con: Task_2 / Feature / Group /
        for i in range(1, 4):
            train_file = os.path.join(train_base_folder, feature, group, f"train_run_{i}.csv")
            test_file = os.path.join(test_base_folder, feature, group, f"test_run_{i}.csv")

            # Bỏ qua nếu cấu trúc rỗng
            if not os.path.exists(train_file):
                continue

            df_train = pd.read_csv(train_file)
            df_test = pd.read_csv(test_file)

            X_train = df_train.drop(columns=['exam_score'])
            y_train = df_train['exam_score']
            X_test = df_test.drop(columns=['exam_score'])
            y_test = df_test['exam_score']

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2_list.append(r2_score(y_test, y_pred))

        if len(r2_list) > 0:
            avg_r2 = np.mean(r2_list)
            std_r2 = np.std(r2_list)
            print(f"{feature:<20} | {group:<10} | {avg_r2:<25.4f} | {std_r2:<15.6f}")
            results_summary.append({'Feature': feature, 'Group': group, 'Avg_R2': avg_r2})

print("=" * 80)
df_results = pd.DataFrame(results_summary)
df_results.to_csv("Task2_R2_Nested_Summary.csv", index=False)
print("\n[HOÀN TẤT] File tổng hợp đã lưu: 'Task2_R2_Nested_Summary.csv'")