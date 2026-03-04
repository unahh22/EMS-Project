import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# =====================================================================
# PHẦN 1: TẢI DỮ LIỆU (SỬA ĐƯỜNG DẪN VÀO THƯ MỤC data)
# =====================================================================
TRAIN_FILE = 'data/train_91_norm.csv'
TEST_FILE = 'data/test_91_norm.csv'

print("Đang tải dữ liệu...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Tách Feature (X) và Target (y)
X_train = train_df.drop(columns=['exam_score'])
y_train = train_df['exam_score']

X_test = test_df.drop(columns=['exam_score'])
y_test = test_df['exam_score']

# =====================================================================
# PHẦN 2: HUẤN LUYỆN MÔ HÌNH RANDOM FOREST
# =====================================================================
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================================
# PHẦN 3: DỰ ĐOÁN & ĐÁNH GIÁ (GIỮ NGUYÊN)
# =====================================================================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n>>> Kết quả RMSE của mô hình: {rmse:.4f}")

if rmse < 8:
    print("=> CHÚC MỪNG! Mô hình đã đạt chỉ tiêu của dự án (RMSE < 8).")
else:
    print("=> Mô hình chưa đạt chỉ tiêu (RMSE >= 8). Cần tinh chỉnh thêm!")
