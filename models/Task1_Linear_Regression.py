import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# GET PATH DATA
# =========================

# Lấy đường dẫn thư mục hiện tại (models)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Đi lên 1 cấp rồi vào thư mục data
data_path = os.path.join(current_dir, "..", "data", "dataADY201m_cleaned_normalized.csv")

print("Loading file:", data_path)

# =========================
# LOAD DATA
# =========================

data = pd.read_csv(data_path)

print("Data loaded successfully!")
print("Dataset shape:", data.shape)

# =========================
# SPLIT DATA
# =========================

# giả sử cột cuối là target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# =========================
# TRAIN MODEL
# =========================

model = LinearRegression()
model.fit(X_train, y_train)

print("Model training completed!")

# =========================
# PREDICT
# =========================

y_pred = model.predict(X_test)

# =========================
# EVALUATE
# =========================

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print("RMSE:", rmse)
print("R2 Score:", r2)
