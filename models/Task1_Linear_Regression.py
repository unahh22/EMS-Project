import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Đường dẫn tới file data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', '04_Normalized_Data.csv')

print("Loading file:", data_path)

# Đọc dữ liệu
data = pd.read_csv(data_path)

print("Dataset shape:", data.shape)
print(data.head())

# Giả sử cột cuối là target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Chia train test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL RESULT =====")
print("MSE:", mse)
print("R2 Score:", r2)
