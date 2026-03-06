import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# lấy đường dẫn file hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# đường dẫn tới data
data_path = os.path.join(current_dir, "..", "data", "dataADV201m_cleaned_normalized.csv")

print("Loading file:", data_path)

data = pd.read_csv(data_path)

print("Data loaded")
print("Shape:", data.shape)

target = data.columns[-1]

X = data.drop(columns=[target])
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)
