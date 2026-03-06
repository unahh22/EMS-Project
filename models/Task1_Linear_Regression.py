
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ==============================
# 1. Load dataset
# ==============================

data = pd.read_csv("data/dataADV201m_cleaned_normalized.csv")

# Target (điểm cuối)
y = data["G3"]

# Features (bỏ G3 ra)
X = data.drop("G3", axis=1)

# ==============================
# 2. Chạy model 3 lần
# ==============================

results = []

for i in range(3):

    random_state = i

    # Shuffle + Split 90/10
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.1,
        shuffle=True,
        random_state=random_state
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append(rmse)

    print(f"Run {i+1} | random_state = {random_state} | RMSE = {rmse:.4f}")

# ==============================
# 3. Tính trung bình
# ==============================

avg_rmse = np.mean(results)

print("\n==============================")
print("Average RMSE:", round(avg_rmse, 4))
print("==============================")
