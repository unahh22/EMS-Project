# ============================================================
# TASK 1 - LINEAR REGRESSION (ADVANCED VERSION)
# Hưng & Đăng
#
# Quy trình:
# 1. Load data
# 2. Shuffle data
# 3. Split Train/Test (90% - 10%)
# 4. Train Linear Regression
# 5. Evaluate RMSE + R2
# 6. Loop 3 lần (random_state khác nhau)
# 7. Hiển thị bảng kết quả
# 8. Vẽ biểu đồ RMSE
# ============================================================

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt


# ============================================================
# LOAD DATA
# ============================================================

current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "..", "data", "dataADV201m_cleaned_normalized.csv")

data = pd.read_csv(data_path)

print("=================================")
print("DATA LOADED SUCCESSFULLY")
print("Shape:", data.shape)
print("=================================")


# ============================================================
# SPLIT FEATURES / TARGET
# ============================================================

X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# ============================================================
# TRAIN MODEL 3 TIMES
# ============================================================

results = []

for i in range(3):

    print(f"\n RUN {i+1}")

    # Shuffle + Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        shuffle=True,
        random_state=i
    )

    # Model
    model = LinearRegression()

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Run": i + 1,
        "Random_State": i,
        "RMSE": rmse,
        "R2_Score": r2
    })


# ============================================================
# RESULT TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n==============================")
print("RESULT TABLE")
print("==============================")

print(results_df)


# ============================================================
# AVERAGE RESULTS
# ============================================================

avg_rmse = results_df["RMSE"].mean()
avg_r2 = results_df["R2_Score"].mean()

print("\n==============================")
print("AVERAGE RESULTS")
print("==============================")

print("Average RMSE :", round(avg_rmse, 4))
print("Average R2   :", round(avg_r2, 4))


# ============================================================
# PLOT RMSE
# ============================================================

plt.figure()

plt.plot(results_df["Run"], results_df["RMSE"], marker="o")

plt.title("RMSE Across 3 Runs")
plt.xlabel("Run")
plt.ylabel("RMSE")

plt.grid(True)

plt.show()
