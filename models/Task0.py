import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# STEP 0: LOAD DATA & PREPROCESSING
# ==========================================
file_path = r'D:\ADY201m\data\data.csv'
df = pd.read_csv(file_path)

print(f"Total initial rows: {len(df)}")

# 1. Clean numeric columns (> 0)
numeric_cols = ['study_hours', 'age', 'class_attendance', 'sleep_hours']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce') # Ensure they are numbers
df = df[(df[numeric_cols] > 0).all(axis=1)]

# 2. Clean categorical columns (remove missing/zero values)
df = df[~df['course'].isin([0, '0', '0.0'])]

print(f"Total clean rows for training: {len(df)}\n")

# ==========================================
# SEPARATE FEATURES (X) AND TARGET (y) & SPLIT 9/1
# ==========================================
X = df.drop(columns=['exam_score'])
y = df['exam_score']

# CRITICAL FIX: Convert categorical text columns (like 'course') into numbers
# Linear Regression cannot process strings without this step!
X = pd.get_dummies(X, drop_first=True)

# Split dataset (90% Train, 10% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# ==========================================
# REPORT 1: TRAINING ON RAW DATA
# ==========================================
print("-" * 50)
print("REPORT 1: BEFORE NORMALIZATION (RAW DATA)")

# Initialize and train model
model_raw = LinearRegression()
model_raw.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred_raw = model_raw.predict(X_test)
r2_raw = r2_score(y_test, y_pred_raw)

# Log the report
print(f"1. Dataset: Raw Data (Cleaned of 0-value errors)")
print(f"2. Model: Linear Regression")
print(f"3. Accuracy (R-squared): {r2_raw:.4f}")
print("-" * 50)


# ==========================================
# REPORT 2: DATA NORMALIZATION AND REPEAT
# ==========================================
print("\n" + "-" * 50)
print("REPORT 2: AFTER NORMALIZATION (NORMALIZED DATA)")

# Perform Data Normalization using StandardScaler
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Train model on normalized data
model_norm = LinearRegression()
model_norm.fit(X_train_norm, y_train)

# Predict and calculate accuracy
y_pred_norm = model_norm.predict(X_test_norm)
r2_norm = r2_score(y_test, y_pred_norm)

# Log the report
print(f"1. Dataset: Normalized Data (StandardScaler)")
print(f"2. Model: Linear Regression")
print(f"3. Accuracy (R-squared): {r2_norm:.4f}")
print("-" * 50)