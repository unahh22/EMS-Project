import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils import shuffle

# ==========================================
# STEP 0: LOAD DATA & PREPROCESSING
# ==========================================
file_path = r'D:\ADY201m\data\data.csv'
df = pd.read_csv(file_path)

print(f"Total initial rows: {len(df)}")

# 1. Clean numeric columns (> 0)
numeric_cols = ['study_hours', 'age', 'class_attendance', 'sleep_hours']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df[(df[numeric_cols] > 0).all(axis=1)]

# 2. Clean categorical columns (remove missing/zero values)
df = df[~df['course'].isin([0, '0', '0.0'])]

print(f"Total clean rows for training: {len(df)}\n")

# ==========================================
# PREPARE FEATURES (X) AND TARGET (y)
# ==========================================
X = df.drop(columns=['exam_score'])
y = df['exam_score']

# Convert categorical text columns (like 'course') into numbers (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# ==========================================
# TASK 1: SHUFFLE, SPLIT, MODELING, ACCURACY (REPEAT 3 TIMES)
# ==========================================
model = LinearRegression()
results = []

print("-" * 50)
print("TASK 1: 3-FOLD SHUFFLE & TRAIN EVALUATION")
print("-" * 50)

# Repeat the process 3 times
for i in range(3):
    # Step 1: Shuffle Dataset
    X_shuffled, y_shuffled = shuffle(X, y, random_state=None)

    # Step 2: Split dataset (Training/Test - 9/1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_shuffled, y_shuffled, test_size=0.1, random_state=None
    )

    # Step 3: Modeling (Find function f(x))
    model.fit(X_train, y_train)

    # Step 4: Calculate Accuracy (R2 Score)
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)

    # Save results to list
    results.append({
        'Iteration': f'Run {i + 1}',
        'Accuracy (R2)': accuracy
    })

    print(f"Run {i + 1} - Accuracy (R2): {accuracy:.4f}")

# ==========================================
# FINAL RESULT: TABLE OF 3 EVALUATIONS & AVERAGE
# ==========================================
# Create DataFrame for the result table
results_df = pd.DataFrame(results)

# Calculate final average accuracy
average_accuracy = results_df['Accuracy (R2)'].mean()

# Add average row to the table
mean_row = pd.DataFrame([{
    'Iteration': 'AVERAGE',
    'Accuracy (R2)': average_accuracy
}])

final_df = pd.concat([results_df, mean_row], ignore_index=True)

print("\n" + "=" * 50)
print("FINAL RESULT TABLE:")
print("=" * 50)
# Print the final table with formatted floating numbers
print(final_df.to_string(index=False, float_format="%.4f"))
print("=" * 50)