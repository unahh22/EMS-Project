import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# =========================================================
# 1. DIRECTORY CONFIGURATION
# =========================================================
train_folder = r"D:\ADY201m\data\Train_split\Task_1"
test_folder = r"D:\ADY201m\data\Test_split\Task_1"

print("=" * 80)
print(f"{'TASK 1 - MODEL STABILITY TRAINING (3 CROSS-VALIDATION RUNS)':^80}")
print("=" * 80)

r2_scores = []
rmse_scores = []
run_labels = []

# =========================================================
# 2. MODEL TRAINING & EVALUATION
# =========================================================
for i in range(1, 4):
    train_file = os.path.join(train_folder, f'train_run_{i}.csv')
    test_file = os.path.join(test_folder, f'test_run_{i}.csv')

    # Load data safely
    if not os.path.exists(train_file):
        print(f"[ERROR] Cannot find {train_file}. Please run Step 1 first.")
        continue

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # Separate Features and Target
    X_train = df_train.drop(columns=['exam_score'])
    y_train = df_train['exam_score']

    X_test = df_test.drop(columns=['exam_score'])
    y_test = df_test['exam_score']

    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict & Evaluate
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2_scores.append(r2)
    rmse_scores.append(rmse)
    run_labels.append(f"Run {i}")

    print(f"--- RUN {i} RESULTS ---")
    print(f"Train: train_run_{i}.csv | Test: test_run_{i}.csv")
    print(f"R-squared (R2): {r2:.4f}")
    print(f"RMSE          : {rmse:.4f}\n")

# =========================================================
# 3. STABILITY REPORT
# =========================================================
avg_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)

print("=" * 80)
print("MODEL STABILITY REPORT")
print(f"-> Average R2 Score : {avg_r2:.4f}")
print(f"-> Variance (StdDev): {std_r2:.6f}")
print("=" * 80)

# =========================================================
# 4. PROFESSIONAL VISUALIZATION
# =========================================================
print("\nGenerating Model Stability Chart...")

# Setup styling
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid", context="talk")

# Create a DataFrame for Seaborn
results_df = pd.DataFrame({
    'Training Run': run_labels,
    'R2 Score': r2_scores
})

# Draw Barplot (ĐÃ FIX LỖI WARNING SEABORN)
colors = ['#3498db', '#9b59b6', '#34495e']
ax = sns.barplot(
    x='Training Run',
    y='R2 Score',
    data=results_df,
    palette=colors,
    hue='Training Run', # FIX: Bắt buộc phải có hue khi dùng palette
    legend=False,       # FIX: Ẩn cái legend thừa đi
    edgecolor='black',
    linewidth=1.5
)

# Dynamically adjust Y-axis to zoom in on the scores (FIX: Cắt từ 0.70)
y_min = 0.70
y_max = max(r2_scores) + 0.02
plt.ylim(y_min, y_max)

# Add the Average Line (Red Dashed Line)
plt.axhline(y=avg_r2, color='#e74c3c', linestyle='--', linewidth=2.5, label=f'Average R² = {avg_r2:.4f}')

# Add text values on top of bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.4f}',
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 8),  # 8 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold')

# Configure titles and labels
plt.title('Task 1: Model Stability Across Cross-Validation Runs', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Cross-Validation Run', fontsize=14, fontweight='bold')
plt.ylabel('R-squared ($R^2$) Score', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', frameon=True, fontsize=12)

plt.tight_layout()

# Export Image
image_name = 'Task1_Model_Stability.png'
plt.savefig(image_name, dpi=300)
print(f"[SUCCESS] Visual report saved as '{image_name}'")