import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==========================================
# SETUP DIRECTORIES
# ==========================================
train_folder = r"D:\ADY201m\data\Train_split\Task_0"
test_folder = r"D:\ADY201m\data\Test_split\Task_0"

# ==========================================
# HELPER FUNCTION FOR TRAINING
# ==========================================
def train_and_evaluate(train_file, test_file):
    # Load files from respective folders
    df_train = pd.read_csv(os.path.join(train_folder, train_file))
    df_test = pd.read_csv(os.path.join(test_folder, test_file))

    X_train = df_train.drop(columns=['exam_score'])
    y_train = df_train['exam_score']

    X_test = df_test.drop(columns=['exam_score'])
    y_test = df_test['exam_score']

    # Handle text columns (One-Hot Encoding) safely
    X_combined = pd.concat([X_train, X_test])
    X_combined = pd.get_dummies(X_combined, drop_first=True)

    X_train_final = X_combined.iloc[:len(X_train)]
    X_test_final = X_combined.iloc[len(X_train):]

    # Train Model
    model = LinearRegression()
    model.fit(X_train_final, y_train)

    # Predict & Evaluate
    y_pred = model.predict(X_test_final)
    return r2_score(y_test, y_pred)

print("=" * 70)
print("STARTING TASK 0: MODEL TRAINING & EVALUATION")
print("=" * 70)

# ==========================================
# RUN TRAINING PIPELINE
# ==========================================
# 1. Cleaned Data (Not Normalized)
r2_cleaned = train_and_evaluate('task0_cleaned_train.csv', 'task0_cleaned_test.csv')

# 2. Cleaned & Normalized Data
r2_normalized = train_and_evaluate('task0_normalized_train.csv', 'task0_normalized_test.csv')

# Print Report
print("\n" + "-" * 70)
print("FINAL RESULTS: R-SQUARED SCORE")
print("-" * 70)
results_df = pd.DataFrame([
    {'Dataset Type': 'Cleaned (Not Normalized)', 'Accuracy (R2)': r2_cleaned},
    {'Dataset Type': 'Cleaned & Normalized', 'Accuracy (R2)': r2_normalized}
])
print(results_df.to_string(index=False))
print("-" * 70)

# ==========================================
# GENERATE VISUALIZATION
# ==========================================
print("\nGenerating comparative visual report...")
sns.set_theme(style="whitegrid", context="talk")
plt.figure(figsize=(10, 7))

# Custom colors for A/B comparison
colors = ['#e74c3c', '#2ecc71']

# Create Barplot
ax = sns.barplot(
    x='Dataset Type',
    y='Accuracy (R2)',
    data=results_df,
    palette=colors,
    hue='Dataset Type',
    legend=False,
    edgecolor='black',
    linewidth=1.5
)

plt.title('Task 0: Accuracy Comparison (Raw vs. Normalized)', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Dataset Pipeline', fontsize=14, fontweight='bold')
plt.ylabel('R-squared ($R^2$) Score', fontsize=14, fontweight='bold')

# Dynamic Y-axis to highlight differences
min_r2 = min(r2_cleaned, r2_normalized)
plt.ylim(min_r2 - 0.05, 1.0)

# Add exact numbers on top of bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.4f}',
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

plt.tight_layout()

# Save image
image_name = 'Task0_Accuracy_Comparison.png'
plt.savefig(image_name, dpi=300)
print(f"[SUCCESS] Visualization saved as '{image_name}'")