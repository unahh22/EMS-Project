import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# ==========================================
# STEP 0: SETUP DIRECTORIES & PATHS
# ==========================================
raw_file_path = r'D:\ADY201m\data\data.csv'
norm_file_path = r'D:\ADY201m\notebooks\dataADY201m_cleaned_normalized1.csv'

# Tách riêng 2 thư mục Train và Test
train_folder = r"D:\ADY201m\data\Train_split\Task_0"
test_folder = r"D:\ADY201m\data\Test_split\Task_0"

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

print("=" * 70)
print("STARTING TASK 0: A/B DATA PARTITIONING (RAW vs NORMALIZED)")
print("=" * 70)

# ==========================================
# STEP 1: PROCESS "CLEANED ONLY" DATA
# ==========================================
print("\n--- PROCESSING 1: CLEANED (NOT NORMALIZED) DATA ---")
df_raw = pd.read_csv(raw_file_path)

# Cleaning logic (same as task 1)
numeric_cols = ['study_hours', 'age', 'class_attendance', 'sleep_hours']
for col in numeric_cols:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        df_raw = df_raw[df_raw[col] > 0]

# Shuffle & Split 9/1
df_raw_shuffled = shuffle(df_raw, random_state=42)
df_raw_train, df_raw_test = train_test_split(df_raw_shuffled, test_size=0.1, random_state=42)

# Export to separate folders
train_raw_out = os.path.join(train_folder, 'task0_cleaned_train.csv')
test_raw_out = os.path.join(test_folder, 'task0_cleaned_test.csv')

df_raw_train.to_csv(train_raw_out, index=False)
df_raw_test.to_csv(test_raw_out, index=False)

print(f"  -> Exported Cleaned Train: {train_raw_out} ({len(df_raw_train)} rows)")
print(f"  -> Exported Cleaned Test:  {test_raw_out} ({len(df_raw_test)} rows)")

# ==========================================
# STEP 2: PROCESS "CLEANED & NORMALIZED" DATA
# ==========================================
print("\n--- PROCESSING 2: CLEANED & NORMALIZED DATA ---")
# Data is already cleaned and normalized, just read it
df_norm = pd.read_csv(norm_file_path)

# Shuffle & Split 9/1
df_norm_shuffled = shuffle(df_norm, random_state=42)
df_norm_train, df_norm_test = train_test_split(df_norm_shuffled, test_size=0.1, random_state=42)

# Export to separate folders
train_norm_out = os.path.join(train_folder, 'task0_normalized_train.csv')
test_norm_out = os.path.join(test_folder, 'task0_normalized_test.csv')

df_norm_train.to_csv(train_norm_out, index=False)
df_norm_test.to_csv(test_norm_out, index=False)

print(f"  -> Exported Normalized Train: {train_norm_out} ({len(df_norm_train)} rows)")
print(f"  -> Exported Normalized Test:  {test_norm_out} ({len(df_norm_test)} rows)")
print("\n[SUCCESS] Task 0 Data splitting completed!")