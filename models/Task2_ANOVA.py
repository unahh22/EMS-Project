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
df_raw = pd.read_csv(file_path)

# Cleaning data (The same logic as Task 0 & 1)
numeric_cols = ['study_hours', 'age', 'class_attendance', 'sleep_hours']
for col in numeric_cols:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
df_clean = df_raw[(df_raw[numeric_cols] > 0).all(axis=1)].copy()
df_clean = df_clean[~df_clean['course'].isin([0, '0', '0.0'])]


def evaluate_model_3_times(data):
    """Function to repeat Task 1: Shuffle, Split 9/1, Train 3 times, and Average Accuracy."""
    X = data.drop(columns=['exam_score'])
    y = data['exam_score']
    X = pd.get_dummies(X, drop_first=True)

    r2_list = []
    for _ in range(3):
        X_sh, y_sh = shuffle(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_sh, y_sh, test_size=0.1)
        model = LinearRegression()
        model.fit(X_train, y_train)
        r2_list.append(r2_score(y_test, model.predict(X_test)))
    return np.mean(r2_list)


# ==========================================
# TASK 2: ATTRIBUTE GROUPING (ANOVA ANALYSIS)
# ==========================================
final_results = []

# --- Group 1: Age Binning ---
# Teenager (0-19), Adult (20-49), Older (50+)
df_age = df_clean.copy()
df_age['age_group'] = pd.cut(df_age['age'], bins=[0, 19, 49, 100], labels=['Teenager', 'Adult', 'Older'])
df_age = df_age.drop(columns=['age'])  # Drop original numeric age
acc_age = evaluate_model_3_times(df_age)
final_results.append({'Grouped Attribute': 'Age (Teen/Adult/Older)', 'Avg Accuracy (R2)': acc_age})

# --- Group 2: Study Hours Binning ---
# Low (<5h), Medium (5-15h), High (>15h)
df_study = df_clean.copy()
df_study['study_level'] = pd.cut(df_study['study_hours'], bins=[0, 5, 15, float('inf')],
                                 labels=['Low', 'Medium', 'High'])
df_study = df_study.drop(columns=['study_hours'])
acc_study = evaluate_model_3_times(df_study)
final_results.append({'Grouped Attribute': 'Study Hours (Low/Med/High)', 'Avg Accuracy (R2)': acc_study})

# --- Group 3: Attendance Binning ---
# Poor (<50%), Average (50-80%), Good (>80%)
df_att = df_clean.copy()
df_att['att_level'] = pd.cut(df_att['class_attendance'], bins=[0, 50, 80, 100], labels=['Poor', 'Average', 'Good'])
df_att = df_att.drop(columns=['class_attendance'])
acc_att = evaluate_model_3_times(df_att)
final_results.append({'Grouped Attribute': 'Attendance (Poor/Avg/Good)', 'Avg Accuracy (R2)': acc_att})

# ==========================================
# FINAL OUTPUT TABLE
# ==========================================
summary_df = pd.DataFrame(final_results)
print("-" * 60)
print("TASK 2 & 3: ANOVA FEATURE GROUPING EVALUATION")
print("-" * 60)
print(summary_df.to_string(index=False, float_format="%.4f"))
print("-" * 60)
print("Conclusion: Grouping continuous data into categories helps in identifying")
print("practical student segments for the Early Warning System (EWS).")