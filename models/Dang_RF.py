import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# ==========================================================
# LOAD DATA
# ==========================================================

TRAIN_FILE = 'data/train_91_norm.csv'
TEST_FILE = 'data/test_91_norm.csv'

print("Đang tải dữ liệu...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# ==========================================================
# ENCODE CATEGORICAL
# ==========================================================

full_df = pd.concat([train_df, test_df], axis=0)
full_df = pd.get_dummies(full_df)

train_df = full_df.iloc[:len(train_df), :]
test_df = full_df.iloc[len(train_df):, :]

X_train = train_df.drop(columns=['exam_score'])
y_train = train_df['exam_score']

X_test = test_df.drop(columns=['exam_score'])
y_test = test_df['exam_score']

# ==========================================================
# RANDOM SEARCH SIÊU RỘNG
# ==========================================================

print("Đang tuning RandomForest...")

param_dist = {
    'n_estimators': [500, 1000, 1500, 2000, 3000],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30,                # thử 30 tổ hợp mạnh
    cv=cv,
    scoring='neg_root_mean_squared_error',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Best parameters:", random_search.best_params_)

# ==========================================================
# TRAIN MODEL TỐT NHẤT
# ==========================================================

best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n>>> RMSE sau tuning: {rmse:.4f}")

if rmse < 8:
    print("🔥 SUCCESS: RandomForest đã xuống dưới 8!")
else:
    print("⚠ Vẫn chưa xuống 8.")
