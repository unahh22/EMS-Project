import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Correct data path
data_path = os.path.join(current_dir, "..", "data", "dataADY201m_cleaned_normalized.csv")

# Load data
data = pd.read_csv(data_path)

print("Dataset loaded successfully")
print("Shape:", data.shape)

# Example target column (change if needed)
target = data.columns[-1]

X = data.drop(columns=[target])
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)
