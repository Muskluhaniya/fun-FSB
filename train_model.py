import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the new dataset
df = pd.read_csv("data/b_data.csv")

# Define features and target variable
X = df.drop(columns=["BF_Score"])  # Features
y = df["BF_Score"]  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model MSE: {mse:.2f}")

# Save model
joblib.dump(model, "model/b_scoring_model.pkl")
print("✅ Model retrained and saved successfully!")
