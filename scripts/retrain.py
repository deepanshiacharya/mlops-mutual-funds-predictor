import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import pickle
import os

# Load your original data (replace with your actual data loading)
df = pd.read_csv(r'C:\Users\DAIICT D\Desktop\202418015\MLops\scripts\Mutual_fund Data.csv')  # Replace with your dataset path

# Identify categorical columns (object dtypes)
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()  # e.g., ['Category', 'Risk']

# Create and save separate encoders for each categorical
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"Encoded {col} with classes: {le.classes_}")

# Save encoders as dict
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print("Encoders saved as label_encoders.pkl")

# Numerical columns (already numeric)
numerical_columns = [col for col in df.columns if col not in categorical_columns + ['3 Year Return']]

# Target
X = df.drop('3 Year Return', axis=1)
y = df['3 Year Return']

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Train model (your code)
rf_model = RandomForestRegressor()
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(x_train_scaled, y_train)

best_model = grid_search_rf.best_estimator_

# Evaluate
y_pred = best_model.predict(x_test_scaled)
print("R2:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Save model as 'randomforest.pkl'
joblib.dump(best_model, 'models/randomforest.pkl')
print("Model saved as randomforest.pkl")