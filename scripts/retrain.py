import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import pickle
import os

# Load data (replace with your actual CSV path)
df = pd.read_csv(r'C:\Users\DAIICT D\Desktop\202418015\MLops\scripts\processed_mutual_fund_data.csv')  # Update this



# Identify categorical columns (AMC is string, so include it)
categorical_columns = ['AMC', 'Category', 'Risk']

# Encode categoricals
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"Encoded {col} with classes: {le.classes_}")

# Save encoders
os.makedirs('models', exist_ok=True)
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print("Saved label_encoders.pkl")

# Features and target
X = df.drop('3 Year Return', axis=1)
y = df['3 Year Return']

# Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved scaler.pkl")

# Train Random Forest
rf_model = RandomForestRegressor()
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(x_train_scaled, y_train)

# Evaluate
best_model = grid_search_rf.best_estimator_
y_pred = best_model.predict(x_test_scaled)
print("Best params:", grid_search_rf.best_params_)
print("R2:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Save model
joblib.dump(best_model, 'models/randomforest.pkl')
print("Saved randomforest.pkl")