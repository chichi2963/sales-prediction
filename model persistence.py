import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
df = pd.read_csv("Advertising.csv")

# Prepare the data
X = df.drop('sales', axis=1)
y = df['sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Split the testing set into validation and holdout sets
X_validation, X_holdout_test, y_validation, y_holdout_test = train_test_split(X_test, y_test, test_size=0.50, random_state=42)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=101, feature_names=X_train.columns)
model.fit(X_train, y_train)

# Save the models and feature names
joblib.dump(model, 'final_model.joblib')
joblib.dump(X_train.columns, 'col_names.joblib')

# Load the models and feature names
loaded_model = joblib.load('final_model.joblib')
loaded_col_names = joblib.load('col_names.joblib')

# Test the loaded model with feature names
loaded_model.set_params(feature_names=loaded_col_names)
prediction = loaded_model.predict([[230.1, 37.8, 69.2]])
print(f'Prediction for new data: {list(prediction)[0]:.2f}')