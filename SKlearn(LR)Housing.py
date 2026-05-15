import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = fetch_california_housing()

# Features and target
X = data.data
y = data.target

print("X shape:", X.shape)
print("y shape:", y.shape)

# Create model
model = LinearRegression()

# Train
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Accuracy
score = r2_score(y, y_pred)

print("R² Score:", score)

# Model parameters
print("Weights:", model.coef_)
print("Bias:", model.intercept_)