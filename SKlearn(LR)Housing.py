'''
Now use Sklearn — Compare with your version

Use the same dataset from above
from sklearn.linear_model import LinearRegression
Fit the model, get predictions, compute R² score
Compare sklearn's weights with the weights your scratch model found
Try on a real dataset: load Boston Housing or California Housing from sklearn.datasets
Check R² score — is the model any good?
'''

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