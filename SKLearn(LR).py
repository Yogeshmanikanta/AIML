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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Data
x = np.array([500,750,1000,1250,1500]).reshape(-1,1)
y = np.array([150,200,250,300,350])

# Model
model = LinearRegression()

# Train
model.fit(x, y)

# Predict
y_pred = model.predict(x)

# Results
print("Weight:", model.coef_)
print("Bias:", model.intercept_)

# Score
print("R² Score:", r2_score(y, y_pred))