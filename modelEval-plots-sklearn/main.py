'''
The diamonds dataset contains the price, cut, color, and other characteristics of a sample of nearly 54,000 diamonds. This data can be used to predict the price of a diamond based on its characteristics. Using a sample of the dataset and scikit-learn's LinearRegression() function, predict the price of a diamond from the user-input features. Using scikit-learn and matplotlib, create plots to evaluate the regression model.

Initialize and fit a multiple linear regression model.
Use the model to predict the prices of instances in X.
Compute the prediction errors.
Plot prediction errors vs predicted values. Label the x-axis as 'Predicted' and the y-axis as 'Prediction error'. Include a dashed line at y=0.
Generate a partial dependence display for both input features.
Ex: If the input features are:

carat
table
The output should be:

A scatter plot with Predicted along the x-axis, which extends from 0 to 18000, and Prediction error along the y-axis, which extends from -4000 to 4000. The points make a rough v-shape. 

Two partial dependence plots with Partial dependence along the y-axis, which extends from 0 to 18000. The left plot has an x-axis labeled carat, which extends from 0 to 2.5, with 6 decile lines between 0 and 1, and 3 decile lines between 1 and 2. A linear line is shown starting at roughly (0, 0) and increasing to roughly (2.5, 18000). The right plot has an x-axis labeled table, which extends from 55 to 64, with 9 decile lines between 60 and 63. A linear line is shown starting at roughly (56, 4000) and increasing to roughly (63, 5000).  

MAE: 929.628
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.inspection import PartialDependenceDisplay
from sklearn import metrics

# Load diamonds sample into dataframe
diamonds = pd.read_csv('diamonds.csv').sample(n=50, random_state=42)

# Get user-input features
feature1 = input()
feature2 = input()

# Define input and output features
X = diamonds[[feature1, feature2]]
y = diamonds['price']

# Initialize and fit a multiple linear regression model
# Your code here
lr = LinearRegression()
lr.fit(X, y)

# Use the model to predict the classification of instances in X
mlrPredY = lr.predict(X)# Your code here

# Compute prediction errors
mlrPredError = y - mlrPredY# Your code here

# Plot prediction errors vs predicted values. Label the x-axis as 'Predicted' and the y-axis as 'Prediction error'
fig = plt.figure()
# Your code here
plt.scatter(mlrPredY, mlrPredError)
plt.xlabel('Predicted')
plt.ylabel('Prediction error')

# Add dashed line at y=0
# Your code here
plt.plot([min(mlrPredY), max(mlrPredY)], [0, 0], linestyle = '--')

plt.savefig('predictionError.png')

# Generate a partial dependence display for both input features
# Your code here
PartialDependenceDisplay.from_estimator(lr, X, features=[feature1, feature2])

plt.savefig('partialDependence.png')

# Calculate mean absolute error for the model
mae = metrics.mean_absolute_error(y, mlrPredY)
print("MAE:", round(mae, 3))