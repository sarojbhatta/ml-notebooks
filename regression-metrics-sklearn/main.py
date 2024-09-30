'''
The diamonds dataset contains the price, cut, color, and other characteristics of a sample of nearly 54,000 diamonds. This data can be used to predict the price of a diamond based on its characteristics. Using a sample of the dataset and scikit-learn's LinearRegression() function, predict the price of a diamond from the diamond's carat and table values. Using scikit-learn's metrics module, calculate the various regression metrics for the model.

Initialize and fit a multiple linear regression model.
Use the model to predict the prices of instances in X.
Calculate mean absolute error for the model.
Calculate mean squared error for the model.
Calculate root mean squared error for the model.
Calculate R-squared for the model.
Ex: If the user-input random state used to take the sample is:

123
the output should be:

MAE: 1045.203
MSE: 2447857.716
RMSE: 1564.563
R-squared: 0.856
'''


import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.linear_model import LinearRegression

# Input the random state
rand = int(input())

# Load sample set by a user-defined random state into a dataframe
diamonds = pd.read_csv('diamonds.csv').sample(n=500, random_state=rand)

# Define input and output features
X = diamonds[['carat', 'table']]
y = diamonds['price']

# Initialize and fit a multiple linear regression model
# Your code here
lr = LinearRegression()
lr.fit(X, y)

# Use the model to predict the classification of instances in X
mlrPredY = lr.predict(X)# Your code here

# Calculate mean absolute error for the model
mae = metrics.mean_absolute_error(y, mlrPredY)# Your code here
print("MAE:", round(mae, 3))

# Calculate mean squared error for the model
mse = metrics.mean_squared_error(y, mlrPredY)# Your code here
print("MSE:", round(mse, 3))

# Calculate root mean squared error for the model
rmse = np.sqrt(mse)# Your code here
print("RMSE:", round(rmse, 3))

# Calculate R-squared for the model
r2 = metrics.r2_score(y, mlrPredY)# Your code here
print("R-squared:", round(r2, 3))