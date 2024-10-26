'''
Initialize and fit a support vector regression model, svr_reg, that predicts the 
standardized value of a home's median sale price with the standardized value of 
number of sales as an input feature. 
Use a polynomial kernel with degree = 3, C = 1, gamma = 2.3, and epsilon = 0.7.
Calculate and print the coefficient of determination r2 using the test set.
'''

# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = pd.read_csv('txhousing.csv')

X = housing[['sales']]
y = housing[['median']]

# Create training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Scale the input and output features from the training set
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = scaler.fit_transform(y_train)
X_test = scaler.fit_transform(X_train)
y_test = scaler.fit_transform(y_train)

# Initialize a SVR model to training data
svr_reg = SVR(kernel='poly', degree=3, C=1, gamma=2.3, epsilon=0.7) # Your code goes here

# Fit the model
# Your code goes here
svr_reg.fit(X_train, np.ravel(y_train))

# Find and print the coefficient of determination using the test data
# Your code goes here
r2_score = svr_reg.score(X_test, np.ravel(y_test))
print(r2_score)