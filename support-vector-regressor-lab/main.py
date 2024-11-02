'''
The dataset mpg contains information on miles per gallon (mpg) and engine size for cars sold from 1970 through 1982. The dataset has the features mpg, cylinders, displacement, horsepower, weight, acceleration, model_year, origin, and name. Using the file mpg.csv and scikit-learn's SVR() function, fit linear, polynomial, and RBF support vector regressors to predict mpg from a user-defined input feature.

Import the correct packages and functions.
Split the data into 75% training data and 25% testing data. Set random_state=123.
Initialize and fit a linear support vector regressor with epsilon=0.2.
Initialize and fit a polynomial support vector regressor with epsilon=0.2, C=0.5, and gamma=0.7.
Initialize and fit an RBF support vector regressor with epsilon=0.2, C=0.5, and gamma=0.7.
Print the coefficient of determination of prediction for each support vector regressor using the score() method.

'''
# Import the necessary packages
# Your code here
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


mpg = pd.read_csv('mpg.csv')

# User-defined input feature
X = mpg[[input()]]

# Output feature: mpg
y = mpg[['mpg']]

# Create training and testing data with 75% training data and 25% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123) # Your code here

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

# Initialize and fit a linear svr model to training data with epsilon = 0.2
eps = 0.2 # Your code here
svr_lin = SVR(kernel='linear', epsilon=eps) # Your code here
svr_lin.fit(X_train, np.ravel(y_train)) # Your code here

# Initialize and fit an svr model using a poly kernel with epsilon=0.2, C=0.5, and gamma=0.7
svr_poly = SVR(kernel='poly', epsilon=0.2, C=0.5, gamma=0.7) # Your code here
svr_poly.fit(X_train, np.ravel(y_train))# Your code here

# Initialize and fit an svr model using an rbf kernel with epsilon=0.2, C=0.5, and gamma=0.7
svr_rbf = SVR(kernel='rbf', epsilon=0.2, C=0.5, gamma=0.7) # Your code here
svr_rbf.fit(X_train, np.ravel(y_train)) # Your code here

# Print the coefficients of determination for each model
lin_score = svr_lin.score(X_test, y_test) # Your code here
print('Linear model:', np.round(lin_score, 3))

poly_score = svr_poly.score(X_test, y_test) # Your code here
print('Polynomial model:', np.round(poly_score, 3))

rbf_score = svr_rbf.score(X_test, y_test) # Your code here
print('RBF model:', np.round(rbf_score, 3))


'''
Input
weight

Expected output
Linear model: 0.699
Polynomial model: 0.432
RBF model: 0.709


Input
model_year

Expected output
Linear model: 0.283
Polynomial model: 0.217
RBF model: 0.274


Input
acceleration

Expected output
Linear model: 0.059
Polynomial model: -0.124
RBF model: 0.175

'''