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

houseDat = [-1.38872698, 0.06678801, 0.40305024]

# Initialize a SVR model to training data
svr_reg = SVR(kernel='poly', C=7.2, gamma=1.2, epsilon=0.95) # Your code goes here

# Fit the model
svr_reg.fit(X_train, np.ravel(y_train))# Your code goes here

# Print the predicted values of the output feature using houseDat
houseDat = np.reshape(houseDat, (-1, 1))
scaled_houseDat = scaler.transform(houseDat)

predictions = svr_reg.predict(houseDat) # Your code goes here
print(predictions) 
