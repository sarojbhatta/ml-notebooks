"""
The taxis dataset contains information on taxi journeys during March 2019 in New York City. The data includes time, number of passengers, distance, taxi color, payment method, and trip locations. Use sklearn's cross_validate() function to fit a linear regression model and a k-nearest neighbors regression model with 10-fold cross-validation.

Create dataframe X with the feature distance.
Create dataframe y with the feature fare.
Split the data into 80% training, 10% validation and 10% testing sets, with random_state = 42.
Initialize a linear regression model.
Initialize a k-nearest neighbors regression model with k = 3.
Define a set of 10 cross-validation folds with random_state=42.
Fit the models with cross-validation to the training data, using the default performance metric.
For each model, print the test score for each fold, as well as the mean and standard deviation for the model.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

taxis = pd.read_csv("taxis.csv")

# Create dataframe X with the feature distance
X = taxis[['distance']]# Your code here
# Create dataframe y with the feature fare
y = taxis['fare']# Your code here

# Set aside 10% of instances for testing
# Your code here
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Split training again into 80% training and 10% validation
# Your code here
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1111, random_state=42)

# Initialize a linear regression model
SLRModel = LinearRegression()# Your code here
# Initialize a k-nearest neighbors regression model with k = 3
knnModel = KNeighborsRegressor(n_neighbors=3)# Your code here

# Define a set of 10 cross-validation folds with random_state=42
kf = KFold(n_splits=10, shuffle=True, random_state=42)# Your code here

# Fit k-nearest neighbors with cross-validation to the training data
knnResults = cross_validate(knnModel, X_train, y_train, cv=kf)# Your code here

# Find the test score for each fold
knnScores = knnResults['test_score']# Your code here
print('k-nearest neighbor scores:', knnScores.round(3))

# Calculate descriptive statistics for k-nearest neighbor model
print('Mean:', knnScores.mean().round(3))
print('SD:', knnScores.std().round(3))

# Fit simple linear regression with cross-validation to the training data
SLRModelResults = cross_validate(SLRModel, X_train, y_train, cv=kf) # Your code here

# Find the test score for each fold
SLRScores = SLRModelResults['test_score']# Your code here
print('Simple linear regression scores:', SLRScores.round(3))

# Calculate descriptive statistics simple linear regression model
print('Mean:', SLRScores.mean().round(3))
print('SD:', SLRScores.std().round(3))