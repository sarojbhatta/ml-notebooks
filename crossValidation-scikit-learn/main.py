"""The taxis dataset contains information on taxi journeys during March 2019 in New York City. The data includes time, number of passengers, distance, taxi color, payment method, and trip locations. Use sklearn's cross_validate() function to fit a linear regression model with 15-fold cross-validation.

Create dataframe X with features passengers and distance.
Create dataframe y with feature fare.
Split the data into 70% training, 20% validation and 10% testing sets, with random_state = 42.
Initialize a linear regression model.
Fit the model with 15-fold cross-validation to the training data, using 'explained_variance' as the performance metric.
Print the explained variance for each fold.
Ex: If the file taxis_small.csv is used, the output is:

Test score: [ 0.73224305  0.99710401  0.07165374  0.99973244  0.77172554  0.63613485
  0.93052017 -0.96913429  0.51933292  0.99722418  0.95547656  0.34065158
  0.22046622  0.92815629  0.61043292]"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

taxis = pd.read_csv('taxis.csv')

# Create dataframe X with features passengers and distance
X = taxis[['passengers', 'distance']]# Your code here

# Create dataframe y with feature rate. 
y = taxis[['fare']]# Your code here

# Set aside 10% of instances for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)# Your code here

# Split training again into 70% training and 20% validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2222, random_state=42)# Your code here

# Initialize a linear regression model
linRegModel = LinearRegression()# Your code here

# Fit the model with 15-fold cross-validation to the training data, 
# using 'explained_variance' as the performance metric
cv_results = cross_validate(linRegModel, X_train, y_train, cv=15, scoring='explained_variance')# Your code here

# Print the explained variance for each fold
print("Test score:", cv_results['test_score'])