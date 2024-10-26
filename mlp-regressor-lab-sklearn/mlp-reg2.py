'''
The diamonds.csv dataset contains the price, cut, color, and other characteristics of a sample of diamonds. The dataframe X contains all the features except cut, color, clarity, and price. The dataframe y contains the feature price.

Define a standardization scaler and apply the scaler to Xtrain and Xtest.
Initialize and fit a multilayer perceptron regressor with with random_state=42, three hidden layers of 50 nodes each, an adaptive learning rate of 0.01, a batch size of 100, and a maximum of 300 iterations.
Print the R-squared scores for the training data and the testing data, rounded to the fourth decimal place.
Ex: If random_state=123 is used instead of random_state=42, the output is:

Score for the training data:  0.8627
Score for the testing data:  0.8761
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

diamonds = pd.read_csv('diamonds.csv')
diamond_sample = diamonds.sample(1000, random_state=123)

X = diamond_sample.drop(columns=['cut', 'color', 'clarity', 'price'])
y = diamond_sample[['price']]

# Split data into train and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=123)

# Define a standardization scaler to transform values
# Your code here
scaler = StandardScaler()

# Apply scaler
Xtrain = scaler.fit_transform(Xtrain) # Your code here
Xtest = scaler.transform(Xtest) # Your code here

# Initialize a multilayer perceptron regressor with random_state=42, three hidden layers of 50 nodes each, 
# an adaptive learning rate of 0.01, a batch size of 100, and a maximum of 300 iterations
mlpDiamond = MLPRegressor(random_state=42, hidden_layer_sizes=(50, 50, 50), learning_rate_init=0.01, batch_size=100, max_iter=300) # Your code here

# Fit the model to the training data
# Your code here
mlpDiamond.fit(Xtrain, np.ravel(ytrain))

# Print the R-squared score for the training data
print("Score for the training data: ", round(mlpDiamond.score(Xtrain, np.ravel(ytrain)), 4))

# Print the R-squared score for the testing data
print("Score for the testing data: ", round(mlpDiamond.score(Xtest, np.ravel(ytest)), 4))