'''
The nbaallelo_log.csv file contains data on 126314 NBA games from 1947 to 2015. The dataset includes the features pts, elo_i, win_equiv, and game_result. Using a sample of the csv file nbaallelo_log.csv and scikit-learn's LogisticRegression() function, construct a logistic regression model to classify whether a team will win or lose a game based on the team's elo_i score. Using scikit-learn's metrics module, calculate the various classification metrics for the model.

Build a logistic model with default parameters, fit to the input and output features X and y.
Use the model to predict the classification of instances in X.
Calculate the confusion matrix for the model.
Calculate the accuracy for the model.
Calculate the precision for the model.
Calculate the recall for the model.
Calculate kappa for the model.
'''

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Input the random state
rand = int(input())

# Load sample set by a user-defined random state into a dataframe. 
NBA = pd.read_csv("nbaallelo_log.csv").sample(n=500, random_state=rand)

# Create binary feature for game_result with 0 for L and 1 for W
NBA['win'] = NBA['game_result'].replace(to_replace = ['L','W'], value = [int(0), int(1)])

# Store relevant columns as variables
X = NBA[['elo_i']]
y = NBA[['win']]

# Build logistic model with default parameters, fit to X and y
# Your code here
logisticModel = LogisticRegression()
logisticModel.fit(X, np.ravel(y))

# Use the model to predict the classification of instances in X
logPredY = logisticModel.predict(X)# Your code here

# Calculate the confusion matrix for the model
confMatrix = metrics.confusion_matrix(np.ravel(y), logPredY)# Your code here
print("Confusion matrix:\n", confMatrix)

# Calculate the accuracy for the model
accuracy = metrics.accuracy_score(np.ravel(y), logPredY)# Your code here
print("Accuracy:", round(accuracy,3))

# Calculate the precision for the model
precision = metrics.precision_score(np.ravel(y), logPredY)# Your code here
print("Precision:", round(precision,3))

# Calculate the recall for the model
recall = metrics.recall_score(np.ravel(y), logPredY)# Your code here
print("Recall:", round(recall, 3))

# Calculate kappa for the model
kappa = metrics.cohen_kappa_score(np.ravel(y), logPredY)# Your code here
print("Kappa:", round(kappa, 3))