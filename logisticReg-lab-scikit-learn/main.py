'''
The nbaallelo_log file contains data on 126314 NBA games from 1947 to 2015. The dataset includes the features pts, elo_i, win_equiv, and game_result. Using the csv file nbaallelo_log.csv and scikit-learn's LogisticRegression() function, construct a logistic regression model to classify whether a team will win or lose a game based on the team's elo_i score.

Create a binary feature win for game_result with 0 for L and 1 for W
Use the LogisticRegression() function to construct a logistic regression model with win as the target and elo_i as the predictor
Print the weights and intercept of the fitted model
Find the proportion of instances correctly classified
Note: Use ravel() from numpy to flatten the second argument of LogisticRegression.fit()into a 1-D array.

Ex: If the program uses the file nbaallelo_small.csv, which contains 100 instances, the output is:

w1: [[3.64194406e-06]]
w0: [-2.80257471e-09]
0.5
'''

# Import the necessary libraries
# Your code here
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

# Load nbaallelo_log.csv into a dataframe
# Your code here
NBA = pd.read_csv('nbaallelo_log.csv').dropna()
#print(NBA.head())

# Create binary feature for game_result with 0 for L and 1 for W
# Your code here
NBA['win'] = NBA['game_result'].replace(to_replace = ['L', 'W'], value = [int(0), int(1)])

# Store relevant columns as variables
X = NBA[['elo_i']]
y = NBA[['win']]

# Initialize and fit the logistic model using the LogisticRegression() function
# Your code here
logisticModel = LogisticRegression()
logisticModel = logisticModel.fit(X, np.ravel(y))

# Print the weights for the fitted model
print('w1:', logisticModel.coef_)

# Print the intercept of the fitted model
print('w0:', logisticModel.intercept_)

# Find the proportion of instances correctly classified
score = logisticModel.score(X, np.ravel(y))
print(round(score, 3))