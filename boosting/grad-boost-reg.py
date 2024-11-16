'''
This dataset contains data for college basketball teams during the 2015-2019 seasons, such as number of games won, rebound percentage, and shooting percentages. There are a total of 24 features in the data, including the team's postseason performance.

Initialize a gradient boosting regressor, modelBoost, with a learning rate of 0.5 and random_state=89.
Initialize a second gradient boosting regressor basketballModel with a learning rate of 1.3 and random_state=89.
'''

# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Load the basketball dataset
basketball = pd.read_csv('cbb.csv')

# Create a new feature representing the fraction of games won
basketball["w_l"] = basketball['W']/basketball['G']
basketball = basketball.dropna()

# Select input and output features
X = basketball.drop(columns=['G', 'W', 'YEAR', 'TEAM', 'CONF', 'POSTSEASON', 'w_l', 
    'BARTHAG', 'TORD', 'ADJOE'])
y = basketball[['w_l']]

# Initialize a gradient boosting regressor with a learning rate of 0.5
modelBoost = GradientBoostingRegressor(learning_rate=0.5, random_state=89) # Your code goes here

# Fit gradient boosting regressor to X and y
modelBoost.fit(X, np.ravel(y))

# Initialize a gradient boosting regressor with a learning rate of 1.3
basketballModel = GradientBoostingRegressor(learning_rate=1.3, random_state=89)# Your code goes here

# Fit gradient boosting regressor to X and y
basketballModel.fit(X, np.ravel(y))

# Print accuracy scores
score1 = modelBoost.score(X, y)
score2 = basketballModel.score(X, y)

print("First score is", score1, "\nSecond score is", score2)