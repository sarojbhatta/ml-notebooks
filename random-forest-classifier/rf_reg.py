'''
This dataset contains data for college basketball teams during the 2015-2019 seasons, such as number of games won, rebound percentage, and shooting percentages. The dataset contains 24 features, including the team's postseason performance.

Calculate the feature importances from a fitted random forest regressor.
'''

# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the basketball dataset
basketball = pd.read_csv('cbb.csv')

# Create a new feature representing the fraction of games won
basketball['w_l'] = basketball['W']/basketball['G']

# Select input and output features
X = basketball.drop(columns=['G', 'W', 'YEAR', 'TEAM', 'CONF', 'POSTSEASON', 'w_l', 'FTRD', 'DRB', '2P_D'])
y = basketball[['w_l']]

# Initialize a random forest regressor with default parameters
rfBasketball = RandomForestRegressor(random_state=72)

# Fit random forest regressor to X and y
rfBasketball.fit(X, np.ravel(y))

# Calculate the feature importances
importances = rfBasketball.feature_importances_ # Your code goes here
print(importances)