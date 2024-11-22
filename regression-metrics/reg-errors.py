'''
The World Happiness Report ranks 155 countries based on data from the Gallup World Poll. People completing the survey are asked to rate their current lives, which is averaged across countries to produce a rating on a scale from 0 (worst possible) to 10 (best possible).

Use the sklearn.metric module to calculate the mean absolute error for an elastic net regression model.
Use the sklearn.metric module to calculate the root mean squared error for an elastic net regression model.

'''


# Import packages and functions
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

# Load the world happiness dataset
happiness = pd.read_csv("world_happiness_2017.csv")

# Define input and output features
X = happiness[['family', 'generosity']].values.reshape(-1, 2)
y = happiness[['happiness_score']].values.reshape(-1, 1)

# Scale the input features
scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)

# Initialize and fit a Gaussian naive Bayes model
enModel = ElasticNet(alpha=1, l1_ratio=0.5)
enModel.fit(Xscaled, y)

# Calculate the predictions for each instance in X
predHappiness = enModel.predict(Xscaled)

# Calculate the mean absolute error
maeHappiness = metrics.mean_absolute_error(y, predHappiness) # Your code goes here

# Calculate the root mean squared error 
rmseHappiness = metrics.mean_squared_error(y, predHappiness, squared=False) # Your code goes here

print("Mean absolute error:", maeHappiness)
print("Root mean squared error:", rmseHappiness)