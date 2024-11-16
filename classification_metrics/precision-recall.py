'''
This dataset contains data on field goal distance and outcome (whether or not the kick was successful) during the 2000 through 2008 NFL seasons.

Use the sklearn.metrics module to calculate precision for a Gaussian Naive Bayes model.
Use the sklearn.metrics module to calculate recall for a Gaussian Naive Bayes model.
'''


# Import packages and functions
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Load the field goal dataset
fieldGoal = pd.read_csv('fg_attempt.csv')

# Define input and output features
X = fieldGoal[['Distance', 'ScoreDiffPreKick']]
y = fieldGoal[['Outcome']]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize and fit a Gaussian naive Bayes model
NBModel = GaussianNB()
NBModel.fit(X, np.ravel(y))

# Calculate the predictions for each instance in X
fgPred = NBModel.predict(X)

# Calculate the precision
precision = metrics.precision_score(y, fgPred) # Your code goes here

# Calculate the recall
recall = metrics.recall_score(y, fgPred) # Your code goes here

print('GaussianNB model precision:', round(precision, 3))
print('GaussianNB model recall:', round(recall, 3))