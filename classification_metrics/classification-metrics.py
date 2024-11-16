'''
This dataset contains body measurements including age, sex, wing, weight, bill length (culmen) and talon length (hallux) for a sample of hawks observed near Iowa City, Iowa. The data was collected by students and faculty at Cornell College over a 10-year period. There are three species represented in this dataset: Cooper's hawk, red-tailed hawk, and sharp-shinned hawk.

Use the sklearn.metrics module to calculate accuracy for a Gaussian Naive Bayes model.
Use the sklearn.metrics module to calculate kappa for a Gaussian Naive Bayes model.
'''

# Import packages and functions
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Load the hawks dataset
hawks = pd.read_csv('hawks.csv')

# Define input and output features
X = hawks[['Weight', 'Hallux']]
y = hawks[['Species']]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize and fit a Gaussian naive Bayes model
NBModel = GaussianNB()
NBModel.fit(X, np.ravel(y))

# Calculate the predictions for each instance in X
nbPred = NBModel.predict(X)

# Calculate the accuracy
accuracy = metrics.accuracy_score(y, nbPred)# Your code goes here

# Calculate kappa
kappa = metrics.cohen_kappa_score(y, nbPred)# Your code goes here

print('GaussianNB model accuracy:', round(accuracy, 3))
print('GaussianNB model kappa:', round(kappa, 3))