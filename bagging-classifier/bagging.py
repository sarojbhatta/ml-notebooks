'''
This dataset contains information on chemical properties of wine from three cultivars of grapes.

Initialize a bagging classifier with a k-nearest neighbors model with k=7 as the base model, 12 estimators, and random_state=83.
'''

# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier


# Load the wines dataset
wines = pd.read_csv('wines.csv')

# Select input and output features
X = wines[['Alcalinity of ash', 'Proanthocyanins', 'Flavanoids']]
y = wines[['Cultivars']]

# Initialize a bagging classifier with a k-nearest neighbors model as a base    
# model and 12 estimators
wineBagging = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=7), n_estimators=12, random_state=83) # Your code goes here

# Fit bagging classifier to X and y
wineBagging.fit(X, np.ravel(y))

# Calculate the accuracy score
score = wineBagging.score(X,y)
print(score)