'''
The diamond dataset contains the price, cut, color, and other characteristics of a sample of nearly 54,000 diamonds. This data can be used to predict the price of a diamond based on its characteristics. Use sklearn's GridSearchCV() function to train and evaluate an elastic net model over a hyperparameter grid.

Create dataframe X with the features carat and depth.
Create dataframe y with the feature price.
Split the data into 80% training and 20% testing sets, with random_state = 42.
Initialize an elastic net model with random_state = 0.
Create a tuning grid with the hyperparameter name alpha and the values 0.1, 0.5, 0.9, 1.0.
Use GridSearchCV() with cv=10 to initialize and fit a tuning grid to the training data.
Print the mean testing score for each fold and the best parameter value.
Ex: If random_state=123 is used to split the data, the output is:

Mean testing scores: [0.84882287 0.81661033 0.76854838 0.75592476]
Best estimator: ElasticNet(alpha=0.1, random_state=0)
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

diamonds = pd.read_csv('diamonds.csv')

# Create dataframe X with the features carat and depth
X = diamonds[['carat', 'depth']]# Your code here
# Create dataframe y with the feature price
y = diamonds['price']# Your code here

# Create training/testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)# Your code here

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize elastic net model
ENModel = ElasticNet(random_state=0)# Your code here

# Create tuning grid
alpha = {'alpha':[0.1, 0.5, 0.9, 1.0]}# Your code here

# Initialize tuning grid and fit to training data
ENTuning = GridSearchCV(ENModel, alpha, cv=10)# Your code here
ENTuning.fit(X_train, y_train)# Your code here

# Mean testing score for each lambda and best model
print('Mean testing scores:', ENTuning.cv_results_['mean_test_score']) # Your code here
print('Best estimator:', ENTuning.best_estimator_) # Your code here