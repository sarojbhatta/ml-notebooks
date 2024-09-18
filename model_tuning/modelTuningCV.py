# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Load the dry beans dataset
beans = pd.read_csv('Dry_Bean_Data.csv')

X = beans[['Compactness', 'ConvexArea', 'Solidity']]
y = beans[['Class']]

# Create training/testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize k-nearest neighbors model
kNNBean = KNeighborsClassifier()

# Create tuning grid
k = {'n_neighbors': [2, 3, 4, 5]}

# Initialize tuning grid 
tuningGrid = GridSearchCV(kNNBean, k, cv=8)

# Fit grid to training data
tuningGrid.fit(X_train, np.ravel(y_train))

print('Best parameter:', tuningGrid.best_params_) # Your code goes here