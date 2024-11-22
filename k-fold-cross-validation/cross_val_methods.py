'''
Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. Using a computer vision process, 16 features of each bean were extracted.

Fit a linear discriminant analysis model with 7-fold cross-validation.
Print the test score.
'''


# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate

# Load the dry beans dataset
beans = pd.read_csv('Dry_Bean_Data.csv')

X = beans[['Eccentricity', 'ConvexArea']]
y = beans[['Class']]

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the model
beansLDA = LinearDiscriminantAnalysis(n_components=2, store_covariance=True)

# Fit linear discriminant analysis model with 7-fold cross-validation
beansCV = cross_validate(estimator=beansLDA, X=X_scaled, y=np.ravel(y), cv=7) # Your code goes here

# Print test score
print('Test score:', beansCV['test_score'])