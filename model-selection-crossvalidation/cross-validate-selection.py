'''
Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. Using a computer vision process, 16 features of each bean were extracted.

Initialize a Gaussian naive Bayes model NBmodel with two components.
Fit the model with cross validation, using kf as the number of folds.
'''

# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

# Load the dry beans dataset
beans = pd.read_csv('Dry_Bean_Data.csv')

X = beans[['Eccentricity', 'Extent']]
y = beans[['Class']]

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a set of 10 cross-validation folds
kf = KFold(n_splits=10, random_state=69, shuffle=True)

# Initialize the linear discriminant analysis model with two components
modelLDA = LinearDiscriminantAnalysis(n_components=2)

# Fit linear discriminant analysis model with cross-validation
modelLDAResults = cross_validate(modelLDA, X, np.ravel(y), cv=kf)

LDAscores = modelLDAResults['test_score']

# View accuracy for each fold
print('Linear discriminant analysis scores:', LDAscores.round(3))

# Calculate descriptive statistics
print('Mean:', LDAscores.mean().round(3))
print('SD:', LDAscores.std().round(3))

# Initialize the Gaussian naive Bayes model
NBmodel = GaussianNB() # Your code goes here

# Fit Gaussian naive Bayes model with cross-validation
NBresults = cross_validate(NBmodel, X, np.ravel(y), cv=kf) # Your code goes here

NBBeanScores = NBresults['test_score']

# View accuracy for each fold
print('Gaussian naive Bayes scores:', NBBeanScores.round(3))

# Calculate descriptive statistics
print('Mean:', NBBeanScores.mean().round(3))
print('SD:', NBBeanScores.std().round(3))