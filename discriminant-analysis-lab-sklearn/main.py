'''
The hawks.csv file contains body measurements including age, sex, wing, weight, bill length (culmen) and talon length (hallux) for a sample of hawks observed near Iowa City, Iowa. The data was collected by students and faculty at Cornell College over a 10-year period. There are three species represented in this dataset: Cooper's hawk, red-tailed hawk, and sharp-shinned hawk. Using the csv file hawks.csv and scikit-learn's LinearDiscriminantAnalysis() function, construct a linear discriminant analysis model to classify a hawk's species based on the hawk's wing, weight, and culmen.

Import the necessary modules for linear discriminant analysis
Create dataframe X with features Wing, Weight, and Culmen
Create dataframe y with feature Species
Standardize the input features
Initialize a linear discriminant model with n_components=2
Fit the model
Print the discriminant intercepts and weight
Find the predicted species for the user-input standardized values for wing, weight, and culmen
Note: The predict() method can be used to find the prediction for a single instance with elements f1, f2â¦ as predict([[f1, f2, ...]]).

Ex: If the input is:

1.1
-0.5
0.9
the output should be:

[ -6.44709653  -4.41864135 -16.6109183 ]
[[ -5.68553588  -3.29948557  -1.68649648]
 [  4.19219327   1.65164801   5.36308861]
 [ -7.78302592  -2.77968595 -11.46863915]]
Predicted species is  ['RT']
'''

# Import packages and functions
# Your code here

import numpy as np 
import pandas as pd 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# Input standardized feature values for a sample instance
wing = float(input())
weight = float(input())
culmen = float(input())

# Load the Hawks dataset
# Your code here
hawks = pd.read_csv('hawks.csv').dropna()

# Define input features and output features
X = hawks[['Wing', 'Weight', 'Culmen']]# Your code here
y = hawks[['Species']] # Your code here

# Standardize input features
# Your code here
scalar = StandardScaler()
standardized_X = scalar.fit_transform(X)


# Initialize a linear discriminant model
# Your code here
lda = LinearDiscriminantAnalysis(n_components=2)


# Fit the model
# Your code here
lda.fit(standardized_X, np.ravel(y))

# Discriminant intercepts
intercept = lda.intercept_ # Your code here
print(intercept)

# Discriminant weights
weights = lda.coef_ # Your code here
print(weights)

# Calculate prediction
#to_predict_input = np.array([[wing, weight, culmen]])
#standardized_input = scalar.transform(to_predict_input)

preds = lda.predict([[wing, weight, culmen]]) # Your code here
print("Predicted species is ", preds)